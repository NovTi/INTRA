#!/usr/bin/env python3
"""Run one attention backend over the six SCROLLS tasks and write predictions (greedy decoding).

Every backend -- dense, INTRA (train-matched `intra_llama`, or the kernel-semantics `intra`), MoBA,
NSA, MInference, XAttention -- goes through this one file with the same prompts and per-task
generation budgets (taken from the eval file built by build_data.py). Score with score.py.

Sharding is by record index (--shard i --nshard N), one process per GPU.
"""
import argparse, json, os, sys, time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
try:
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
except ImportError:
    # transformers<4.48 (the MInference/XAttention environment) has no pluggable-attention
    # registry. Only the intra/moba/nsa backends need it.
    ALL_ATTENTION_FUNCTIONS = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import truncate_prediction  # noqa: E402


def build_model(args):
    import transformers
    # transformers 5.x renamed from_pretrained's torch_dtype= to dtype=; 4.46 only accepts the old name.
    _dt = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
    kw = {_dt: torch.bfloat16, "device_map": None}
    post = None                      # called after from_pretrained, before .cuda()
    if args.backend == "dense":
        impl = "sdpa"
    elif args.backend == "minference":
        # Training-free dynamic-sparse baseline (Jiang et al.). Needs transformers 4.46.x because
        # MInference monkey-patches modeling internals that moved in transformers 5.x. The model
        # loads as plain sdpa; the patch then wraps each layer's PREFILL attention in place. Decode
        # stays dense (kv_type=dense).
        from minference import MInference as _MInference
        _minf = _MInference("minference", args.minference_config_name)

        def post(m):
            patched = _minf(m)
            if patched is not m:
                raise RuntimeError("MInference returned a different model object; wiring assumes "
                                   "in-place patching -- update build_model to keep the return")
            from transformers.utils import is_flash_attn_2_available
            if not is_flash_attn_2_available():
                # Without flash_attn, transformers 4.46 never defines its flash globals and
                # MInference's decode / short-prefill fallback dies on NameError. Rebind that
                # fallback to SDPA: it only serves the DENSE paths (q_len==1 decode, sub-threshold
                # prefill) -- the sparse prefill runs MInference's own triton kernels -- so this
                # changes the kernel, not the math. bs=1 is asserted because the shim ignores
                # padding masks.
                def _sdpa_stand_in(q, k, v, attention_mask, query_length, *a, **kw):
                    assert q.shape[0] == 1, "SDPA stand-in assumes unpadded bs=1 eval"
                    qt, kt, vt = (t.transpose(1, 2) for t in (q, k, v))
                    if kt.shape[1] != qt.shape[1]:
                        rep = qt.shape[1] // kt.shape[1]
                        kt = kt.repeat_interleave(rep, dim=1)
                        vt = vt.repeat_interleave(rep, dim=1)
                    Lq, Lk = qt.shape[2], kt.shape[2]
                    sc = kw.get("softmax_scale")
                    if Lq == 1:
                        o = torch.nn.functional.scaled_dot_product_attention(qt, kt, vt, scale=sc)
                    elif Lq == Lk:
                        o = torch.nn.functional.scaled_dot_product_attention(
                            qt, kt, vt, is_causal=True, scale=sc)
                    else:
                        msk = torch.ones(Lq, Lk, dtype=torch.bool,
                                         device=qt.device).tril(diagonal=Lk - Lq)
                        o = torch.nn.functional.scaled_dot_product_attention(
                            qt, kt, vt, attn_mask=msk, scale=sc)
                    return o.transpose(1, 2)
                import sys as _sys
                import transformers.modeling_flash_attention_utils as _mfa
                _mfa._flash_attention_forward = _sdpa_stand_in
                n_bound = 0
                for _name, _mod in list(_sys.modules.items()):
                    if _name.startswith("minference") and hasattr(_mod, "_flash_attention_forward"):
                        _mod._flash_attention_forward = _sdpa_stand_in
                        n_bound += 1
                print(f"  flash_attn absent -> SDPA stand-in bound into {n_bound} minference "
                      f"module(s) (dense decode/fallback only)", flush=True)
            print(f"  MInference patched (config={args.minference_config_name}, kv=dense)", flush=True)
        impl = "sdpa"
    elif args.backend == "xattention":
        # Training-free (Xu et al.), official repo wiring: each layer's attention forward becomes
        # xattn's forward_eval -- prefill runs the antidiagonal block estimate + the official
        # block_sparse_attn kernel with the repo's calibrated per-layer/per-head thresholds
        # (stride 16 == the paper's S=16 config); decode goes through flashinfer. GQA is handled
        # by the repo's own repeat_kv. Runs in the same transformers 4.46 environment as
        # MInference, which is what the XAttention repo was written against.
        import types as _types
        from xattn.src.load_llama import forward_eval as _xattn_fwd, FastPrefillConfig as _XCfg

        def post(m):
            _xcfg = _XCfg(stride=args.xattn_stride)
            n = 0
            for lyr in m.model.layers:
                lyr.self_attn.fastprefillconfig = _xcfg
                lyr.self_attn.forward = _types.MethodType(_xattn_fwd, lyr.self_attn)
                n += 1
            print(f"  XAttention patched on {n} layers (stride={args.xattn_stride}, "
                  f"official thresholds, decode=flashinfer)", flush=True)
        impl = "sdpa"
    elif args.backend == "intra":
        from intra_token_attn import make_intra_attention
        ALL_ATTENTION_FUNCTIONS["intra"] = make_intra_attention(
            num_groups=args.intra_groups, back=args.intra_window,
            block=int(str(args.intra_block).split(',')[0]), n_sink=args.intra_sink)
        impl = "intra"
    elif args.backend == "intra_llama":
        # The TRAIN-MATCHED path: llama_intra.py is the module train_llama_distill.py trains
        # through. Its Window branch is a plain per-token sliding window of `window` keys, not the
        # kernel's 64-aligned block window implemented by intra_token_attn.py, so adapters trained with
        # train_llama_distill.py must be evaluated here.
        from llama_intra import install_intra_llama
        blocks = tuple(int(x) for x in str(args.intra_block).split(","))

        def post(m):
            n_s, n_w, n_d = install_intra_llama(
                m, num_groups=args.intra_groups, window=args.intra_window_flex,
                dense_debug=False, compile_flex=bool(args.compile_flex),
                n_sink=args.intra_sink, scatter_blocks=blocks,
                layer_pattern=args.layer_pattern,
                scatter_stripe=bool(args.intra_stripe), scatter_impl=args.intra_impl,
                scatter_n_offsets=args.intra_n_offsets)
            print(f"  INTRA(train-matched): {n_s} scatter / {n_w} window / {n_d} dense, "
                  f"window={args.intra_window_flex} sink={args.intra_sink} blocks={blocks}, "
                  f"stripe={bool(args.intra_stripe)} impl={args.intra_impl}",
                  flush=True)
        impl = "intra"
    elif args.backend == "moba":
        from moba_attn import make_moba_attention
        ALL_ATTENTION_FUNCTIONS["moba"] = make_moba_attention(
            args.moba_chunk, args.moba_topk, args.moba_density or None)
        impl = "moba"
    elif args.backend == "nsa":
        # Train-matched by construction: nsa_student.py imports this same nsa_attn.
        from nsa_attn import make_nsa_attention
        ALL_ATTENTION_FUNCTIONS["nsa"] = make_nsa_attention(
            cmp_block=args.nsa_cmp_block, cmp_stride=args.nsa_cmp_stride,
            sel_block=args.nsa_sel_block,
            n_sel=(args.nsa_n_sel if args.nsa_density <= 0 else None),
            density=(args.nsa_density or None), window=args.nsa_window)
        impl = "nsa"
    else:
        raise ValueError(f"unknown backend {args.backend}")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, attn_implementation=impl, **kw)
    if post is not None:
        post(model)
    if args.lora:
        # Same adapter shape the trainer built, then load the saved lora_ tensors. strict=False is
        # required (base weights are absent from the checkpoint), but a rank/target mismatch then
        # loads nothing and returns cleanly -- so count the matches and refuse zero.
        from peft import LoraConfig
        targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        model.add_adapter(LoraConfig(r=args.lora_rank, lora_alpha=args.lora_rank,
                                     init_lora_weights="gaussian", target_modules=targets))
        sd = torch.load(args.lora, map_location="cpu")
        missing, unexpected = model.load_state_dict(sd, strict=False)
        loaded = len(sd) - len(unexpected)
        print(f"  LoRA r={args.lora_rank}: {loaded}/{len(sd)} tensors matched "
              f"(unexpected={len(unexpected)}) from {args.lora}", flush=True)
        if loaded == 0:
            raise RuntimeError(f"LoRA {args.lora} matched no parameters -- wrong rank or targets")
    model = model.to("cuda").eval()
    # a wrong attn_implementation silently falls back to sdpa, which would make a "sparse" run dense
    got = getattr(model.config, "_attn_implementation", None)
    print(f"  attn_implementation requested={impl} actual={got}", flush=True)
    if got != impl:
        raise RuntimeError(f"backend not applied: asked for {impl}, model reports {got}")
    return tok, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--backend", default="dense",
                    choices=["dense", "moba", "intra", "intra_llama", "nsa", "minference",
                             "xattention"])
    ap.add_argument("--minference_config_name", default="meta-llama/Meta-Llama-3.1-8B-Instruct",
                    help="minference backend: HF name used to look up its published per-head config")
    ap.add_argument("--xattn_stride", type=int, default=16, choices=[4, 8, 16],
                    help="xattention backend: antidiagonal stride; picks the official per-layer/"
                         "per-head threshold table from the repo (16 = the paper's S=16)")
    ap.add_argument("--lora", default="", help="lora_final.pt from train_llama_distill.py")
    ap.add_argument("--lora_rank", type=int, default=16, help="must match the checkpoint's rank")
    ap.add_argument("--intra_window_flex", type=int, default=575,
                    help="intra_llama backend only: plain sliding-window width, as trained")
    ap.add_argument("--layer_pattern", default="SSWW", help="intra_llama backend only")
    ap.add_argument("--intra_stripe", type=int, default=0,
                    help="intra_llama backend only: shifted-stripe Scatter; must match training")
    ap.add_argument("--intra_n_offsets", type=int, default=0,
                    help="distinct stripe offsets (0 = intra_groups). MUST match training.")
    ap.add_argument("--intra_impl", default="loop", choices=["loop", "flex"],
                    help="intra_llama backend only: loop or flex (same math, block-sparse cost); "
                         "keep it at the value the adapter was trained with")
    ap.add_argument("--nsa_cmp_block", type=int, default=32, help="nsa backend: pooling block l")
    ap.add_argument("--nsa_cmp_stride", type=int, default=16, help="nsa backend: pooling stride d")
    ap.add_argument("--nsa_sel_block", type=int, default=64, help="nsa backend: selection block")
    ap.add_argument("--nsa_n_sel", type=int, default=0, help="nsa backend: fixed block count")
    ap.add_argument("--nsa_window", type=int, default=512, help="nsa backend: sliding window")
    ap.add_argument("--nsa_density", type=float, default=0.0,
                    help="nsa backend: NOMINAL density; 0.1438 measures ~0.126")
    ap.add_argument("--compile_flex", type=int, default=1, help="intra_llama backend only")
    ap.add_argument("--moba_chunk", type=int, default=2048)
    ap.add_argument("--moba_topk", type=int, default=3)
    ap.add_argument("--intra_groups", type=int, default=8,
                    help="intra backend: Scatter stride (8 in the original configuration)")
    ap.add_argument("--intra_window", type=int, default=512,
                    help="intra backend: window look-back; the 576-key span is this + the 64-aligned chunk")
    ap.add_argument("--intra_block", default="1",
                    help="Scatter group = (pos//block) %% groups. Comma list allowed for the "
                         "intra_llama backend (block sizes cycled over Scatter layers)")
    ap.add_argument("--intra_sink", type=int, default=0,
                    help="attention sinks kept visible to every query")
    ap.add_argument("--enable_thinking", type=int, default=0,
                    help="Qwen3 chat template: emit <think> blocks (off; they would consume the "
                         "generation budget)")
    ap.add_argument("--moba_density", type=float, default=0.0,
                    help="if >0, keep this FRACTION of blocks instead of a fixed top-k")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max_input_tokens", type=int, default=0, help="0 = no cap; else middle-drop")
    ap.add_argument("--chat_template", type=int, default=1)
    a = ap.parse_args()

    rows = []
    with open(a.data) as fh:
        for line in fh:
            r = json.loads(line)
            if "__meta__" in r:
                continue
            rows.append(r)
    rows = [r for i, r in enumerate(rows) if i % a.nshard == a.shard]
    if a.limit:
        rows = rows[: a.limit]

    print(f"shard {a.shard}/{a.nshard}: {len(rows)} records | backend={a.backend}"
          + (f" groups={a.intra_groups} window={a.intra_window} block={a.intra_block} "
             f"sink={a.intra_sink}" if a.backend == "intra" else "")
          + (f" chunk={a.moba_chunk} "
             + (f"density={a.moba_density}" if a.moba_density else f"topk={a.moba_topk}")
             if a.backend == "moba" else ""), flush=True)

    tok, model = build_model(a)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fout = open(a.out, "w")
    t0 = time.time()

    for i, r in enumerate(rows):
        prompt = r["prompt"]
        if a.chat_template:
            # Qwen3's template opens a <think> block unless told not to; Llama's takes no such
            # parameter, so pass it only where the template mentions it.
            kw = {}
            if "enable_thinking" in (getattr(tok, "chat_template", None) or ""):
                kw["enable_thinking"] = bool(a.enable_thinking)
            prompt = tok.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False,
                add_generation_prompt=True, **kw)
        ids = tok(prompt, return_tensors="pt", add_special_tokens=not a.chat_template).input_ids
        if a.max_input_tokens and ids.shape[1] > a.max_input_tokens:
            # middle-drop, as the repo's harness does: the instruction header and the question both
            # sit at the ends, so dropping the middle of the document is the only safe cut
            h = a.max_input_tokens // 2
            ids = torch.cat([ids[:, :h], ids[:, -(a.max_input_tokens - h):]], dim=1)
        ids = ids.to("cuda")

        with torch.no_grad():
            out = model.generate(
                ids, max_new_tokens=r["max_new_tokens"], do_sample=False,
                num_beams=1, temperature=None, top_p=None, top_k=None,
                pad_token_id=tok.eos_token_id,
            )
        text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        fout.write(json.dumps({
            "task": r["task"], "id": r.get("id"), "pid": r.get("pid"),
            "pred": truncate_prediction(text), "pred_raw": text,
            "answer": r["answer"], "metric": r["metric"], "n_tokens": r.get("n_tokens"),
            "choices": r.get("choices"),
        }) + "\n")
        fout.flush()
        if (i + 1) % 10 == 0 or i == 0:
            el = time.time() - t0
            print(f"  [{i+1}/{len(rows)}] {el:.0f}s  {el/(i+1):.1f}s/rec  "
                  f"eta {(len(rows)-i-1)*el/(i+1)/60:.0f}m", flush=True)

    fout.close()
    print(f"done in {(time.time()-t0)/60:.1f} min -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
