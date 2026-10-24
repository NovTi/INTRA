"""
niah_eval.py -- needle-in-a-haystack retrieval for dense / INTRA / MoBA / NSA (optionally with a
LoRA adapter from train_llama_distill.py), plus the training-free MInference and XAttention
baselines.

Protocol (paper Table 4a/5/15/16): PG19 haystack, needle "The special magic {city} number is:
{num}." planted at each depth, the question appended in a chat turn, 9 depths (0.1..0.9) x 10
trials per length, lengths 4K/8K/16K/32K (one process per length).

Scoring is TEACHER-FORCED (single prefill forward, no KV cache): the gold answer is appended and
we check whether its numeric tokens are all argmax (exact), whether the first one is (first-tok),
and their mean log-prob. The reported NIAH score is the mean of `exact` over depths and lengths.

Usage:
  python niah_eval.py --mode dense --model meta-llama/Llama-3.1-8B-Instruct \
      --haystack "$DATA_DIR/pg19/*.parquet" --lengths 4096 --out niah_dense_L4096.json
  python niah_eval.py --mode sparse --lora runs/g2s4/lora_final.pt --lora_rank 16 ...
"""
import os, sys, json, glob, argparse, random
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
# Module-level: these modules register their attention fn as an import side effect, and
# transformers resolves attn_implementation inside from_pretrained(). Guarded because the
# MInference/XAttention environment pins transformers 4.46, where the registration API does not
# exist -- there only --mode dense/minference/xattention is available.
try:
    from llama_intra import install_intra_llama
    from moba_student import install_moba_llama   # noqa: F401  (registers "moba")
    from nsa_student import install_nsa_llama     # noqa: F401  (registers "nsa")
    _STUDENTS_ERR = None
except Exception as _e:
    _STUDENTS_ERR = _e


NEEDLE_TMPL = "The special magic {city} number is: {num}."
QUESTION_TMPL = "What is the special magic {city} number?"
ANSWER_TMPL = "The special magic {city} number is: {num}"
CITIES = ["San Francisco", "Tokyo", "Reykjavik", "Nairobi", "Montevideo",
          "Helsinki", "Kyoto", "Lisbon"]


def load_haystack_tokens(tok, parquet_glob, need_tokens):
    """Tokenize parquet rows (sorted file order) until `need_tokens` tokens are collected."""
    import pyarrow.parquet as pq
    files = sorted(glob.glob(parquet_glob))
    if not files:
        raise FileNotFoundError(f"no parquet matched {parquet_glob}")

    def _rows():
        for f in files:
            t = pq.read_table(f)
            for batch in t.to_batches():
                d = batch.to_pydict()
                n = len(next(iter(d.values())))
                for i in range(n):
                    yield {k: d[k][i] for k in d}

    buf = []
    for ex in _rows():
        t = None
        for k in ("text", "content", "raw_content", "document"):
            if k in ex and ex[k]:
                t = ex[k]
                break
        if not t:
            continue
        buf.extend(tok(t, add_special_tokens=False)["input_ids"])
        if len(buf) >= need_tokens:
            break
    assert len(buf) >= need_tokens, f"haystack too short: {len(buf)} < {need_tokens}"
    return buf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--mode", choices=["dense", "sparse", "moba", "nsa", "minference", "xattention"],
                    required=True)
    ap.add_argument("--minference_config_name", default="meta-llama/Meta-Llama-3.1-8B-Instruct",
                    help="minference mode: HF name for its published per-head config lookup")
    ap.add_argument("--xattn_stride", type=int, default=16, choices=[4, 8, 16],
                    help="xattention mode: antidiagonal stride; selects the official per-layer/"
                         "per-head threshold table shipped with the repo (16 = the paper's S=16)")
    ap.add_argument("--lora", default="")
    ap.add_argument("--lora_rank", type=int, default=8)
    ap.add_argument("--num_groups", type=int, default=8)
    ap.add_argument("--window", type=int, default=575)
    ap.add_argument("--n_sink", type=int, default=0)
    ap.add_argument("--scatter_blocks", default="1",
                    help="comma-separated block sizes cycled across Scatter layers")
    ap.add_argument("--moba_chunk", type=int, default=2048)
    ap.add_argument("--moba_density", type=float, default=0.125)
    ap.add_argument("--nsa_density", type=float, default=0.1438, help="nominal; measures 0.126")
    ap.add_argument("--nsa_sel_block", type=int, default=64)
    ap.add_argument("--haystack", required=True, help="glob of PG19 parquet files")
    ap.add_argument("--lengths", default="4096,8192,16384")
    ap.add_argument("--depths", default="0.1,0.3,0.5,0.7,0.9")
    ap.add_argument("--trials", type=int, default=2, help="different cities/numbers per cell")
    ap.add_argument("--out", default="")
    ap.add_argument("--layer_pattern", default="SSWW",
                    help="branch per layer, cycled: S=Scatter, W=window, D=full causal")
    ap.add_argument("--scatter_stripe", type=int, default=0,
                    help="shifted-stripe Scatter; must match how the adapter was trained")
    ap.add_argument("--scatter_n_offsets", type=int, default=0,
                    help="distinct stripe offsets (0 = num_groups); must match training")
    ap.add_argument("--scatter_impl", default="loop", choices=["loop", "flex"],
                    help="loop or flex (same math); forced to flex when a stripe offset is non-zero")
    ap.add_argument("--tag_suffix", default="", help="appended to the auto-derived tag")
    a = ap.parse_args()
    dev = "cuda"
    random.seed(1234)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    lengths = [int(x) for x in a.lengths.split(",")]
    depths = [float(x) for x in a.depths.split(",")]

    print(f"[niah] loading haystack (need {max(lengths)} tok)", flush=True)
    hay = load_haystack_tokens(tok, a.haystack, max(lengths) + 4096)

    if a.mode in ("sparse", "moba", "nsa") and _STUDENTS_ERR is not None:
        raise RuntimeError(f"--mode {a.mode} needs the student modules, whose import failed in this "
                           f"environment: {_STUDENTS_ERR!r}")
    impl = {"dense": "sdpa", "sparse": "intra", "moba": "moba", "nsa": "nsa",
            "minference": "sdpa", "xattention": "sdpa"}[a.mode]
    print(f"[niah] loading model (attn={impl})", flush=True)
    import transformers
    _dt = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
    model = AutoModelForCausalLM.from_pretrained(a.model, attn_implementation=impl,
                                                 **{_dt: torch.bfloat16}).to(dev).eval()
    model.config.use_cache = False
    tag = a.mode
    if a.mode == "sparse":
        blocks = tuple(int(x) for x in a.scatter_blocks.split(","))
        ns, nw, nd = install_intra_llama(model, num_groups=a.num_groups, window=a.window,
                                         dense_debug=False, compile_flex=True, n_sink=a.n_sink,
                                         scatter_blocks=blocks, layer_pattern=a.layer_pattern,
                                         scatter_stripe=bool(a.scatter_stripe),
                                         scatter_impl=a.scatter_impl,
                                         scatter_n_offsets=a.scatter_n_offsets)
        tag += f"_g{a.num_groups}_w{a.window}_sink{a.n_sink}_blk{a.scatter_blocks.replace(',','-')}"
        if a.layer_pattern.upper() != "SSWW":
            tag += f"_pat{a.layer_pattern.upper()}"
        if a.scatter_stripe:
            tag += f"_stripe{a.scatter_n_offsets or a.num_groups}"
        print(f"[niah] INTRA: {ns} scatter / {nw} window / {nd} dense, n_sink={a.n_sink}, "
              f"blocks={blocks}, pattern={a.layer_pattern}", flush=True)
        if a.lora:
            from peft import LoraConfig
            model.add_adapter(LoraConfig(
                r=a.lora_rank, lora_alpha=a.lora_rank, init_lora_weights="gaussian",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"]))
            sd = torch.load(a.lora, map_location="cpu")
            missing, unexpected = model.load_state_dict(sd, strict=False)
            print(f"[niah] loaded LoRA {len(sd)} tensors (unexpected={len(unexpected)})", flush=True)
            tag += "_lora"
    elif a.mode == "moba":
        n_l = install_moba_llama(model, chunk_size=a.moba_chunk, topk=None, density=a.moba_density)
        tag += f"_d{a.moba_density}_c{a.moba_chunk}"
        print(f"[niah] MoBA on {n_l} layers: chunk={a.moba_chunk} density={a.moba_density}", flush=True)
    elif a.mode == "nsa":
        n_l = install_nsa_llama(model, sel_block=a.nsa_sel_block, n_sel=None,
                                density=a.nsa_density, window=512)
        tag += f"_d{a.nsa_density}_sel{a.nsa_sel_block}"
        print(f"[niah] NSA on {n_l} layers: nominal density={a.nsa_density} "
              f"sel_block={a.nsa_sel_block}", flush=True)
    elif a.mode == "minference":
        # Training-free; patches prefill attention in place. Scoring here is a single prefill
        # forward, which is exactly the phase MInference sparsifies.
        from minference import MInference as _MInference
        patched = _MInference("minference", a.minference_config_name)(model)
        if patched is not model:
            raise RuntimeError("MInference returned a different model object; expected in-place patch")
        print(f"[niah] MInference patched (config={a.minference_config_name}, kv=dense)", flush=True)
    elif a.mode == "xattention":
        # Training-free (Xu et al.). Official repo wiring: forward_eval replaces each layer's
        # attention forward; prefill runs Xattention_prefill (antidiagonal block estimate + the
        # official block_sparse_attn kernel) with the repo's calibrated per-layer/per-head
        # thresholds; decode goes through flashinfer. Scoring here is a single prefill forward,
        # exactly the phase XAttention sparsifies. GQA is handled by the repo's own repeat_kv.
        import types as _types
        from xattn.src.load_llama import forward_eval as _xattn_fwd, FastPrefillConfig as _XCfg
        _xcfg = _XCfg(stride=a.xattn_stride)  # metric="xattn"; official thresholds for the stride
        _n = 0
        for _lyr in model.model.layers:
            _lyr.self_attn.fastprefillconfig = _xcfg
            _lyr.self_attn.forward = _types.MethodType(_xattn_fwd, _lyr.self_attn)
            _n += 1
        print(f"[niah] XAttention patched on {_n} layers (stride={a.xattn_stride}, "
              f"official thresholds, decode=flashinfer)", flush=True)

    # LoRA for the dense/MoBA/NSA modes. strict=False means a rank/target mismatch loads nothing
    # and returns cleanly, so the match count is checked.
    if a.lora and a.mode != "sparse":
        from peft import LoraConfig
        model.add_adapter(LoraConfig(
            r=a.lora_rank, lora_alpha=a.lora_rank, init_lora_weights="gaussian",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"]))
        sd = torch.load(a.lora, map_location="cpu")
        missing, unexpected = model.load_state_dict(sd, strict=False)
        loaded = len(sd) - len(unexpected)
        print(f"[niah] loaded LoRA {loaded}/{len(sd)} tensors from {a.lora}", flush=True)
        if loaded == 0:
            raise RuntimeError(f"LoRA {a.lora} matched no parameters -- wrong rank or targets")
        tag += "_lora"

    if a.tag_suffix:
        tag += "_" + a.tag_suffix.lstrip("_")

    results = []
    for L in lengths:
        for dep in depths:
            hit1 = hitall = 0
            lps = []
            for t in range(a.trials):
                city = CITIES[(hash((L, dep, t)) % len(CITIES))]
                num = str(random.randint(1000000, 9999999))
                needle = tok(NEEDLE_TMPL.format(city=city, num=num),
                             add_special_tokens=False)["input_ids"]
                question = QUESTION_TMPL.format(city=city)
                answer = ANSWER_TMPL.format(city=city, num=num)

                # ---- build the context: haystack with the needle planted at `dep` ----
                budget = L - len(needle) - 256          # leave room for template + question
                body = hay[:budget]
                cut = int(len(body) * dep)
                ctx_ids = body[:cut] + needle + body[cut:]
                ctx_text = tok.decode(ctx_ids)

                msgs = [{"role": "user",
                         "content": ctx_text + "\n\n" + question}]
                prompt = tok.apply_chat_template(msgs, tokenize=False,
                                                 add_generation_prompt=True)
                p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
                a_ids = tok(answer, add_special_tokens=False)["input_ids"]
                ids = torch.tensor([p_ids + a_ids], device=dev)

                # positions predicting the answer tokens
                start = len(p_ids)
                with torch.no_grad():
                    # Apply lm_head only to the answer positions (position-wise, so slicing the
                    # hidden states first is exact); full-length fp32 logits would not fit at
                    # long context.
                    base = getattr(model, "model", None)
                    head = getattr(model, "lm_head", None)
                    if base is not None and head is not None:
                        h = base(input_ids=ids, use_cache=False).last_hidden_state
                        pred = head(h[:, start - 1: start - 1 + len(a_ids)]).float()[0]
                        logits = h          # only so the `del` below still frees the big tensor
                    else:
                        # Unknown architecture: fall back to the original path rather than guess.
                        logits = model(input_ids=ids, use_cache=False).logits.float()
                        pred = logits[0, start - 1: start - 1 + len(a_ids)]
                gold = torch.tensor(a_ids, device=dev)
                lp = F.log_softmax(pred, -1).gather(-1, gold[:, None]).squeeze(-1)
                am = pred.argmax(-1)
                # the number is what matters: score only the numeric tail tokens
                num_ids = tok(num, add_special_tokens=False)["input_ids"]
                k = len(num_ids)
                hit1 += int((am[-k:] == gold[-k:])[0].item())
                hitall += int((am[-k:] == gold[-k:]).all().item())
                lps.append(lp[-k:].mean().item())
                del logits
                torch.cuda.empty_cache()

            row = {"len": L, "depth": dep, "first_tok_acc": hit1 / a.trials,
                   "exact": hitall / a.trials, "mean_logprob": sum(lps) / len(lps)}
            results.append(row)
            print(f"[niah] L={L:6d} depth={dep:.1f}  first-tok {row['first_tok_acc']:.2f}  "
                  f"exact {row['exact']:.2f}  logp {row['mean_logprob']:7.3f}", flush=True)

    n = len(results)
    summ = {"tag": tag, "mode": a.mode, "lora": a.lora, "n_sink": a.n_sink,
            "mean_first_tok": sum(r["first_tok_acc"] for r in results) / n,
            "mean_exact": sum(r["exact"] for r in results) / n,
            "mean_logprob": sum(r["mean_logprob"] for r in results) / n,
            "grid": results}
    print(f"\n[niah] === {tag} ===")
    print(f"[niah] first-tok acc {summ['mean_first_tok']:.3f} | exact {summ['mean_exact']:.3f} "
          f"| mean logprob {summ['mean_logprob']:.3f}")
    out = a.out or f"niah_{tag}.json"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    json.dump(summ, open(out, "w"), indent=2)
    print(f"[niah] -> {out}")


if __name__ == "__main__":
    main()
