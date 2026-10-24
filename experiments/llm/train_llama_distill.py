"""
train_llama_distill.py -- LoRA self-distillation of a sparse-attention student from its own dense
teacher (LLaMA-3.1-8B-Instruct or Qwen3-8B).

  * One base model. The teacher pass is the same weights with the LoRA adapter disabled and
    attention forced dense; the student pass has the adapter on and the sparse pattern installed.
    The teacher therefore cannot drift from the base model and costs no extra memory.
  * Loss = alpha * T^2 * KL(teacher || student) + ce_weight * next-token CE, per token
    (reduction='sum' / (B*S)), computed in fp32. lm_head, KL and CE are evaluated over sequence
    chunks against a detached leaf, so full-vocabulary logits are never materialised; one
    hidden.backward(leaf.grad) then propagates into the transformer.
  * Data: fixed-length packed blocks (block_*.pt, no padding) from pack_scrolls_mix.py.

Students (--student_attn): intra (llama_intra.py), moba (moba_student.py), nsa (nsa_student.py),
dense (no sparse attention; the KL term then acts as a proximal pull toward the base model).

Note on multi-GPU runs: the model is NOT wrapped in DistributedDataParallel and gradients are not
all-reduced. Under torchrun each rank optimises its own adapter copy on its DistributedSampler shard
and only rank 0 saves its adapter; the effective batch size is 1 sequence per step. All reported
adapters were trained this way (torchrun --nproc_per_node 8).

Launch:
  torchrun --standalone --nproc_per_node 8 train_llama_distill.py \
      --model meta-llama/Llama-3.1-8B-Instruct --data $DATA_DIR/mixgen_16k_llama \
      --out $OUT_DIR/runs/g2s4 --seq_len 16384 --steps 1200 --lora_rank 16 ...
"""
import os, sys, json, time, math, glob, argparse
from contextlib import contextmanager

import torch
import torch.nn.functional as F
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Module-level imports: each module registers its attention fn as an import side effect, and
# transformers resolves attn_implementation inside from_pretrained().
from llama_intra import install_intra_llama, set_dense_debug
from moba_student import install_moba_llama, set_moba_dense  # noqa: F401  (registers "moba")
from nsa_student import install_nsa_llama, set_nsa_dense    # noqa: F401  (registers "nsa")


def log(msg):
    if int(os.environ.get("RANK", 0)) == 0:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


@contextmanager
def adapter_off(model):
    """Temporarily disable the LoRA adapter -> the frozen dense teacher.

    peft's PeftModel exposes `disable_adapter()`; transformers' PeftAdapterMixin (what
    `model.add_adapter(...)` gives you) exposes `disable_adapters()` / `enable_adapters()`.
    """
    if hasattr(model, "disable_adapter"):
        with model.disable_adapter():
            yield
    else:
        model.disable_adapters()
        try:
            yield
        finally:
            model.enable_adapters()


class PackedBlocks(torch.utils.data.Dataset):
    """Fixed-length token blocks (block_*.pt) -- no padding, no labels needed."""
    def __init__(self, path, seq_len):
        self.files = sorted(glob.glob(os.path.join(path, "block_*.pt")))
        assert self.files, f"no blocks in {path}"
        self.seq_len = seq_len

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        ids = torch.load(self.files[i], map_location="cpu")
        if ids.dim() == 2:
            ids = ids.squeeze(0)
        return ids[: self.seq_len].long()


def _per_token_div(logp_s, log_p_t, mode):
    """Per-token divergence, shape (B,c). All inputs are log-probs over the vocab.

    forward = KL(teacher||student) (mode-covering, standard distillation)
    reverse = KL(student||teacher) (mode-seeking)
    jsd     = symmetric midpoint, computed in log space for stability.
    """
    if mode == "forward":
        return (log_p_t.exp() * (log_p_t - logp_s)).sum(-1)
    if mode == "reverse":
        return (logp_s.exp() * (logp_s - log_p_t)).sum(-1)
    if mode == "jsd":
        log_m = torch.logaddexp(logp_s, log_p_t) - math.log(2.0)
        return 0.5 * (logp_s.exp() * (logp_s - log_m)).sum(-1) \
             + 0.5 * (log_p_t.exp() * (log_p_t - log_m)).sum(-1)
    raise ValueError(mode)


def chunked_distill_loss(lm_head, h_s, h_t, ids, chunk, T, alpha, ce_weight,
                         kl_mode="forward", focal_gamma=0.0):
    """Chunked KL(+CE) that never materializes full-vocab logits.

    h_s: (B,S,H) student hidden WITH grad.  h_t: (B,S,H) teacher hidden, no grad.
    Returns (kl_value, ce_value, grad_for_h_s).

    focal_gamma > 0 re-weights each token by (kl_tok / mean_kl)**gamma, normalized over the whole
    sequence (hence the no-grad pre-pass). Off (0) for all reported runs.
    """
    B, S, _ = h_s.shape
    n_tok = B * S
    leaf = h_s.detach().requires_grad_(True)
    kl_tot, ce_tot = 0.0, 0.0

    w_all = None
    if focal_gamma > 0:
        with torch.no_grad():
            parts = []
            for s0 in range(0, S, chunk):
                e = min(s0 + chunk, S)
                lps = F.log_softmax(lm_head(leaf[:, s0:e]).float() / T, dim=-1)
                lpt = F.log_softmax(lm_head(h_t[:, s0:e]).float() / T, dim=-1)
                parts.append(_per_token_div(lps, lpt, kl_mode))
                del lps, lpt
            d_all = torch.cat(parts, dim=1)                              # (B,S)
            w_all = (d_all / d_all.mean().clamp_min(1e-6)).pow(focal_gamma)
            w_all = w_all / w_all.mean().clamp_min(1e-6)                 # keep loss scale ~unchanged
            del parts, d_all

    for s0 in range(0, S, chunk):
        e = min(s0 + chunk, S)
        ls = lm_head(leaf[:, s0:e]).float()                 # (B,c,V) fp32 BEFORE softmax
        with torch.no_grad():
            lt = lm_head(h_t[:, s0:e]).float()
        logp_s = F.log_softmax(ls / T, dim=-1)
        log_p_t = F.log_softmax(lt / T, dim=-1)
        d_tok = _per_token_div(logp_s, log_p_t, kl_mode)     # (B,c)
        if w_all is not None:
            d_tok = d_tok * w_all[:, s0:e]
        # true per-token KL; T^2 keeps gradient magnitude independent of temperature
        kl = (T * T) * d_tok.sum() / n_tok
        loss_c = alpha * kl
        if ce_weight > 0:
            # next-token CE on the packed text (labels = shifted input ids)
            tgt = ids[:, s0 + 1: e + 1]
            m = tgt.shape[1]
            if m > 0:
                ce = F.cross_entropy(ls[:, :m].reshape(-1, ls.shape[-1]),
                                     tgt.reshape(-1), reduction="sum") / n_tok
                loss_c = loss_c + ce_weight * ce
                ce_tot += ce.item()
        loss_c.backward()                                    # -> leaf.grad
        kl_tot += kl.item()
        del ls, lt, logp_s, log_p_t, d_tok
    return kl_tot, ce_tot, leaf.grad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct",
                    help="HF model id or local path (LLaMA-3.1-8B-Instruct or Qwen3-8B)")
    ap.add_argument("--data", required=True, help="dir of block_*.pt from pack_scrolls_mix.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq_len", type=int, default=4096)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--lora_rank", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--student_attn", default="intra", choices=["intra", "moba", "dense", "nsa"],
                    help="sparse pattern of the STUDENT. The teacher is always the same weights "
                         "with the adapter off and dense attention.")
    ap.add_argument("--moba_chunk", type=int, default=2048, help="--student_attn moba: block size")
    ap.add_argument("--nsa_cmp_block", type=int, default=32, help="--student_attn nsa: pooling block l")
    ap.add_argument("--nsa_cmp_stride", type=int, default=16, help="--student_attn nsa: pooling stride d")
    ap.add_argument("--nsa_sel_block", type=int, default=64, help="--student_attn nsa: selection block l'")
    ap.add_argument("--nsa_n_sel", type=int, default=0, help="--student_attn nsa: fixed block count; "
                    "ignored if --nsa_density > 0")
    ap.add_argument("--nsa_window", type=int, default=512, help="--student_attn nsa: sliding window w")
    ap.add_argument("--nsa_density", type=float, default=0.0,
                    help="--student_attn nsa: NOMINAL total density; all three branches are resized "
                         "to fit it (nsa_attn.plan_budget). 0.1438 nominal measures ~0.126.")
    ap.add_argument("--moba_topk", type=int, default=3,
                    help="--student_attn moba: fixed block count; ignored if --moba_density > 0")
    ap.add_argument("--moba_density", type=float, default=0.0,
                    help="--student_attn moba: keep this FRACTION of blocks (constant density "
                         "across sequence lengths)")
    ap.add_argument("--num_groups", type=int, default=8)
    ap.add_argument("--window", type=int, default=575)
    ap.add_argument("--scatter_blocks", default="1",
                    help="block sizes cycled across Scatter layers; group(i)=(i//block)%%num_groups. "
                         "Keep block <= seq_len/num_groups.")
    ap.add_argument("--n_sink", type=int, default=0,
                    help="attention sinks: first N tokens stay visible to every query in both the "
                         "window and Scatter branches")
    ap.add_argument("--kl_T", type=float, default=2.0)
    ap.add_argument("--alpha", type=float, default=0.9, help="KL weight")
    ap.add_argument("--layer_pattern", default="SSWW",
                    help="S=Scatter, W=window, D=full causal, cycled across layers")
    ap.add_argument("--scatter_stripe", type=int, default=0,
                    help="shifted-stripe Scatter: the s-th Scatter layer reads residue class "
                         "(g + s %% n_offsets) %% num_groups plus its own block")
    ap.add_argument("--scatter_n_offsets", type=int, default=0,
                    help="number of distinct stripe offsets cycled (0 = num_groups)")
    ap.add_argument("--scatter_impl", default="loop", choices=["loop", "flex"],
                    help="loop = gathered per-group SDPA; flex = same math via a cached block mask "
                         "(block-sparse cost). Forced to flex when a stripe offset is non-zero.")
    ap.add_argument("--kl_mode", default="forward", choices=["forward", "reverse", "jsd"],
                    help="forward=KL(t||s) standard distillation; reverse=KL(s||t); jsd=symmetric")
    ap.add_argument("--focal_gamma", type=float, default=0.0,
                    help="re-weight tokens by (kl/mean_kl)**gamma; 0 = uniform")
    ap.add_argument("--ce_weight", type=float, default=0.1)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--grad_ckpt", type=int, default=1)
    ap.add_argument("--compile_flex", type=int, default=1)
    ap.add_argument("--save_every", type=int, default=200)
    ap.add_argument("--log_every", type=int, default=10)
    a = ap.parse_args()

    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    if world > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local)
    dev = f"cuda:{local}"
    os.makedirs(a.out, exist_ok=True)
    if rank == 0:
        json.dump(vars(a), open(os.path.join(a.out, "config.json"), "w"), indent=2)

    from transformers import AutoModelForCausalLM
    # "dense" is not a registered attention implementation; it means "leave the model alone",
    # which transformers spells sdpa.
    impl = "sdpa" if a.student_attn == "dense" else a.student_attn
    log(f"loading base model {a.model} (bf16, attn={a.student_attn} -> impl={impl})")
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16, attn_implementation=impl).to(dev)
    model.config.use_cache = False
    model.requires_grad_(False)

    # `set_student_dense(model, True)` is the teacher pass; every student type must implement it.
    if a.student_attn == "intra":
        blocks = tuple(int(x) for x in a.scatter_blocks.split(","))
        n_s, n_w, n_d = install_intra_llama(model, num_groups=a.num_groups, window=a.window,
                                           dense_debug=False, compile_flex=bool(a.compile_flex),
                                           n_sink=a.n_sink, scatter_blocks=blocks,
                                           layer_pattern=a.layer_pattern,
                                           scatter_stripe=bool(a.scatter_stripe),
                                           scatter_impl=a.scatter_impl,
                                           scatter_n_offsets=a.scatter_n_offsets)
        log(f"INTRA installed: {n_s} scatter (G={a.num_groups}), {n_w} window "
            f"(sliding {a.window}), {n_d} dense layers, n_sink={a.n_sink}, blocks={blocks}, "
            f"stripe={bool(a.scatter_stripe)}, impl={a.scatter_impl}")
        set_student_dense = set_dense_debug
    elif a.student_attn == "moba":
        n_l = install_moba_llama(model, chunk_size=a.moba_chunk,
                                 topk=(a.moba_topk if a.moba_density <= 0 else None),
                                 density=(a.moba_density if a.moba_density > 0 else None))
        log(f"MoBA installed on {n_l} layers: chunk={a.moba_chunk} "
            + (f"density={a.moba_density}" if a.moba_density > 0 else f"topk={a.moba_topk}"))
        set_student_dense = set_moba_dense
    elif a.student_attn == "nsa":
        n_l = install_nsa_llama(model, cmp_block=a.nsa_cmp_block, cmp_stride=a.nsa_cmp_stride,
                                sel_block=a.nsa_sel_block,
                                n_sel=(a.nsa_n_sel if a.nsa_density <= 0 else None),
                                density=(a.nsa_density if a.nsa_density > 0 else None),
                                window=a.nsa_window)
        log(f"NSA installed on {n_l} layers: cmp={a.nsa_cmp_block}/{a.nsa_cmp_stride} "
            f"sel={a.nsa_sel_block} win={a.nsa_window} "
            + (f"density={a.nsa_density}" if a.nsa_density > 0 else f"n_sel={a.nsa_n_sel}"))
        set_student_dense = set_nsa_dense
    elif a.student_attn == "dense":
        # Equal-LoRA-budget dense control. At initialisation (LoRA B = 0) student and teacher are
        # the same function, so the KL term acts as a proximal pull toward the base model while
        # the CE term pushes toward the data.
        set_student_dense = lambda _m, _flag: None      # noqa: E731  (already dense)
        log("DENSE student: no sparse attention installed; teacher = same weights, adapter off")
    else:
        raise ValueError(f"unknown --student_attn {a.student_attn}")

    lm_head = model.lm_head
    base = model.model  # decoder only -> hidden states; lm_head is applied manually in chunks

    from peft import LoraConfig
    targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    model.add_adapter(LoraConfig(r=a.lora_rank, lora_alpha=a.lora_rank,
                                 init_lora_weights="gaussian", target_modules=targets))
    if a.grad_ckpt:
        model.gradient_checkpointing_enable()
    trainable = [p for p in model.parameters() if p.requires_grad]
    log(f"trainable(LoRA) = {sum(p.numel() for p in trainable)/1e6:.2f}M over {targets}")

    def save_ckpt(fname):
        sd = {k: v.detach().cpu() for k, v in model.state_dict().items() if "lora_" in k}
        assert sd, "empty checkpoint -- no lora_ weights found"
        p = os.path.join(a.out, fname)
        torch.save(sd, p)
        return p

    ds = PackedBlocks(a.data, a.seq_len)
    sampler = (torch.utils.data.DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True)
               if world > 1 else None)
    dl = torch.utils.data.DataLoader(ds, batch_size=1, sampler=sampler,
                                     shuffle=(sampler is None), num_workers=2, drop_last=True)
    log(f"dataset {len(ds)} packed blocks of {a.seq_len} tokens")

    opt = torch.optim.AdamW(trainable, lr=a.lr, betas=(0.9, 0.999), weight_decay=0.0)

    step, micro, t0 = 0, 0, time.time()
    run_kl = run_ce = 0.0
    model.train()
    it = iter(dl)
    epoch = 0
    if sampler is not None:
        sampler.set_epoch(0)

    while step < a.steps:
        try:
            ids = next(it)
        except StopIteration:
            epoch += 1
            if sampler is not None:
                sampler.set_epoch(epoch)
            it = iter(dl)
            ids = next(it)
        ids = ids.to(dev)

        # ---- TEACHER: same weights, adapter OFF, attention DENSE, no grad ----
        with torch.no_grad():
            set_student_dense(model, True)
            with adapter_off(model):
                h_t = base(input_ids=ids, use_cache=False).last_hidden_state
            set_student_dense(model, False)

        # ---- STUDENT: adapter ON, sparse attention ----
        h_s = base(input_ids=ids, use_cache=False).last_hidden_state

        # ---- chunked KL(+CE): full-vocab logits never materialized ----
        kl_v, ce_v, g = chunked_distill_loss(lm_head, h_s, h_t, ids,
                                             a.chunk, a.kl_T, a.alpha, a.ce_weight,
                                             kl_mode=a.kl_mode, focal_gamma=a.focal_gamma)
        h_s.backward(g / a.grad_accum)       # single backward into the transformer graph
        run_kl += kl_v
        run_ce += ce_v
        micro += 1
        del h_s, h_t, g

        if micro % a.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % a.log_every == 0:
                d = a.log_every * a.grad_accum
                log(f"step {step}/{a.steps} KL {run_kl/d:.4f} CE {run_ce/d:.4f} "
                    f"{(time.time()-t0)/step:.1f}s/step len={ids.shape[1]}")
                run_kl = run_ce = 0.0
            if step % a.save_every == 0 and rank == 0:
                log(f"saved step{step} -> {save_ckpt(f'lora_step{step}.pt')}")

    if rank == 0:
        log(f"DONE {a.steps} steps in {(time.time()-t0)/60:.1f} min -> {save_ckpt('lora_final.pt')}")
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
