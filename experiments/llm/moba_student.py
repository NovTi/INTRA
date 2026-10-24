#!/usr/bin/env python3
"""MoBA as a *student* attention for train_llama_distill.py, mirroring llama_intra.py's contract.

    install_moba_llama(model, ...)   stamps per-layer cfg, like install_intra_llama
    set_moba_dense(model, flag)      the teacher/student toggle, like set_dense_debug

The attention itself is moba_attn.moba_attn (verified element-wise against upstream MoBA's
moba_naive.py; see moba_attn.py), so training and evaluation run the same code.

Density, not top-k, is the matched quantity: a fixed top-k is a different fraction of the context
at every input length, while --moba_density keeps the fraction of selected blocks fixed.
"""
import os, sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from moba_attn import moba_attn  # noqa: E402

try:
    from transformers.modeling_utils import AttentionInterface
except Exception:
    AttentionInterface = None
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS  # noqa: E402


def _dense_causal(q, k, v, scaling):
    Hq, Hkv = q.shape[1], k.shape[1]
    if Hq != Hkv:
        rep = Hq // Hkv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    return F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scaling)


def moba_attention_forward(module, query, key, value, attention_mask=None,
                           scaling=None, dropout=0.0, **kwargs):
    """transformers attention-interface fn. q/k/v: (B,H,S,D) -> ((B,S,Hq,D), None)."""
    cfg = getattr(module, "_moba", None)
    if scaling is None:
        scaling = getattr(module, "scaling", query.shape[-1] ** -0.5)
    if cfg is None or cfg.get("dense_debug", False) or query.shape[2] == 1:
        # dense_debug -> the teacher pass. q_len==1 -> decode, where there is nothing to select
        # over; upstream MoBA falls back to dense there too.
        out = _dense_causal(query, key, value, scaling)
        return out.transpose(1, 2).contiguous(), None
    # moba_attn already returns (B,S,Hq,D)
    return moba_attn(query, key, value, chunk_size=cfg["chunk_size"], topk=cfg.get("topk"),
                     density=cfg.get("density"), q_block=cfg.get("q_block")), None


if AttentionInterface is not None:
    try:
        AttentionInterface.register("moba", moba_attention_forward)
    except Exception:
        ALL_ATTENTION_FUNCTIONS["moba"] = moba_attention_forward
else:
    ALL_ATTENTION_FUNCTIONS["moba"] = moba_attention_forward


def install_moba_llama(model, chunk_size=2048, topk=None, density=None, q_block=None):
    """Stamp per-layer MoBA config. Model must be loaded with attn_implementation='moba'.

    Model-agnostic on purpose: it walks model.model.layers, so LLaMA-3.1 (32 layers) and Qwen3
    (36 layers) both work without a branch.
    """
    layers = model.model.layers if hasattr(model, "model") else model.layers
    n = 0
    for li, layer in enumerate(layers):
        layer.self_attn._moba = dict(chunk_size=chunk_size, topk=topk, density=density,
                                     q_block=q_block, layer_idx=li, dense_debug=False)
        n += 1
    return n


def set_moba_dense(model, flag: bool):
    """Teacher/student toggle -- the exact counterpart of llama_intra.set_dense_debug."""
    layers = model.model.layers if hasattr(model, "model") else model.layers
    for layer in layers:
        if hasattr(layer.self_attn, "_moba"):
            layer.self_attn._moba["dense_debug"] = flag


if __name__ == "__main__":
    # gradient sanity: top-k selection is non-differentiable, so confirm gradient reaches q/k/v
    # (it flows through the selected values, as in upstream MoBA).
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    B, Hq, Hkv, S, D = 1, 8, 2, 512, 64
    q = torch.randn(B, Hq, S, D, device=dev, requires_grad=True)
    k = torch.randn(B, Hkv, S, D, device=dev, requires_grad=True)
    v = torch.randn(B, Hkv, S, D, device=dev, requires_grad=True)
    o = moba_attn(q, k, v, chunk_size=64, density=0.125)
    print(f"  out {tuple(o.shape)} expected {(B, S, Hq, D)}")
    o.square().mean().backward()
    for nm, t in (("q", q), ("k", k), ("v", v)):
        g = t.grad
        assert g is not None and torch.isfinite(g).all(), f"{nm} grad bad"
        print(f"  grad {nm}: |g|={g.abs().mean().item():.3e} finite=True")
    print("  moba_student gradient check PASS")
