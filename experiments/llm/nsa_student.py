#!/usr/bin/env python3
"""NSA as a *student* attention for train_llama_distill.py, mirroring moba_student.py.

    install_nsa_llama(model, ...)   stamps per-layer cfg
    set_nsa_dense(model, flag)      teacher/student toggle

The attention itself is nsa_attn.nsa_attn, so training and evaluation run the same code.
"""
import os, sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nsa_attn import nsa_attn  # noqa: E402

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


def nsa_attention_forward(module, query, key, value, attention_mask=None,
                          scaling=None, dropout=0.0, **kwargs):
    """transformers attention-interface fn. q/k/v: (B,H,S,D) -> ((B,S,Hq,D), None)."""
    cfg = getattr(module, "_nsa", None)
    if scaling is None:
        scaling = getattr(module, "scaling", query.shape[-1] ** -0.5)
    if cfg is None or cfg.get("dense_debug", False) or query.shape[2] == 1:
        # dense_debug -> teacher pass. q_len==1 -> decode: no block structure to select over with a
        # single query, so fall back to dense (same as the MoBA student and eval-time behaviour).
        out = _dense_causal(query, key, value, scaling)
        return out.transpose(1, 2).contiguous(), None
    return nsa_attn(query, key, value, cmp_block=cfg["cmp_block"], cmp_stride=cfg["cmp_stride"],
                    sel_block=cfg["sel_block"], n_sel=cfg.get("n_sel"),
                    density=cfg.get("density"), window=cfg["window"],
                    use_cmp=cfg.get("use_cmp", True)), None


if AttentionInterface is not None:
    try:
        AttentionInterface.register("nsa", nsa_attention_forward)
    except Exception:
        ALL_ATTENTION_FUNCTIONS["nsa"] = nsa_attention_forward
else:
    ALL_ATTENTION_FUNCTIONS["nsa"] = nsa_attention_forward


def install_nsa_llama(model, cmp_block=32, cmp_stride=16, sel_block=64, n_sel=None,
                      density=None, window=512, use_cmp=True):
    """Stamp per-layer NSA config. Model must be loaded with attn_implementation='nsa'.

    Model-agnostic: walks model.model.layers, so LLaMA-3.1 (32) and Qwen3 (36) both work.
    """
    layers = model.model.layers if hasattr(model, "model") else model.layers
    n = 0
    for li, layer in enumerate(layers):
        layer.self_attn._nsa = dict(cmp_block=cmp_block, cmp_stride=cmp_stride,
                                    sel_block=sel_block, n_sel=n_sel, density=density,
                                    window=window, use_cmp=use_cmp, layer_idx=li,
                                    dense_debug=False)
        n += 1
    return n


def set_nsa_dense(model, flag: bool):
    """Teacher/student toggle -- counterpart of set_moba_dense / set_dense_debug."""
    layers = model.model.layers if hasattr(model, "model") else model.layers
    for layer in layers:
        if hasattr(layer.self_attn, "_nsa"):
            layer.self_attn._nsa["dense_debug"] = flag


if __name__ == "__main__":
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    B, Hq, Hkv, S, D = 1, 8, 2, 1024, 64
    q = torch.randn(B, Hq, S, D, device=dev, requires_grad=True)
    k = torch.randn(B, Hkv, S, D, device=dev, requires_grad=True)
    v = torch.randn(B, Hkv, S, D, device=dev, requires_grad=True)
    o = nsa_attn(q, k, v, density=0.1438, sel_block=64)
    print(f"  out {tuple(o.shape)} expected {(B, S, Hq, D)}")
    assert tuple(o.shape) == (B, S, Hq, D)
    o.square().mean().backward()
    for nm, t in (("q", q), ("k", k), ("v", v)):
        g = t.grad
        assert g is not None and torch.isfinite(g).all(), f"{nm} grad bad"
        print(f"  grad {nm}: |g|={g.abs().mean().item():.3e} finite=True")

    # GQA must be expanded, not broadcast: selection is per query head.
    cfg = dict(density=0.1438, sel_block=64)
    o_gqa = nsa_attn(q.detach(), k.detach(), v.detach(), **cfg)
    kx = k.detach().repeat_interleave(Hq // Hkv, dim=1)
    vx = v.detach().repeat_interleave(Hq // Hkv, dim=1)
    o_exp = nsa_attn(q.detach(), kx, vx, **cfg)
    d = (o_gqa - o_exp).abs().max().item()
    print(f"  GQA expansion matches pre-expanded input: max diff {d:.3e}")
    assert d < 1e-5, f"GQA path differs from explicitly expanded input ({d:.3e})"
    print("  nsa_student PASS")
