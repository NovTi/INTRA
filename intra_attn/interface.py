"""Python entry points for the INTRA CUDA kernels (forward only).

All functions take q, k, v of shape (batch, seqlen, heads, 128) in bfloat16 on an sm80+ GPU and
return the attention output in the same layout. k/v may have fewer heads than q (GQA) as long as
they divide the query heads. There is no backward pass; training uses the PyTorch paths in
`experiments/` (SDPA / FlexAttention over the same patterns).
"""
import torch

import intra_attn_cuda as _C

LLM_PATTERNS = {"scatter": 0, "gather1": 1, "window": 4}
IMAGE_PATTERNS = {"scatter": 0, "gather0": 4}
TEXT_LEN = 512


def _check(q, k, v):
    for x in (q, k, v):
        if not x.is_cuda or x.dtype != torch.bfloat16 or x.shape[-1] != 128:
            raise ValueError("INTRA kernels need CUDA bfloat16 tensors with head_dim 128")
    if torch.is_grad_enabled() and any(x.requires_grad for x in (q, k, v)):
        raise RuntimeError("INTRA kernels are forward-only; call them under torch.no_grad()")
    return [x if x.stride(-1) == 1 else x.contiguous() for x in (q, k, v)]


def _scale(q, softmax_scale):
    return q.shape[-1] ** -0.5 if softmax_scale is None else softmax_scale


def intra_llm_attn(q, k, v, pattern="scatter", group_num=8, window=576, softmax_scale=None):
    """Causal prefill attention for language models.

    pattern="scatter": token i attends to earlier tokens j with j % group_num == i % group_num.
    pattern="gather1": groups are segments of group_num tokens, segment s in group s % group_num.
    pattern="window":  each 64-query block attends to itself and the preceding window - 64 tokens.

    Any seqlen works for scatter and window; the gather1 kernel is instantiated for
    group_num=8 and needs seqlen % 64 == 0.
    """
    q, k, v = _check(q, k, v)
    if pattern == "gather1" and (group_num != 8 or q.shape[1] % (group_num * group_num)):
        raise ValueError(f"gather1 needs group_num=8 and seqlen % 64 == 0, got G={group_num}, S={q.shape[1]}")
    out = _C.fwd_llm(q, k, v, None, None, 0.0, _scale(q, softmax_scale), True, -1, -1, 0.0,
                     False, None, LLM_PATTERNS[pattern], group_num, window)[0]
    return out


def intra_image_attn(q, k, v, pattern="scatter", width=64, softmax_scale=None):
    """Bidirectional attention for FLUX-style sequences: 512 text tokens followed by a
    width x width grid of image tokens.

    pattern="scatter": 8 groups, image token (r, c) in group (r % 2) * 4 + c % 4; text chunks of
                       64 tokens ride with groups 0..7; all text keys are always visible.
    pattern="gather0": each 8 x 8 image patch attends to a 16 x 16 window around it plus all text.

    The kernels are instantiated for width 64 (1024 x 1024 FLUX images, 4096 image tokens) and
    width 128 (2048 x 2048 images, 16384 image tokens).
    """
    q, k, v = _check(q, k, v)
    if width not in (64, 128):
        raise ValueError("image kernels are compiled for width 64 and 128")
    if q.shape[1] != TEXT_LEN + width * width:
        raise ValueError(f"expected {TEXT_LEN} text + {width}x{width} image tokens, got {q.shape[1]}")
    out = _C.fwd(q, k, v, None, None, 0.0, _scale(q, softmax_scale), False, -1, -1, 0.0,
                 False, None, IMAGE_PATTERNS[pattern], width, width)[0]
    return out
