"""Dense boolean masks for every pattern the INTRA kernels implement.

These are the ground truth the CUDA kernels are tested against (tests/test_kernel.py) and a
readable specification of each pattern. `mask[q, k] == True` means query q may attend to key k.
They materialise an S x S mask, so use them for testing, not for long sequences.
"""
import torch
import torch.nn.functional as F

TEXT_LEN = 512  # FLUX: T5 text tokens placed before the image tokens


# ----------------------------------------------------------------------------- language (causal)
def llm_scatter_mask(S, group_num=8, device=None):
    """Scatter: token i belongs to group i % G; causal attention within each group."""
    i = torch.arange(S, device=device)
    return (i[None, :] <= i[:, None]) & (i[:, None] % group_num == i[None, :] % group_num)


def llm_gather1_mask(S, group_num=8, device=None):
    """Gather 1: segments of G consecutive tokens, segment s belongs to group s % G; causal."""
    i = torch.arange(S, device=device)
    g = (i // group_num) % group_num
    return (i[None, :] <= i[:, None]) & (g[:, None] == g[None, :])


def llm_window_mask(S, window=576, block=64, device=None):
    """Kernel sliding window: each 64-query block sees its own block plus `window - 64` earlier
    tokens (the lookback is aligned to the query block, so it spans window-64 .. window-1 keys)."""
    i = torch.arange(S, device=device)
    start = ((i // block) * block - (window - block)).clamp(min=0)
    return (i[None, :] <= i[:, None]) & (i[None, :] >= start[:, None])


# ----------------------------------------------------------------------- image (bidirectional)
def _image_scatter_group(width, text_len=TEXT_LEN, device=None):
    """Group id of every token for the 8-group image Scatter.

    Image token (r, c) on a width x width latent grid is in group (r % 2) * 4 + c % 4, i.e. one
    token of every 2 x 4 neighbourhood per group. Text tokens are split into 8 contiguous chunks,
    chunk j riding with image group j.
    """
    t = torch.arange(text_len, device=device) // (text_len // 8)
    p = torch.arange(width * width, device=device)
    r, c = p // width, p % width
    return torch.cat([t, (r % 2) * 4 + c % 4])


def image_scatter_mask(width, text_len=TEXT_LEN, device=None):
    """Scatter (image): every query sees all text keys plus the image keys of its own group."""
    g = _image_scatter_group(width, text_len, device)
    is_text = torch.arange(g.numel(), device=device) < text_len
    return is_text[None, :] | ((g[:, None] == g[None, :]) & ~is_text[None, :])


def image_gather0_mask(width, text_len=TEXT_LEN, patch=8, window=16, device=None):
    """Gather 0 (image): each 8 x 8 patch of image queries sees a 16 x 16 window of image keys
    centred on the patch (shifted inwards at the borders) plus all text keys. Text queries keep
    their Scatter rows."""
    m = image_scatter_mask(width, text_len, device)
    n_patch = width // patch
    p = torch.arange(width * width, device=device)
    r, c = p // width, p % width

    def start(idx):  # first row/col of the window for a patch index
        s = idx * patch - (window - patch) // 2
        s = torch.where(idx == 0, torch.zeros_like(s), s)
        return torch.where(idx == n_patch - 1, idx * patch - (window - patch), s)

    r0, c0 = start(r // patch), start(c // patch)
    inside = ((r[None, :] >= r0[:, None]) & (r[None, :] < r0[:, None] + window)
              & (c[None, :] >= c0[:, None]) & (c[None, :] < c0[:, None] + window))
    m[text_len:, text_len:] = inside
    return m


# ---------------------------------------------------------------------------------- reference
def masked_attention(q, k, v, mask, softmax_scale=None):
    """fp32 reference. q, k, v: (B, S, H, D) with H % H_kv == 0; mask: (S, S) bool."""
    qf, kf, vf = (x.float().transpose(1, 2) for x in (q, k, v))
    out = F.scaled_dot_product_attention(qf, kf, vf, attn_mask=mask, scale=softmax_scale,
                                         enable_gqa=qf.shape[1] != kf.shape[1])
    return out.transpose(1, 2).to(q.dtype)


if __name__ == "__main__":
    # Self-check against the brute-force loops in csrc/flash_attn/flash_attn2_check*.py
    S, G = 32, 4
    m = llm_scatter_mask(S, G)
    assert all(m[a, b] == (b <= a and a % G == b % G) for a in range(S) for b in range(S))
    m = llm_gather1_mask(S, G)
    assert all(m[a, b] == (b <= a and (a // G) % G == (b // G) % G) for a in range(S) for b in range(S))
    m = llm_window_mask(200, window=96, block=64)
    assert m[130, 128 - 32] and not m[130, 128 - 33] and m[63, 0] and not m[10, 11]
    W = 64
    m = image_scatter_mask(W)
    assert m.shape == (TEXT_LEN + W * W,) * 2 and m[:, :TEXT_LEN].all()
    assert m[TEXT_LEN:, TEXT_LEN:].sum(1).eq(W * W // 8).all()
    m = image_gather0_mask(W)
    assert m[TEXT_LEN:, TEXT_LEN:].sum(1).eq(256).all()
    q = TEXT_LEN + 9 * W + 9  # patch (1, 1) -> window rows/cols 4..19
    assert m[q, TEXT_LEN + 4 * W + 4] and not m[q, TEXT_LEN + 3 * W + 4] and m[q, TEXT_LEN + 19 * W + 19]
    print("reference masks OK")
