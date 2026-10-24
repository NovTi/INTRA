"""INTRA sparse attention processor for FLUX.1-dev (diffusers 0.33).

FLUX attends over a joint sequence of 512 T5 text tokens followed by a width x width grid of image
tokens (64 x 64 at 1024 x 1024, 128 x 128 at 2048 x 2048). The student uses two static patterns,
switching every two layers:

  Scatter : 8 groups; image token (r, c) is in group (r % 2) * 4 + c % 4, text chunk j (64 tokens)
            rides with group j. Every query sees all text keys plus the image keys of its group.
  Gather 0: each 8 x 8 patch of image queries sees a 16 x 16 window of image keys around it
            (shifted inwards at the borders) plus all text keys. Text queries keep their Scatter rows.

The masks come from intra_attn/reference.py, the same specification the CUDA kernel is tested
against. Training uses SDPA with the dense additive {0, -inf} mask (differentiable). Under
torch.no_grad(), the forward-only INTRA CUDA kernel is used when it is built (1024^2 and 2048^2).
"""
import math
from functools import lru_cache
from typing import Optional

import torch
import torch.nn.functional as F

from intra_attn.reference import TEXT_LEN, image_gather0_mask, image_scatter_mask

try:
    import intra_attn_cuda  # noqa: F401  (compiled extension present)
    from intra_attn import intra_image_attn
except ImportError:
    intra_image_attn = None

KERNEL_WIDTHS = (64, 128)  # latent sides the CUDA kernel is compiled for (1024^2 and 2048^2 images)


@lru_cache(maxsize=None)
def get_mask(pattern, width, device):
    """Additive bf16 {0, -inf} mask of shape (1, 1, S, S), S = 512 + width^2, cached per device."""
    if pattern == "scatter":
        allowed = image_scatter_mask(width)
    elif pattern == "gather0":
        allowed = image_gather0_mask(width)
    else:
        raise ValueError(f"unknown INTRA image pattern {pattern!r}")
    mask = torch.zeros(allowed.shape, dtype=torch.bfloat16).masked_fill_(~allowed, float("-inf"))
    return mask[None, None].to(device)


def latent_width(seq_len):
    width = math.isqrt(seq_len - TEXT_LEN)
    if width * width != seq_len - TEXT_LEN or width % 8:
        raise ValueError(f"expected 512 text + a square latent grid with side % 8 == 0, got {seq_len}")
    return width


class FluxINTRAAttnProcessor:
    """FLUX joint attention with an INTRA sparse pattern ("scatter" or "gather0").

    Same projections / QK-norm / RoPE as diffusers' FluxAttnProcessor2_0; only the attention call differs.
    """

    def __init__(self, pattern="scatter", use_kernel=True):
        self.pattern = pattern
        self.use_kernel = use_kernel

    def __call__(
        self,
        attn,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        # (text and image are already concatenated, text first)
        if encoder_hidden_states is not None:
            # `context` projections.
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            # attention: sequence is [text (512) | image (width^2)]
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        attention_scale = math.sqrt(1 / head_dim)

        width = latent_width(query.shape[2])
        if (self.use_kernel and intra_image_attn is not None and not torch.is_grad_enabled()
                and query.is_cuda and query.dtype == torch.bfloat16 and head_dim == 128
                and width in KERNEL_WIDTHS):
            # forward-only CUDA kernel; it takes (B, S, H, D)
            hidden_states = intra_image_attn(
                query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2),
                pattern=self.pattern, width=width, softmax_scale=attention_scale,
            ).transpose(1, 2)
        else:
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, dropout_p=0.0, is_causal=False, scale=attention_scale,
                attn_mask=get_mask(self.pattern, width, query.device),
            )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


def set_intra_processors(transformer, use_kernel=True):
    """Install INTRA on all 57 FLUX attention layers (19 double- then 38 single-stream blocks),
    switching pattern every two layers: Scatter, Scatter, Gather 0, Gather 0, ..."""
    processors = {}
    for idx, name in enumerate(transformer.attn_processors.keys()):
        pattern = "gather0" if idx % 4 in (2, 3) else "scatter"
        processors[name] = FluxINTRAAttnProcessor(pattern, use_kernel=use_kernel)
    transformer.set_attn_processor(processors)
