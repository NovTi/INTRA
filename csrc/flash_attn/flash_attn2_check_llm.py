import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

import time
import math
import numpy as np
from einops import rearrange, repeat
from tqdm import tqdm

from flash_attn import flash_attn_func



def make_mask_0_llama_prefill_causal(sequence_length, mask):
    group_num = 16

    for idx in range(sequence_length):
        gourp_idx = idx % group_num
        for i in range(idx+1):
            if i % group_num == gourp_idx:
                mask[:, :, idx, i] = 0

    return mask


def make_mask_0_llama_prefill(sequence_length, mask):
    group_num = 16

    for idx in range(sequence_length):
        gourp_idx = idx % group_num
        for i in range(sequence_length):
            if i % group_num == gourp_idx:
                mask[:, :, idx, i] = 0

    return mask



def make_mask_1_llama_prefill_causal(sequence_length, mask):
    group_size = 16
    group_num = 16

    for idx in range(sequence_length):
        group_idx = (idx // group_num) % group_num
        for i in range(idx+1):
            if (i // group_num) % group_num == group_idx:
                mask[:, :, idx, i] = 0
            
    return mask


def make_mask_1_llama_prefill(sequence_length, mask):
    group_size = 16
    group_num = 16

    for idx in range(sequence_length):
        group_idx = (idx // group_num) % group_num
        for i in range(sequence_length):
            if (i // group_num) % group_num == group_idx:
                mask[:, :, idx, i] = 0
            
    return mask




def attention_ref(
    q,
    k,
    v,
    mask=None,
    query_padding_mask=None,
    key_padding_mask=None,
    attn_bias=None,
    dropout_p=0.0,
    dropout_mask=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite window size
    softcap=0.0,
    upcast=True,
    reorder_ops=False,
    key_leftpad=None,
):
    if causal:
        window_size = (window_size[0], 0)
    dtype_og = q.dtype
    if upcast:
        q, k, v = q.float(), k.float(), v.float()
    
    seqlen_q, seqlen_k = q.shape[1], k.shape[1]
    k = repeat(k, "b s h d -> b s (h g) d", g=q.shape[2] // k.shape[2])
    v = repeat(v, "b s h d -> b s (h g) d", g=q.shape[2] // v.shape[2])

    d = q.shape[-1]

    scores = torch.einsum("bthd,bshd->bhts", q / math.sqrt(d), k)

    scores_check = torch.einsum("bthd,bshd->bhts", q, k)
    scores_check = scores_check + mask

    # scores_check[0, 0, 0, 0]
    # scores_check[0, 0, 0, 16]
    # import pdb; pdb.set_trace()

    if mask is not None:
        scores = scores + mask

    attention = torch.softmax(scores, dim=-1).to(v.dtype)
    # Some rows might be completely masked out so we fill them with zero instead of NaN
    if window_size[0] >= 0 or window_size[1] >= 0:
        attention = attention.masked_fill(torch.all(local_mask, dim=-1, keepdim=True), 0.0)
    # We want to mask here so that the attention matrix doesn't have any NaNs
    # Otherwise we'll get NaN in dV
    if query_padding_mask is not None:
        attention = attention.masked_fill(rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0)
    dropout_scaling = 1.0 / (1 - dropout_p)
    # attention_drop = attention.masked_fill(~dropout_mask, 0.0) * dropout_scaling
    # output = torch.einsum('bhts,bshd->bthd', attention_drop , v)
    if dropout_mask is not None:
        attention_drop = attention.masked_fill(~dropout_mask, 0.0)
    else:
        attention_drop = attention
    output = torch.einsum("bhts,bshd->bthd", attention_drop, v * dropout_scaling)

    if query_padding_mask is not None:
        output.masked_fill_(rearrange(~query_padding_mask, "b s -> b s 1 1"), 0.0)

    return output.to(dtype=dtype_og)




# mask0 = make_mask_0(4608, torch.full((1, 24, 4608, 4608), float('-inf'), dtype=torch.bfloat16)).cuda()
# mask1 = make_mask_1(4608, torch.full((1, 24, 4608, 4608), float('-inf'), dtype=torch.bfloat16)).cuda()



# ##### Single Output Check ######
# torch.manual_seed(0)

# # threshodl: 1280 = 1024 + 256  has problem with single block
# seq_length = 1545

# batch_size = 2
# head_dim = 128
# dtype = torch.bfloat16

# q = torch.randn(batch_size, seq_length, 32, head_dim, dtype=dtype, device='cuda')
# k = torch.randn(batch_size, seq_length, 8, head_dim, dtype=dtype, device='cuda')
# v = torch.randn(batch_size, seq_length, 8, head_dim, dtype=dtype, device='cuda')

# mask = make_mask_0_llama_prefill(seq_length, torch.full((1, 1, seq_length, seq_length), float('-inf'), dtype=torch.bfloat16)).cuda()
# # mask = make_mask_1_llama_prefill(seq_length, torch.full((1, 1, seq_length, seq_length), float('-inf'), dtype=torch.bfloat16)).cuda()

# for i in range(30):
#     out = flash_attn_func(q, k, v, mask=0)

#     out_ref = attention_ref(q, k, v, mask=mask)

#     result = torch.allclose(out_ref, out, rtol=1e-02, atol=1e-02)

#     print(f"seq: {seq_length} | batch: {batch_size} | i: {i} | result: {result}")

# # close_mask = torch.isclose(out_ref, out, rtol=1e-2, atol=1e-3)
# # num_not_close = torch.numel(close_mask) - close_mask.sum().item()
# # abs(out_ref - out).max()

# # import pdb; pdb.set_trace()

# # # compute-sanitizer --tool memcheck --target-processes all python flash_attn2_check_llm.py >> output.log 2>&1





###################### benchmark time batch ######################
def benchmark_batch(seq_length, width, head, mask_type, func_type, dtype=torch.float16, device='cuda:0'):
    time_lst = []
    batch_lst = [2, 2, 2, 2, 3, 5, 7, 10, 12, 15, 17, 20, 22, 25, 27, 30]
    # batch_lst = [2, 2, 2]
    for i in tqdm(batch_lst):
        start_time = time.time()
        batch = i

        q = torch.randn(batch, seq_length, head, 128, dtype=dtype, device=device)
        k = torch.randn(batch, seq_length, head, 128, dtype=dtype, device=device)
        v = torch.randn(batch, seq_length, head, 128, dtype=dtype, device=device)


        q_sdpa = torch.randn(batch, head, seq_length, 128, dtype=dtype, device=device)
        k_sdpa = torch.randn(batch, head, seq_length, 128, dtype=dtype, device=device)
        v_sdpa = torch.randn(batch, head, seq_length, 128, dtype=dtype, device=device)

        flash_attn_func(q, k, v, mask=mask_type, width=width, length=width)
        
        flash_attn_func(q, k, v, length=width, width=width, mask=0)

        # time.perf_counter()
        if func_type == 'org':
            start_time = time.time()
            for i in range(2000):
                F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa)
            end_time = time.time()

        else:
            start_time = time.time()
            for i in range(2000):
                flash_attn_func(q, k, v, mask=mask_type, width=width, length=width)
            end_time = time.time()

        time_lst.append(((end_time - start_time)/2000) * 1e6)

    return time_lst


def benchmark_length(width, head, mask_type, func_type, dtype=torch.float16):
    time_lst = []
    batch = 2
    length_lst = [64, 128, 256, 382, 512, 768, 1024]
    for i in tqdm(length_lst):
        seq_length = i * width + 512

        # time.perf_counter()
        if func_type == 'org':
            q_sdpa = torch.randn(batch, head, seq_length, 64, dtype=dtype, device='cuda')
            k_sdpa = torch.randn(batch, head, seq_length, 64, dtype=dtype, device='cuda')
            v_sdpa = torch.randn(batch, head, seq_length, 64, dtype=dtype, device='cuda')

            start_time = time.time()
            for i in range(2000):
                F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa)
            end_time = time.time()
        else:
            q = torch.randn(batch, seq_length, head, 64, dtype=dtype, device='cuda')
            k = torch.randn(batch, seq_length, head, 64, dtype=dtype, device='cuda')
            v = torch.randn(batch, seq_length, head, 64, dtype=dtype, device='cuda')

            start_time = time.time()
            for i in range(2000):
                flash_attn_func(q, k, v, mask=mask_type, width=width, length=i)

            end_time = time.time()

        time_lst.append((end_time - start_time) / 2)

    return time_lst




# batch_lst = [1, 1, 1, 2, 2, 3, 5, 7, 10, 12, 15, 17, 20, 22, 25, 27, 30]
# # batch_lst = [2, 2, 2]
# mask = 0
# for seq_length, width in [(4608, 64)]:
#     org_lst = benchmark_batch(seq_length, width, 24, mask, 'org', dtype=torch.bfloat16)
    
#     kernel_lst = benchmark_batch(seq_length, width, 24, mask, 'kernel', dtype=torch.bfloat16)

#     kernel_org_ratio = [kernel_lst[i] / org_lst[i] for i in range(len(org_lst))]

#     for i in range(len(kernel_org_ratio)):
#         print(f"seq_length: {seq_length} | mask: {mask} | batch: {batch_lst[i]} | kernel/org: {kernel_org_ratio[i]:.5f} | kernel_time: {kernel_lst[i]:.5f} | org_time: {org_lst[i]:.5f}")

    
#     print()


# length_lst = [64, 128, 256, 382, 512, 768, 1024]
# for width in [64]:
#     for mask in [0]:
#         head = 24
#         org_lst = benchmark_length(width, head, mask, 'org', dtype=torch.bfloat16)
#         kernel_lst = benchmark_length(width, head, mask, 'kernel', dtype=torch.bfloat16)

#         kernel_org_ratio = [kernel_lst[i] / org_lst[i] for i in range(len(org_lst))]
#         for i in range(len(kernel_org_ratio)):
#             print(f"seq_length: {length_lst[i]*width} | mask: {mask} | length: {length_lst[i]} | kernel/org: {kernel_org_ratio[i]:.5f}")

#         print()




# batch_lst = [2, 3, 5, 7, 10, 12, 15, 17, 20, 22, 25, 27, 30]
# for mask in [0, 1]:
#     for seq_length, width in [(1024, 32), (4096, 64)]:
#         org_lst = benchmark_batch(seq_length, width, mask, 'org')
#         kernel_lst = benchmark_batch(seq_length, width, mask, 'kernel')

#         kernel_org_ratio = [kernel_lst[i] / org_lst[i] for i in range(len(org_lst))]

#         for i in range(len(kernel_org_ratio)):
#             print(f"seq_length: {seq_length} | mask: {mask} | batch: {batch_lst[i]} | kernel/org: {kernel_org_ratio[i]:.5f}")

#         print()


# length_lst = [64, 128, 256, 382, 512, 768, 1024]
# width = 64
# for width in [32, 64]:
#     for mask in [0, 1]:
#         kernel_lst = benchmark_length(width, mask, 'kernel')
#         org_lst = benchmark_length(width, mask, 'org')

#         kernel_org_ratio = [kernel_lst[i] / org_lst[i] for i in range(len(org_lst))]
#         for i in range(len(kernel_org_ratio)):
#             print(f"seq_length: {length_lst[i]*width} | mask: {mask} | length: {length_lst[i]} | kernel/org: {kernel_org_ratio[i]:.5f}")

#         print()




##### Result Check ######
def result_check(seq_length, mask_type):
    if mask_type == 0:
        mask = make_mask_0_llama_prefill(seq_length, torch.full((1, 1, seq_length, seq_length), float('-inf'), dtype=torch.bfloat16)).cuda()
    else:
        mask = make_mask_1_llama_prefill(seq_length, torch.full((1, 1, seq_length, seq_length), float('-inf'), dtype=torch.bfloat16)).cuda()

    false_lst = []
    for batch_size in tqdm(range(2, 4)):
        for i in range(500):
            torch.manual_seed(i)
            q = torch.randn(batch_size, seq_length, 32, 128, dtype=torch.bfloat16, device='cuda')
            k = torch.randn(batch_size, seq_length, 8, 128, dtype=torch.bfloat16, device='cuda')
            v = torch.randn(batch_size, seq_length, 8, 128, dtype=torch.bfloat16, device='cuda')

            out = flash_attn_func(q, k, v, mask=mask_type)

            out_ref = attention_ref(q, k, v, mask=mask)

            result = torch.allclose(out_ref, out, rtol=1e-02, atol=1e-02)

            if not result:
                print(f"Not passed | seq: {seq_length} | mask: {mask_type} | batch_size: {batch_size}, i: {i}")
                # torch.max(torch.abs(out_ref - out))
                false_lst.append((batch_size, i))




for seq_length in [1545, 1853, 2745, 7261, 8911, 16384]:
    for mask in [1, 0]:
        result_check(seq_length, mask)



# @pytest.mark.parametrize("kvpacked", [True, False])
# # @pytest.mark.parametrize("kvpacked", [False])
# @pytest.mark.parametrize("dtype", ([torch.float16] if is_sm75 else [torch.float16, torch.bfloat16]))
# # @pytest.mark.parametrize("dtype", [torch.bfloat16])
# @pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
# # @pytest.mark.parametrize("mha_type", ["mha"])
# @pytest.mark.parametrize("deterministic", [False, True])
# # @pytest.mark.parametrize("deterministic", [True])
# @pytest.mark.parametrize("alibi", [False, True])
# # @pytest.mark.parametrize("alibi", [False])
# @pytest.mark.parametrize("local", [False, True])
# # @pytest.mark.parametrize("local", [False])
# @pytest.mark.parametrize("causal", [False, True])
# # @pytest.mark.parametrize("causal", [True])
# @pytest.mark.parametrize("d", [32, 40, 59, 64, 96, 111, 128, 160, 192, 224, 256])
# # @pytest.mark.parametrize("d", [32, 64, 96, 128, 160, 192, 224, 256])
# # @pytest.mark.parametrize('d', [32, 40, 64, 80, 96, 128, 160, 192])
# # @pytest.mark.parametrize('d', [32, 64, 96, 128, 160, 192])
# # @pytest.mark.parametrize('d', [56, 80])
# # @pytest.mark.parametrize("d", [64])
# @pytest.mark.parametrize(
#     "seqlen_q,seqlen_k",
#     [
#         (113, 203),
#         (128, 217),
#         (113, 211),
#         (108, 256),
#         (256, 512),
#         (512, 256),
#         (1024, 1024),
#         (1023, 1024),
#         (1024, 1023),
#         (2048, 2048),
#     ],
# )
# # @pytest.mark.parametrize('seqlen_q,seqlen_k', [(256, 128)])
# @pytest.mark.parametrize("dropout_p", [0.0, 0.17])
# # @pytest.mark.parametrize("dropout_p", [0.0])
# @pytest.mark.parametrize("softcap", [0.0, 50.0])
# def test_flash_attn_output(
#     seqlen_q, seqlen_k, d, dropout_p, causal, local, alibi, deterministic, mha_type, dtype, kvpacked, softcap
# ):
#     if (
#         max(seqlen_q, seqlen_k) >= 2048
#         and torch.cuda.get_device_properties("cuda").total_memory <= 16 * 2**30
#     ):
#         pytest.skip()  # Reference implementation OOM
#     if softcap > 0.0 and dropout_p > 0.0:
#         pytest.skip("Softcap and dropout not supported together")
#     device = "cuda"
#     # set seed
#     torch.random.manual_seed(0)
#     batch_size = 4
#     nheads = 6 if softcap == 0.0 else 4  # softcap reference impl takes more memory
#     nheads_k = nheads if mha_type == "mha" else (1 if mha_type == "mqa" else 2)
#     assert nheads % nheads_k == 0
#     window_size = (-1, -1) if not local else torch.randint(0, seqlen_k, (2,))
#     q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype, requires_grad=True)
#     if softcap > 0:
#         # Ensure the values of qk are at least within softcap range.
#         q = q * softcap
#     if kvpacked:
#         kv = torch.randn(
#             batch_size, seqlen_k, 2, nheads_k, d, device=device, dtype=dtype, requires_grad=True
#         )
#     else:
#         k = torch.randn(
#             batch_size, seqlen_k, nheads_k, d, device=device, dtype=dtype, requires_grad=True
#         )
#         v = torch.randn(
#             batch_size, seqlen_k, nheads_k, d, device=device, dtype=dtype, requires_grad=True
#         )
#     if alibi:
#         alibi_slopes = torch.rand(batch_size, nheads, device=device, dtype=torch.float32) * 0.3
#         attn_bias = attn_bias_from_alibi_slopes(alibi_slopes, seqlen_q, seqlen_k, causal=causal)
#     else:
#         alibi_slopes, attn_bias = None, None

#     if kvpacked:
#         out, lse, S_dmask = flash_attn_kvpacked_func(
#             q,
#             kv,
#             dropout_p,
#             causal=causal,
#             window_size=window_size,
#             softcap=softcap,
#             alibi_slopes=alibi_slopes,
#             deterministic=deterministic,
#             return_attn_probs=True,
#         )
#     else:
#         out, lse, S_dmask = flash_attn_func(
#             q,
#             k,
#             v,
#             dropout_p,
#             causal=causal,
#             window_size=window_size,
#             softcap=softcap,
#             alibi_slopes=alibi_slopes,
#             deterministic=deterministic,
#             return_attn_probs=True,
#         )
#     if dropout_p > 0.0:
#         S_dmask_converted = convert_flash_attn_S_to_softmax(
#             S_dmask,
#             seqlen_q,
#             seqlen_k,
#             None,
#             None,
#             d,
#             dropout_p > 0.0,
#             causal=causal,
#             window_size=window_size,
#         )
#         dropout_mask = S_dmask_converted >= 0
#         attn_unnorm = S_dmask_converted.abs()
#         if kvpacked:
#             kv_rep = repeat(kv, "b s two h d -> b s two (h g) d", g=nheads // nheads_k)
#             k_rep, v_rep = kv_rep.unbind(dim=2)
#         else:
#             k_rep = repeat(k, "b s h d -> b s (h g) d", g=nheads // nheads_k)
#             v_rep = repeat(v, "b s h d -> b s (h g) d", g=nheads // nheads_k)
#         attn = normalize_flash_attn_S(
#             attn_unnorm,
#             q,
#             k_rep,
#             v_rep,
#             None,
#             None,
#             attn_bias,
#             dropout_p > 0.0,
#             causal=causal,
#             window_size=window_size,
#         )
#         dropout_fraction = get_dropout_fraction(
#             dropout_mask, None, None, causal=causal, window_size=window_size
#         ).item()
#         print(f"Actual dropout fraction: {dropout_fraction}")
#     else:
#         dropout_mask = None

#     if kvpacked:
#         out_ref, attn_ref = attention_kvpacked_ref(
#             q,
#             kv,
#             None,
#             None,
#             attn_bias,
#             dropout_p,
#             dropout_mask,
#             causal=causal,
#             window_size=window_size,
#             softcap=softcap,
#         )
#         out_pt, attn_pt = attention_kvpacked_ref(
#             q,
#             kv,
#             None,
#             None,
#             attn_bias,
#             dropout_p,
#             dropout_mask,
#             causal=causal,
#             window_size=window_size,
#             softcap=softcap,
#             upcast=False,
#             reorder_ops=True,
#         )
#     else:
#         out_ref, attn_ref = attention_ref(
#             q,
#             k,
#             v,
#             None,
#             None,
#             attn_bias,
#             dropout_p,
#             dropout_mask,
#             causal=causal,
#             window_size=window_size,
#             softcap=softcap,
#         )
#         out_pt, attn_pt = attention_ref(
#             q,
#             k,
#             v,
#             None,
#             None,
#             attn_bias,
#             dropout_p,
#             dropout_mask,
#             causal=causal,
#             window_size=window_size,
#             softcap=softcap,
#             upcast=False,
#             reorder_ops=True,
#         )

#     print(f"Output max diff: {(out - out_ref).abs().max().item()}")
#     print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
#     print(f"Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")
#     print(f"Pytorch mean diff: {(out_pt - out_ref).abs().mean().item()}")
#     if dropout_p > 0.0:
#         print(f"Attention max diff: {(attn - attn_ref).abs().max().item()}")
#         print(f"Attention Pytorch max diff: {(attn_pt - attn_ref).abs().max().item()}")

#     g = torch.randn_like(out)
#     do_o = (g.float() * out.float()).sum(-1)
#     if (d <= MAX_HEADDIM_SM8x or dropout_p == 0) or (is_sm80 or is_sm90):
#         if kvpacked:
#             (
#                 dq,
#                 dkv,
#             ) = torch.autograd.grad(out, (q, kv), g)
#             dk, dv = dkv.unbind(2)
#             (
#                 dq_ref,
#                 dkv_ref,
#             ) = torch.autograd.grad(out_ref, (q, kv), g)
#             dk_ref, dv_ref = dkv_ref.unbind(2)
#             (
#                 dq_pt,
#                 dkv_pt,
#             ) = torch.autograd.grad(out_pt, (q, kv), g)
#             dk_pt, dv_pt = dkv_pt.unbind(2)
#         else:
#             (
#                 dq,
#                 dk,
#                 dv,
#             ) = torch.autograd.grad(out, (q, k, v), g)
#             (
#                 dq_ref,
#                 dk_ref,
#                 dv_ref,
#             ) = torch.autograd.grad(out_ref, (q, k, v), g)
#             (
#                 dq_pt,
#                 dk_pt,
#                 dv_pt,
#             ) = torch.autograd.grad(out_pt, (q, k, v), g)
#         print(f"dQ max diff: {(dq - dq_ref).abs().max().item()}")
#         print(f"dK max diff: {(dk - dk_ref).abs().max().item()}")
#         print(f"dV max diff: {(dv - dv_ref).abs().max().item()}")
#         print(f"dQ mean diff: {(dq - dq_ref).abs().mean().item()}")
#         print(f"dK mean diff: {(dk - dk_ref).abs().mean().item()}")
#         print(f"dV mean diff: {(dv - dv_ref).abs().mean().item()}")
#         print(f"dQ Pytorch max diff: {(dq_pt - dq_ref).abs().max().item()}")
#         print(f"dK Pytorch max diff: {(dk_pt - dk_ref).abs().max().item()}")
#         print(f"dV Pytorch max diff: {(dv_pt - dv_ref).abs().max().item()}")
#         print(f"dQ Pytorch mean diff: {(dq_pt - dq_ref).abs().mean().item()}")
#         print(f"dK Pytorch mean diff: {(dk_pt - dk_ref).abs().mean().item()}")
#         print(f"dV Pytorch mean diff: {(dv_pt - dv_ref).abs().mean().item()}")

#     # Check that FlashAttention's numerical error is at most twice the numerical error
#     # of a Pytorch implementation.
#     assert (out - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item()

#     if dropout_p > 0.0:
#         assert (attn - attn_ref).abs().max().item() <= 2 * (attn_pt - attn_ref).abs().max().item()
#         # With alibi, many of the prob values are 0.0 & -0.0 so dropout_fraction isn't accurate
#         if not alibi:
#             assert abs(dropout_fraction - dropout_p) <= (0.01 if not local else 0.025)

#     if (d <= MAX_HEADDIM_SM8x or dropout_p == 0) or (is_sm80 or is_sm90):
#         assert (dq - dq_ref).abs().max().item() <= 3 * (dq_pt - dq_ref).abs().max().item()
#         assert (dk - dk_ref).abs().max().item() <= 3 * (dk_pt - dk_ref).abs().max().item()
#         assert (dv - dv_ref).abs().max().item() <= 3 * (dv_pt - dv_ref).abs().max().item()

