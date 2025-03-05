# Install the newest triton version with
# pip install "git+https://github.com/openai/triton.git#egg=triton&subdirectory=python"
import pickle
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.benchmark as benchmark

from einops import rearrange, repeat
from tqdm import tqdm

from flash_attn import flash_attn_func



def flops(batch, seqlen, headdim, nheads, causal, mode="fwd"):
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    f = 4 * batch * seqlen**2 * nheads * headdim // (2 if causal else 1)
    return f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)

def efficiency(flop, time):
    return (flop / time / 10**12) if not math.isnan(time) else 0.0


def benchmark_forward(
    fn, *inputs, repeats=1000, desc="", verbose=False, amp=False, amp_dtype=torch.float16, **kwinputs
):
    """Use Pytorch Benchmark on the forward pass of an arbitrary function."""
    if verbose:
        print(desc, "- Forward pass")

    def amp_wrapper(*inputs, **kwinputs):
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp):
            fn(*inputs, **kwinputs)

    t = benchmark.Timer(
        stmt="fn_amp(*inputs, **kwinputs)",
        globals={"fn_amp": amp_wrapper, "inputs": inputs, "kwinputs": kwinputs},
        num_threads=torch.get_num_threads(),
    )
    m = t.timeit(repeats)
    if verbose:
        print(m)
    return t, m


device = torch.device("cuda")
batch_lst = [2, 3, 5, 7, 10, 12, 15, 17, 20, 22, 25, 27, 30]
for batch in tqdm(batch_lst):
    seq_length = 4608
    head = 24
    dtype = torch.bfloat16

    q_i = torch.randn(batch, seq_length, head, 128, dtype=dtype, device=device)
    k_i = torch.randn(batch, seq_length, head, 128, dtype=dtype, device=device)
    v_i = torch.randn(batch, seq_length, head, 128, dtype=dtype, device=device)


    q_sdpa = torch.randn(batch, head, seq_length, 128, dtype=dtype, device=device)
    k_sdpa = torch.randn(batch, head, seq_length, 128, dtype=dtype, device=device)
    v_sdpa = torch.randn(batch, head, seq_length, 128, dtype=dtype, device=device)

    t_i, m_i = benchmark_forward(
        flash_attn_func, q_i, k_i, v_i, mask=0, width=64, length=64, 
        repeats=1000, desc="Flash Attention Benchmark"
    )

    t_sdpa, m_sdpa = benchmark_forward(
        F.scaled_dot_product_attention, q_sdpa, k_sdpa, v_sdpa,
        repeats=1000, desc="Flash Attention Benchmark"
    )

    m_i = m_i.mean * 1000
    m_sdpa = m_sdpa.mean * 1000

    print(f"org: {m_sdpa:.4f} ms | i att: {m_i:.4f} ms | i att / org: {m_i/m_sdpa:.4f}")