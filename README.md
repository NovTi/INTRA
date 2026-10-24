# INTRA: Interleaved Non-contiguous Token Sparse Attention

Code for the paper **INTRA: Interleaved Non-contiguous Token Sparse Attention**
(Yuxuan Xia, Wenbo Lu, Shaoyi Zheng, Shengjie Wang — New York University).

INTRA is a hardware-aware *static* sparse attention design. Layers alternate between
complementary sparse groups: **Scatter** groups sample one token from every local neighbourhood
(a global, low-resolution relay), and **Gather** groups reconnect those channels in the next
layers, so information can reach any token through short mediated paths. Each group is a
Computational Query Set (CQS): its non-contiguous tokens are located by a structured index map at
runtime, packed into dense shared-memory tiles, and processed by a FlashAttention-style kernel, so
no work is spent on masked-out scores.

This repository contains

| path | what |
|---|---|
| `csrc/`, `intra_attn/`, `setup.py` | CUDA kernels (a FlashAttention-2 v2.7.0 fork) and their Python API |
| `intra_attn/reference.py` | dense-mask specification of every pattern (the ground truth for the tests) |
| `tests/test_kernel.py` | kernel-vs-reference parity for every pattern |
| `benchmarks/` | attention latency: INTRA vs FlashAttention-2, MoBA and NSA |
| `experiments/flux/` | FLUX.1-dev: INTRA attention processor and LoRA self-distillation |
| `experiments/llm/` | LLaMA-3.1-8B / Qwen3-8B: INTRA attention, LoRA self-distillation, NIAH and SCROLLS |
| `experiments/wan/` | Wan2.2-TI2V-5B: INTRA video attention, distillation, VBench and speed |

## Installation

Requirements: an NVIDIA GPU with compute capability 8.0 or newer (developed and measured on A100;
also tested on H100), CUDA >= 11.7, PyTorch >= 2.4 and `ninja`.

```bash
git clone --recursive https://github.com/NovTi/IN-Attention.git
cd IN-Attention
pip install --no-build-isolation .          # builds intra_attn_cuda (sm80, plus sm90/sm100 cubins)
python tests/test_kernel.py                 # every pattern against the fp32 dense reference
```

The kernels are compiled for bfloat16 with head dimension 128. They are forward-only; training
uses the PyTorch implementations of the same patterns in `experiments/`.

## Kernel API

```python
import torch
from intra_attn import intra_llm_attn, intra_image_attn

# Language prefill (causal). q: (B, S, H, 128), k/v: (B, S, H_kv, 128), bfloat16, H % H_kv == 0.
with torch.no_grad():
    o = intra_llm_attn(q, k, v, "scatter", group_num=8)   # token i <-> tokens j with j % 8 == i % 8
    o = intra_llm_attn(q, k, v, "gather1", group_num=8)   # segments of 8 tokens, segment s in group s % 8
    o = intra_llm_attn(q, k, v, "window", window=576)     # 64-query blocks + preceding window-64 tokens

# FLUX image attention (bidirectional): 512 text tokens followed by a width x width latent grid,
# width 64 (1024 x 1024 images) or 128 (2048 x 2048 images).
with torch.no_grad():
    o = intra_image_attn(q, k, v, "scatter", width=64)    # 8 groups on a 2 x 4 sub-lattice, text dense
    o = intra_image_attn(q, k, v, "gather0", width=64)    # 16 x 16 window around each 8 x 8 patch
```

| pattern | paper | supported shapes |
|---|---|---|
| `intra_llm_attn(..., "scatter")` | Scatter, App. C.1 | any `S`, any `group_num` |
| `intra_llm_attn(..., "gather1")` | Gather 1, App. A | `group_num=8`, `S % 64 == 0` |
| `intra_llm_attn(..., "window")` | local window layers | any `S`, any `window` |
| `intra_image_attn(..., "scatter")` | Scatter (image), Fig. 3 | 512 text + 64x64 or 128x128 image tokens |
| `intra_image_attn(..., "gather0")` | Gather 0, App. C.2 | 512 text + 64x64 or 128x128 image tokens |

`intra_attn/reference.py` builds the exact boolean mask of each pattern; `tests/test_kernel.py`
checks every kernel against it (max abs error ≤ 1.6e-2, i.e. bf16 rounding, on A100 and H100).

## Benchmarks

```bash
python benchmarks/bench_kernel.py llm      # Scatter / Gather 1 / window vs FA2, plus the 32-layer g2s4 total
python benchmarks/bench_kernel.py image    # FLUX 1K / 2K-image Scatter / Gather 0 vs FA2
```

Reference results on one A100-SXM4-80GB (PyTorch 2.9.1, CUDA 12.9; batch 1, 32 query / 8 KV
heads for language, 24 heads for image; median of 50 calls; full output in `benchmarks/results/`):

| ms | 4K | 8K | 16K | 32K | 64K |
|---|---|---|---|---|---|
| FlashAttention-2 (causal) | 1.01 | 3.38 | 12.35 | 47.31 | 189.5 |
| INTRA Scatter, G=8 | 0.17 | 0.53 | 1.89 | 7.28 | 28.4 |
| INTRA Gather 1, G=8 | 0.16 | 0.50 | 1.79 | 6.80 | 26.7 |
| INTRA window 576 | 0.27 | 0.54 | 1.07 | 2.16 | 4.42 |
| 32 layers, dense | 32.3 | 108.2 | 395.3 | 1513.9 | 6063.8 |
| 32 layers, INTRA g2s4 (4 x Scatter G=2 + 28 x window) | 9.5 | 22.4 | 58.5 | 172.0 | 573.7 |

| ms, FLUX image attention (24 heads) | FlashAttention-2 | INTRA Scatter | INTRA Gather 0 | INTRA, alternating (mean) |
|---|---|---|---|---|
| 1024 x 1024 (512 + 4096 tokens) | 1.39 | 0.37 | 0.31 | 0.34 |
| 2048 x 2048 (512 + 16384 tokens) | 17.86 | 3.25 | 1.15 | 2.20 |

MoBA and NSA baselines (Table 4b / 17) are timed with `benchmarks/baselines/moba_bench.py` and
`benchmarks/baselines/nsa_bench.py`, which use the official MoBA kernel and the
Flash-Sparse-Attention NSA kernels respectively (setup instructions in each file's docstring).

## Reproducing the paper

Each experiment directory has its own README with setup, exact hyperparameters and a
"paper number -> command" table.

| paper | where |
|---|---|
| Table 1, Table 3(a), Tables 7-9 (single-attention latency) | `benchmarks/bench_kernel.py` |
| Table 4(b), Table 17 (INTRA vs MoBA vs NSA attention latency) | `benchmarks/bench_kernel.py llm`, `benchmarks/baselines/` |
| Table 2, Table 6, Table 13, Figures 6-8 (FLUX.1-dev) | `experiments/flux/` |
| Table 3(b), Table 4(a), Table 5, Tables 14-16 (LLaMA-3.1-8B / Qwen3-8B) | `experiments/llm/` |
| Table 18, Table 19 (Wan2.2-TI2V-5B) | `experiments/wan/` |
| Appendix C (index maps), Appendix A (token-level patterns) | `intra_attn/reference.py` |

## Limitations of this release

- The kernels are forward-only, bfloat16, head dimension 128, and written for sm80; the sm90/sm100
  builds run the same Ampere code path.
- The image kernels are instantiated for 64 x 64 and 128 x 128 latent grids (1024 x 1024 and
  2048 x 2048 FLUX images) with 512 text tokens.

## Acknowledgements

The kernels are built on [FlashAttention-2](https://github.com/Dao-AILab/flash-attention) and
[CUTLASS](https://github.com/NVIDIA/cutlass). This repository keeps FlashAttention's BSD 3-Clause
license (see `LICENSE` and `AUTHORS`).

## Citation

```bibtex
@article{xia2026intra,
  title   = {{INTRA}: Interleaved Non-contiguous Token Sparse Attention},
  author  = {Xia, Yuxuan and Lu, Wenbo and Zheng, Shaoyi and Wang, Shengjie},
  journal = {arXiv preprint},
  year    = {2026}
}
```
