# Build script for the INTRA sparse-attention CUDA kernels.
# Derived from the FlashAttention-2 (v2.7.0) setup.py, Copyright (c) 2023, Tri Dao.
import os
import subprocess
from pathlib import Path

from packaging.version import Version
from setuptools import setup
from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CUDAExtension

this_dir = Path(__file__).resolve().parent
src = "csrc/flash_attn/src"


def nvcc_version():
    out = subprocess.check_output([os.path.join(CUDA_HOME, "bin", "nvcc"), "-V"], text=True)
    return Version(out.split("release ")[1].split(",")[0])


if CUDA_HOME is None:
    raise RuntimeError("CUDA_HOME is not set; INTRA needs nvcc (CUDA >= 11.7) to build.")
cuda = nvcc_version()
if cuda < Version("11.7"):
    raise RuntimeError(f"INTRA needs CUDA >= 11.7, found {cuda}.")

# The kernels are sm80 (Ampere) code. sm90/sm100 cubins of the same code are added so the
# extension also loads on Hopper/Blackwell; they are not tuned for those architectures.
cc_flag = ["-gencode", "arch=compute_80,code=sm_80"]
if cuda >= Version("11.8"):
    cc_flag += ["-gencode", "arch=compute_90,code=sm_90"]
if cuda >= Version("12.8"):
    cc_flag += ["-gencode", "arch=compute_100,code=sm_100"]

nvcc_threads = os.getenv("NVCC_THREADS", "4")

ext = CUDAExtension(
    name="intra_attn_cuda",
    sources=[
        "csrc/flash_attn/flash_api.cpp",
        # Image (FLUX), bidirectional, 512 dense text tokens in front of the image tokens.
        f"{src}/flash_fwd_hdim128_bf16_sm80_mask0_8group_text_front.cu",  # Scatter (8 groups)
        f"{src}/flash_fwd_hdim128_bf16_sm80_masksw_text_front.cu",        # Gather 0 (16x16)
        # Language, causal prefill.
        f"{src}/flash_fwd_hdim128_bf16_sm80_mask0_16groups_llm_causal.cu",  # Scatter
        f"{src}/flash_fwd_hdim128_bf16_sm80_mask1_16groups_llm_causal.cu",  # Gather 1
        f"{src}/flash_fwd_hdim128_bf16_sm80_masksw_llm_causal.cu",          # sliding window
    ],
    extra_compile_args={
        "cxx": ["-O3", "-std=c++17"],
        "nvcc": [
            "-O3",
            "-std=c++17",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "--use_fast_math",
            "--threads",
            nvcc_threads,
        ]
        + cc_flag,
    },
    include_dirs=[
        this_dir / "csrc" / "flash_attn",
        this_dir / "csrc" / "flash_attn" / "src",
        this_dir / "csrc" / "cutlass" / "include",
    ],
)

setup(
    name="intra_attn",
    version="0.1.0",
    description="INTRA: Interleaved Non-contiguous Token spaRse Attention (CUDA kernels)",
    packages=["intra_attn"],
    ext_modules=[ext],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.9",
    install_requires=["torch"],
)
