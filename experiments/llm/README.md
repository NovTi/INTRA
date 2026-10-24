# INTRA for language models (LLaMA-3.1-8B / Qwen3-8B)

Code for the language-model experiments of *INTRA: Interleaved Non-contiguous Token Sparse
Attention*: LoRA self-distillation of sparse-attention students, needle-in-a-haystack (NIAH)
retrieval, SCROLLS evaluation and the MoBA / NSA / MInference / XAttention baselines.

The paper INTRA configuration for LLaMA-3.1-8B (`g2s4`) uses:

- block-level Scatter with 2 groups (256-token blocks, two stripe offsets) in 4 of the 32 layers,
  with layer pattern `SWWWWWWW`, so layers 0, 8, 16 and 24 are Scatter;
- a 575-token sliding window plus 64 attention-sink tokens in the other 28 layers.

The adapter is rank-16 LoRA, trained for 1200 steps on 16K-token blocks against the model's own
dense output.

## Files

| file | purpose |
|---|---|
| `llama_intra.py` | INTRA attention for HF transformers 5.x: per-layer S/W/D pattern, block Scatter (loop or FlexAttention), FlexAttention sliding window with sinks. Registered as attention implementation `"intra"`; decode falls back to dense. `python llama_intra.py` runs a CPU self-test against a dense-mask oracle. |
| `intra_token_attn.py` | Kernel-semantics INTRA reference (token-stride Scatter, 64-aligned window), used by `run_eval.py --backend intra`. Has a CPU self-test. |
| `moba_attn.py`, `moba_student.py` | MoBA baseline: a chunked port of MoBA's `moba_naive`, and its registration as a student. |
| `nsa_attn.py`, `nsa_student.py` | NSA baseline (compression + selection + window branches, combined in one softmax) and its registration as a student. |
| `train_llama_distill.py` | LoRA self-distillation trainer (teacher = same weights, adapter off, dense attention). |
| `niah_eval.py`, `niah_table.py` | Teacher-forced NIAH over a PG19 haystack, and the per-length summary table. |
| `build_data.py` | SCROLLS evaluation file: 6 tasks x 200 validation records, with prompts and generation budgets. |
| `pack_scrolls_mix.py`, `pack_retrieval.py` | The `mixgen` distillation corpus: 400 blocks x 16384 tokens (task / retrieval / plain), with a leakage guard against the eval file. |
| `run_eval.py`, `score.py`, `metrics.py` | SCROLLS generation for every backend, and scoring (max over all references). |
| `density.py` | Retained attention density of a layer stack (the density column of Tables 5 and 16). |
| `scripts/common.sh` | Shared settings and the arm specification (`arm_flags`). |
| `scripts/prepare_data.sh` | Downloads SCROLLS and PG19, builds the eval file and the training corpus. |
| `scripts/train_arm.sh`, `scripts/niah.sh`, `scripts/scrolls_eval.sh` | Train, NIAH-evaluate, or SCROLLS-evaluate one arm. |
| `scripts/reproduce_table4.sh` | Every arm behind Tables 4a, 5, 15 and 16. |
| `llamafactory_scrolls/` | The LLaMA-Factory pipeline behind the SCROLLS fine-tuning results (Table 3b / 14); see its README. |

## Setup

```bash
pip install -r requirements.txt
huggingface-cli login        # meta-llama/Llama-3.1-8B-Instruct is gated
```

All scripts read their paths from the environment. The defaults are:

| variable | default | meaning |
|---|---|---|
| `MODEL` | `meta-llama/Llama-3.1-8B-Instruct` | HF id or local path; `Qwen/Qwen3-8B` for the Qwen3 runs |
| `DATA_DIR` | `experiments/llm/data` | downloads and generated data |
| `OUT_DIR` | `experiments/llm/outputs` | adapters (`runs/<arm>/`), NIAH and SCROLLS results |
| `SCROLLS_DIR` | `$DATA_DIR/scrolls` | `<task>.{train,validation}.jsonl` |
| `EVAL_FILE` | `$DATA_DIR/scrolls6_n200.jsonl` | SCROLLS eval file |
| `CORPUS` | `$DATA_DIR/mixgen_16k_<model basename>` | training corpus (tokenizer-specific) |
| `HAYSTACK` | `$DATA_DIR/pg19/*.parquet` | NIAH haystack |
| `NGPU` | `8` | GPUs for training and SCROLLS evaluation (NIAH uses GPUs 0-3) |
| `PYTHON` | `python` | interpreter (point it at the MInference/XAttention environment for those arms) |

## Data

```bash
bash scripts/prepare_data.sh                     # LLaMA-3.1 corpus
MODEL=Qwen/Qwen3-8B bash scripts/prepare_data.sh # Qwen3 corpus (downloads are reused)
```

This script:

1. downloads the official SCROLLS train and validation splits (`tau/scrolls`);
2. downloads the PG19 test split (`emozilla/pg19-test`, parquet), which is the NIAH haystack;
3. builds the eval file: 6 tasks x 200 validation records, seed 1234;
4. packs `mixgen`: 240 task, 100 retrieval and 60 plain blocks of 16384 tokens, seed 7. Training
   documents that collide with an eval document are dropped.

## Arms

Each experiment arm is described by environment variables. The same variables drive training, NIAH
and SCROLLS evaluation, so all three always use the same mask.

| variable | meaning |
|---|---|
| `ARM` | run name; the adapter is written to `$OUT_DIR/runs/$ARM/lora_final.pt` |
| `STUDENT` | `intra` (default), `dense`, `moba`, `nsa`, or the training-free `minference` / `xattention` |
| `NGROUPS`, `NOFF`, `PATTERN`, `WINDOW` | `intra` only: Scatter groups, stripe offsets (1 = none), layer pattern over `S`/`W`/`D` repeated over depth, window size (default 575) |
| `LORA` | adapter for evaluation (default: the arm's own); `NONE` evaluates without one |
| `STEPS` | training steps (default 1200) |

```bash
ARM=g2s4 NGROUPS=2 NOFF=2 PATTERN=SWWWWWWW bash scripts/train_arm.sh    # ~1 h on 8x A100
ARM=g2s4 NGROUPS=2 NOFF=2 PATTERN=SWWWWWWW bash scripts/niah.sh
ARM=g2s4 NGROUPS=2 NOFF=2 PATTERN=SWWWWWWW bash scripts/scrolls_eval.sh
```

## Paper result to command

`scripts/reproduce_table4.sh` trains and NIAH-evaluates every arm below. It prints a table with one
column per length (4K, 8K, 16K, 32K) and the mean. "Mean NIAH" in the paper is that mean.

| paper | quantity | arm(s) | spec |
|---|---|---|---|
| Table 4a | Dense + LoRA | `dense` | `STUDENT=dense` |
| Table 4a | MoBA | `moba` | `STUDENT=moba` (density 0.125, chunk 2048) |
| Table 4a | NSA | `nsa` | `STUDENT=nsa` (nominal density 0.1438, measured about 0.126) |
| Table 4a, 5, 16 | INTRA (mean NIAH and 32K value) | mean of `g2s4`, `g2s4b`, `g2s4c` | `NGROUPS=2 NOFF=2 PATTERN=SWWWWWWW` |
| Table 5 | sliding window in the same four layers | `wonly575` | `NGROUPS=2 NOFF=1 PATTERN=WWWWWWWW` |
| Table 5 | dense in the same four layers | `dense4` | `NGROUPS=2 NOFF=1 PATTERN=DWWWWWWW` |
| Table 15 | window width 575 / 2048 / 4096 (4K and 32K columns) | `wonly575`, `wonly2048`, `wonly4096` | `PATTERN=WWWWWWWW WINDOW=575/2048/4096` |
| Table 16 | sparse attention in 16 layers | `baseb` | `NGROUPS=8 NOFF=1 PATTERN=SSWW` |
| Table 16 | wider windows | `wonly2048` | as above |
| Tables 5, 16 | retained density | all of the above | `python density.py` |
| Appendix G.5 | NIAH without LoRA adaptation | any arm | same spec plus `LORA=NONE`, with `scripts/niah.sh` |

Notes:

- **Seeds.** `g2s4`, `g2s4b` and `g2s4c` are three runs of the same command (different LoRA
  initialisation and data order); the paper reports their mean.
- **GPUs.** All reported adapters were trained with `NGPU=8`.
- **Density.** `density.py` counts attended pairs exactly. It matches the paper's values to within
  0.0015; the paper rounds and uses the nominal Scatter density 0.5 for `g2s4`.
- **Qwen3-8B.** The same recipe applies with `MODEL=Qwen/Qwen3-8B`. With `PATTERN=SWWWWWWW`, 5 of
  Qwen3's 36 layers are Scatter:
  `MODEL=Qwen/Qwen3-8B ARM=q3g2s4 NGROUPS=2 NOFF=2 PATTERN=SWWWWWWW bash scripts/train_arm.sh`,
  then the same with `scripts/niah.sh`.
- **MInference and XAttention** are training-free. They come from their own repositories and need a
  separate transformers 4.46.3 environment (see `requirements.txt`). Run them through the same
  harness, for example
  `ARM=minference STUDENT=minference PYTHON=/path/to/env/bin/python bash scripts/niah.sh`.
  We do not redistribute or modify their code. `--backend minference` falls back to an SDPA
  stand-in when flash-attn is not installed.

## Other results

- **SCROLLS (Table 3b, Table 14):** fine-tuning with the LLaMA-Factory pipeline in
  `llamafactory_scrolls/`; SCROLLS generation and scoring with `scripts/scrolls_eval.sh`
  (`run_eval.py` + `score.py`).
- **Timing results** (Table 3a, Table 4b, Table 17) are kernel benchmarks: see `benchmarks/` at the
  repository root.

## Self-tests (CPU)

```bash
python llama_intra.py      # Scatter (loop + flex) and window vs. a dense-mask oracle
python intra_token_attn.py
python nsa_attn.py
python metrics.py
python density.py
MOBA_REPO=/path/to/MoBA python moba_attn.py   # compares with MoBA's moba_naive
python llamafactory_scrolls/make_window_mask.py --selftest
```
