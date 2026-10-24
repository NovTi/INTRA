# INTRA on Wan2.2-TI2V-5B (video)

Code for the video-transfer experiment of the paper (Section 6.1, Appendix G.8, Tables 18 and 19):
INTRA sparse self-attention in the Wan2.2-TI2V-5B diffusion transformer, DoRA self-distillation
from the dense model, VBench evaluation and end-to-end speed measurements. Generation is
text-to-video only (no conditioning image).

## Attention pattern

Wan's `attn1` is bidirectional self-attention over the latent token grid `(T, H, W)` (VAE 4x
temporal / 16x spatial, patch 1x2x2; 480x832x49 frames -> (13, 15, 26) = 5,070 tokens, 704x1280x121
-> (31, 22, 40) = 27,280). INTRA replaces `attn1` in blocks 1-28 with a period-5 schedule; blocks
0 and 29 and all cross-attention (`attn2`) stay dense.

| `layer % 5` | layer | tokens attended |
|---|---|---|
| 0 | Scatter-A | same `(h mod 2, w mod 4)` residue, all frames (8 groups) |
| 1 | Gather | same spatial tile, all frames (tile auto-chosen: (5,13) at 480p, (11,10) at 720p) |
| 2 | Scatter-B | same `(h mod 4, w mod 2)` residue, all frames (8 groups) |
| 3 | Gather-shift | as Gather, tiles torus-shifted by half a tile |
| 4 | Cone | 1-D window of radius `4*H*W + H*W - 1` on the T-major sequence (covers frames within +/-4) |

There are two implementations of this one pattern:

* `intra_wan.py: sparse_self_attention` is the **reference**: it gathers each group with
  `index_select` and runs exact softmax attention (the Cone uses a band mask). It supports backward
  and was used for training and for every VBench clip. It is slower than dense attention.
* `intra_wan_fast.py: fast_sparse_self_attention` is the **speed path** (forward only, CUDA): it
  permutes tokens group by group, runs one varlen FlashAttention-2 call (the FA2 build bundled with
  PyTorch, the same kernel dense SDPA uses on A100), and inverts the permutation; the Cone is FA2
  with `window_size=(R, R)`. All Table 19 timings use this path. It does not use the CUDA kernels in
  `intra_attn/` at the repository root.

`bench_speed.py parity` checks that both agree with an fp32 masked oracle; `results/parity.json`
has our A100 run.

## Files

| file | purpose |
|---|---|
| `intra_wan.py` | pattern, reference attention, diffusers 0.37 `attn1` processor, DoRA helper, CPU self-test |
| `intra_wan_fast.py` | FlashAttention-2 speed path for the same pattern |
| `selfgen.py` | teacher data: templated prompts, dense Wan latents, empty-prompt embedding |
| `train.py` | DoRA self-distillation (dense teacher vs INTRA student), DDP |
| `gen_vbench.py` | VBench clip generation, dense or INTRA + adapter |
| `vbench_score.py` | per-dimension VBench scoring and the Table 18 summary |
| `bench_speed.py` | parity / processor check / attention, DiT-forward and end-to-end timing |
| `scripts/*.sh` | the exact commands: `selfgen`, `train`, `gen_vbench`, `score_vbench`, `bench_speed` |
| `results/*.json` | our Table 19 measurements (A100-SXM4-80GB, torch 2.11.0+cu129, diffusers 0.37.0) |

## Setup

```bash
pip install -r requirements.txt
python intra_wan.py          # CPU self-test: every layer kind == dense masked attention (float64)
```

All scripts take `MODEL` (default `Wan-AI/Wan2.2-TI2V-5B-Diffusers`, a Hugging Face id or a local
copy; download it once with `huggingface-cli download Wan-AI/Wan2.2-TI2V-5B-Diffusers` before
launching multi-GPU jobs), `OUT_DIR` (default `outputs/`) and `NGPU` (default 8), and run from any
directory. Everything is bf16.

## 1. Teacher data

```bash
bash scripts/selfgen.sh      # -> $OUT_DIR/teacher_clips/{prompts.json, uncond_embeds.pt, clip_00000..00499.pt}
```

600 prompts are built deterministically from subject x action x setting templates with a camera
move (seed 1234), so all clips contain motion and none overlap VBench. Prompts 0-499 are generated
with dense Wan (480x832, 49 frames, 50 steps, guidance 5.0, seed `1000 + index`); prompts 500-599
are held out. Each clip stores the final latent (no VAE round trip) and the prompt's UMT5
embedding. `uncond_embeds.pt` is the embedding of the empty negative prompt that `WanPipeline` uses
for classifier-free guidance.

## 2. Training

```bash
bash scripts/train.sh        # -> $OUT_DIR/intra_r32/lora_final.pt (+ lora_step*.pt every 1000 steps)
```

The recipe of the checkpoint behind Table 18:

| | |
|---|---|
| student | frozen Wan2.2-TI2V-5B + INTRA (scatter (2,4)/(4,2), auto tiles, cone W_t = 4, blocks 0/29 dense) |
| adapters | DoRA rank 32, alpha 32, on `attn1.to_{q,k,v,out.0}`, `ffn.net.0.proj`, `ffn.net.2` |
| objective | MSE between student and dense-teacher outputs on the same noisy latent |
| noise | sigma = 5u / (1 + 4u), u ~ logit-normal(0, 1); x_t = (1 - sigma) x0 + sigma * noise |
| text | clip prompt, or with probability 0.5 the empty-prompt embedding (trains the CFG branch) |
| data | the 500 teacher clips of step 1 |
| optimisation | Prodigy (lr 1, weight decay 1e-4), grad-norm clip 1, 20,000 steps, 8 GPUs x batch 1, gradient checkpointing |

Only the adapter tensors are saved. DoRA only reparameterises fixed weight matrices, so the
adapter can be merged into the base weights for deployment; inference cost is then that of the
base model, which is what `bench_speed.py` times. The VBench clips were generated with the adapter
applied unmerged through peft.

## 3. VBench (Table 18)

Generation (Wan environment). `VBench_full_info.json` is the standard VBench prompt suite from the
[VBench repository](https://github.com/Vchitect/VBench) (`vbench/VBench_full_info.json`; 946 entries,
944 distinct prompts; md5 `893fde62c4472b74d8da0adcb3281307` for the file we used):

```bash
PROMPTS=/path/to/VBench_full_info.json bash scripts/gen_vbench.sh
# -> $OUT_DIR/vbench_videos/{dense,intra}/<prompt>-<i>.mp4, 944 x 5 = 4720 clips per arm
```

480x832, 49 frames, 50 steps, guidance 5.0, 16 fps; seed `20260728 + 5 * prompt_index + sample`,
identical for both arms. File names must stay exactly `<prompt>-<i>.mp4` and the directories must
contain only .mp4 files (VBench matches by name and takes the extension from the first file).

Scoring uses a separate environment with VBench and its dependencies, plus detectron2 for the GRiT
dimensions:

```bash
pip install vbench
pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'
export VBENCH_CACHE_DIR=/path/to/vbench_checkpoints    # VBench downloads its models here
VBENCH_PY=$(which python) PROMPTS=/path/to/VBench_full_info.json bash scripts/score_vbench.sh
```

`score_vbench.sh` first applies three small source fixes that let VBench import under numpy 2 and
transformers 5 (dropped `numpy.lib.function_base` imports; per-name transformers imports in
`third_party/tag2Text/med.py`; both only affect dimensions not in Table 18) and restores
`torch.load(weights_only=False)` for VBench's own checkpoints. It then scores each dimension of each
arm as a separate single-process job (one per GPU) and prints the table with the unweighted
13-dimension mean that the paper reports (dense 0.4745, INTRA 0.4677).

Notes on the reported numbers:

* The mean covers the 13 dimensions of Table 18; `color`, `scene` and `human_action` are not part
  of it.
* `vbench_score.py` works with vbench 0.1.1 and 0.1.5.

## 4. Speed (Table 19)

```bash
bash scripts/bench_speed.sh          # one GPU; JSON + mp4 in $OUT_DIR/speed
```

| mode | measures |
|---|---|
| `parity` | per layer kind: fast path vs reference vs fp32 masked oracle; FA2 vs dense SDPA |
| `plumbing` | our processor with a dense core reproduces stock diffusers on the real weights |
| `attn` | one attention call per layer kind (B=1, 24 heads, d=128, bf16), median of 20 after 5 warm-up; dense baseline = fastest of SDPA flash / cuDNN / efficient / raw FA2. The 30-layer total adds 2 dense + 28 INTRA layers |
| `dit` | full transformer forward, real weights, random inputs, batch 1, per-token timestep; arms run dense, intra, dense, intra (10 iterations, 3 warm-up) and the faster median of each arm is reported |
| `e2e` | one video: text encoding + 50 steps x 2 CFG branches (100 DiT forwards) + VAE decoding; one warm-up run, then the timed run (`runs[1]` in the JSON) |

INTRA in `dit` and `e2e` is the fast path on the base weights (no adapter).

## Paper numbers -> commands

| paper | value(s) | source |
|---|---|---|
| Sec. 6.1 / Table 19, total generation 704x1280, 121 f | 340.9 -> 238.4 s (1.43x) | `bench_speed.py e2e --grids 720p_121f --arm {dense,intra}`; `results/e2e_*_720p_121f.json`, `runs[1].total_s` |
| Sec. 6.1 / Table 19, total generation 480x832, 49 f | 44.1 -> 41.2 s (1.07x) | same with `480p_49f`; `results/e2e_*_480p_49f.json` |
| Table 19, DiT total (100 forwards), VAE decoding | 40.9 -> 38.0, 326.9 -> 224.2; 2.6, 13.4 s | `runs[1].transformer_s`, `runs[1].vae_decode_s` in the same files |
| Table 19, standalone DiT forward | 399 -> 371, 3257 -> 2198, 9143 -> 4855 ms | `bench_speed.py dit --iters 10 --warmup 3`; `results/dit.json` |
| Table 19, self-attention (30 layers) | 49.3 -> 23.2, 1423 -> 342, 5528 -> 1129 ms | `bench_speed.py attn`; `results/attn.json`, `grids.*.stack` |
| Table 19, attention share of dense forward | 12%, 44%, 60% | `attn.json` `stack.dense_ms` / `dit.json` dense forward |
| Table 18, 13 VBench dimensions and mean | dense 0.4745, INTRA 0.4677 | `scripts/selfgen.sh` -> `scripts/train.sh` -> `scripts/gen_vbench.sh` -> `scripts/score_vbench.sh` |
