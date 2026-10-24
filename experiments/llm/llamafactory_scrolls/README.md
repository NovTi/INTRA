# SCROLLS fine-tuning pipeline (LLaMA-Factory `distill` stage)

The SCROLLS numbers for LLaMA-3.1-8B (paper Table 3b and Table 14) came from this pipeline: a LoRA
student with INTRA attention (Scatter with 8 groups in layers where `layer % 4 in {0, 1}`, a 64-aligned
512-token sliding window in the remaining layers), trained against the frozen dense model. It is an
add-on to LLaMA-Factory, not a standalone package. The NIAH / LoRA-adaptation experiments in `../`
use a different, self-contained trainer (`../train_llama_distill.py`).

## Base version

The files were developed against LLaMA-Factory `main` between the v0.9.2 and v0.9.3 tags, with
transformers 4.49.0 (that release's upper pin). `model/llama3_student.py` and `model/llama3_teacher.py`
are modified copies of transformers' `modeling_llama.py` from that version and import names
(`LossKwargs`, `ALL_ATTENTION_FUNCTIONS` from `modeling_utils`) that later transformers releases
moved. Expect small signature fixes if you start from a different LLaMA-Factory commit.

## Installing into LLaMA-Factory

Copy the files over a LLaMA-Factory checkout:

| file in this directory | destination in LLaMA-Factory | |
|---|---|---|
| `src/llamafactory/model/llama3_student.py` | same path | new: INTRA student |
| `src/llamafactory/model/llama3_teacher.py` | same path | new: dense teacher |
| `src/llamafactory/model/loader.py` | same path | replaces upstream; adds `load_model_distill` |
| `src/llamafactory/data/loader.py` | same path | replaces upstream; adds `get_dataset_distill` |
| `src/llamafactory/train/distill/trainer.py` | same path | new |
| `src/llamafactory/train/distill/workflow.py` | same path | new: `run_distill` |
| `configs/*.yaml`, `configs/train.sh` | anywhere (e.g. `examples/distill/`) | |

Then register the `distill` stage:

1. `src/llamafactory/train/distill/__init__.py` exporting `run_distill`, and
   `src/llamafactory/train/distill/metric.py` copied unchanged from `train/sft/metric.py`.
2. `"distill"` added to the `stage` choices in `hparams/finetuning_args.py`, and a
   `elif finetuning_args.stage == "distill": run_distill(model_args, data_args, training_args, finetuning_args, generating_args, callbacks)`
   branch in `train/tuner.py`.
3. `get_dataset_distill` exported from `data/__init__.py` and `load_model_distill` from
   `model/__init__.py`.

## Data and mask

- **Window mask.** `python make_window_mask.py --out mask4.pt` (16400 x 16400, bf16, about 540 MB).
  The student loads it from `$INTRA_WINDOW_MASK` and slices it to the sequence length, so sequences
  longer than 16400 tokens are not supported.
- **Training data.** `data/loader.py` ignores the path of the yaml `dataset:` entry; the entry
  (`alpaca_en`) only supplies the alpaca column mapping (`instruction` / `input` / `output`). The rows
  are read with `datasets.load_dataset($INTRA_DISTILL_DATA, split="train")`. The main run
  (`configs/distill_scrolls_rank8.yaml`) used the six SCROLLS training sets combined into one
  alpaca-format dataset; `configs/distill_contract_nli_rank32.yaml` is a single-task run on
  Contract NLI.
- `$INTRA_EVAL_DATA` and `$INTRA_LONGBENCH_DATA` are read only if you set `eval_dataset` or enable
  the LongBench variants; the paper runs used neither.

## Training

```bash
cd /path/to/LLaMA-Factory
INTRA_DISTILL_DATA=/path/to/scrolls_alpaca INTRA_WINDOW_MASK=/path/to/mask4.pt \
  bash examples/distill/train.sh examples/distill/distill_scrolls_rank8.yaml
```

- **Hardware.** One A100 per run (batch 1, gradient accumulation 8, 8 epochs), with gradient
  checkpointing on (the LLaMA-Factory default).
- **Loss.** `0.5 * KL + 0.5 * CE` (`torch.nn.KLDivLoss()` against the dense teacher, CE on the gold
  `output` tokens).
- **Mode switches.** Training modes are selected by substrings of `output_dir`, as set in
  `workflow.py`. Any `output_dir` that contains `orgcompare` trains the dense model with plain CE
  and no teacher (the dense fine-tuning baseline). The other switches
  (`only_longbench`, `middle`, `last`, `longbench`, `mlp`, `celoss`, `200kl`, `noscale`) were not used
  for reported results. Make sure your `output_dir` does not contain any of these substrings by
  accident.

## Evaluation

Generate and score the six SCROLLS tasks with `../scripts/scrolls_eval.sh` (`../run_eval.py` +
`../score.py`).
