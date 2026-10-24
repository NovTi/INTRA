#!/usr/bin/env python3
"""
VBench scoring helpers (run in the VBench environment; see scripts/score_vbench.sh).

  score  one dimension of one arm, single process
  table  per-dimension dense vs INTRA and the unweighted mean over the Table 18 dimensions

Each dimension runs in its own single-process job rather than vbench's distributed path: in
vbench 0.1.5, appearance_style under world_size > 1 averages the last frame's similarity instead of
the per-frame mean, and every rank writes the shared meta json without a barrier.
"""
import argparse, glob, json, os, sys

# The 13 dimensions reported in the paper (Table 18); their unweighted mean is the summary number.
TABLE18 = ["subject_consistency", "background_consistency", "aesthetic_quality", "imaging_quality",
           "temporal_flickering", "motion_smoothness", "dynamic_degree", "overall_consistency",
           "appearance_style", "temporal_style", "object_class", "multiple_objects",
           "spatial_relationship"]


def cmd_score(a):
    import torch
    # vbench 0.1.5 calls torch.load without weights_only; torch >= 2.6 defaults it to True, which
    # fails on vbench's own checkpoints (e.g. motion_smoothness). Restore the old default.
    _load = torch.load
    torch.load = lambda *x, **k: _load(*x, **{"weights_only": False, **k})
    from vbench import VBench
    vb = VBench(torch.device("cuda"), a.info, a.results)
    name = f"{a.arm}_{a.dim}"
    # A file-name mismatch gives every prompt an empty video list and scores of 0.0 without any
    # error, so check the matching first.
    info = json.load(open(vb.build_full_info_json(a.videos, name, [a.dim])))
    matched = sum(1 for x in info if x.get("video_list"))
    print(f"{name}: {matched}/{len(info)} prompts have videos", flush=True)
    if matched < 0.9 * len(info):
        sys.exit(f"only {matched} prompts matched; refusing to score")
    vb.evaluate(videos_path=a.videos, name=name, dimension_list=[a.dim], local=True, read_frame=False)


def cmd_table(a):
    def load(arm):
        out = {}
        for f in glob.glob(os.path.join(a.results, f"{arm}_*_eval_results.json")):
            for k, v in json.load(open(f)).items():
                out[k] = v[0] if isinstance(v, list) else v
        return out
    D, I = load("dense"), load("intra")
    print(f"{'dimension':<26}{'dense':>9}{'INTRA':>9}{'|diff|':>9}")
    ds, is_ = [], []
    for k in TABLE18:
        d, i = D.get(k), I.get(k)
        if d is None or i is None:
            print(f"{k:<26}{'--' if d is None else f'{d:.4f}':>9}{'--' if i is None else f'{i:.4f}':>9}")
            continue
        ds.append(d); is_.append(i)
        print(f"{k:<26}{d:>9.4f}{i:>9.4f}{abs(d - i):>9.4f}")
    if len(ds) == len(TABLE18):
        print(f"{'13-dimension mean':<26}{sum(ds) / len(ds):>9.4f}{sum(is_) / len(is_):>9.4f}")
    else:
        sys.exit(f"{len(TABLE18) - len(ds)} dimension(s) missing; no mean reported")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--videos", required=True, help="directory of <prompt>-<i>.mp4 for one arm")
    s.add_argument("--arm", required=True)
    s.add_argument("--dim", required=True)
    s.add_argument("--info", required=True, help="VBench_full_info.json")
    s.add_argument("--results", required=True)
    t = sub.add_parser("table")
    t.add_argument("--results", required=True)
    a = ap.parse_args()
    (cmd_score if a.cmd == "score" else cmd_table)(a)


if __name__ == "__main__":
    main()
