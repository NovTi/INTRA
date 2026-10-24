#!/usr/bin/env python3
"""NIAH exact-match by length for a set of arms, side by side.

Usage:  niah_table.py --dir $OUT_DIR/niah <arm> [<arm> ...]
        (arm = the prefix of <dir>/<arm>_L<len>.json written by scripts/niah.sh)

The `tag` recorded inside each json names the mask the run actually used and is printed alongside.
"""
import argparse
import glob
import json
import os

LENGTHS = [4096, 8192, 16384, 32768]


def load(d, arm):
    out = {}
    for f in glob.glob(os.path.join(d, f"{arm}_L*.json")):
        try:
            L = int(os.path.basename(f).split("_L")[-1][:-5])
            out[L] = json.load(open(f))
        except Exception:
            continue
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="directory holding <arm>_L<len>.json")
    ap.add_argument("arms", nargs="+")
    a = ap.parse_args()
    data = {arm: load(a.dir, arm) for arm in a.arms}
    hdr = f"{'arm':<16}" + "".join(f"{('L=' + str(L)):>10}" for L in LENGTHS) + f"{'MEAN':>10}   tag"
    print(hdr)
    print("-" * (len(hdr) + 20))
    for arm in a.arms:
        d = data[arm]
        if not d:
            print(f"{arm:<16}{'(no results)':>10}")
            continue
        cells, vals = "", []
        for L in LENGTHS:
            if L in d:
                v = d[L]["mean_exact"]
                vals.append(v)
                cells += f"{v:>10.3f}"
            else:
                cells += f"{'--':>10}"
        m = sum(vals) / len(vals) if vals else float("nan")
        tag = d[max(d)].get("tag", "?")
        print(f"{arm:<16}{cells}{m:>10.3f}   {tag}")
    print("\n  exact-match, 9 depths x 10 trials = 90 items per length. dense = 1.000.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
