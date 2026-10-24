#!/usr/bin/env python3
"""Score prediction files and compare arms.

Comparisons are PAIRED by record id: every arm sees the identical prompt set, so the meaningful
question is not "what did each arm average" but "on the same item, how much did the sparse arm lose".
Paired differences remove between-item variance, which on these tasks is far larger than the gap
between methods -- an unpaired table would hide a real regression behind item noise.
"""
import argparse, glob, json, math, os, sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import score_one, truncate_prediction


def load_ref_pool(src):
    """{(task, id): [every official reference for that question]}.

    SCROLLS ships one ROW PER REFERENCE (qasper validation: 1726 rows over 984 questions, 727 with
    2-3 gold answers), while the eval file stores one row's single reference. Pooling by question id
    restores max-over-references and de-duplicates questions sampled under two pids. Only qasper
    is affected; the other five splits have one row per id. The full pool has 5351 ids, 727 of
    them with more than one reference.
    """
    pool = defaultdict(list)
    if not src or not os.path.isdir(src):
        return {}
    for task in ("gov_report", "qasper", "qmsum", "quality", "summ_screen_fd", "contract_nli"):
        f = os.path.join(src, f"{task}.validation.jsonl")
        if not os.path.exists(f):
            continue
        for line in open(f):
            r = json.loads(line)
            g = r.get("output", "")
            if g is not None and str(g).strip():
                pool[(task, r.get("id"))].append(g)
    return dict(pool)


def load(paths, meta_by_key=None, ref_pool=None):
    """-> {task: {key: score}} plus the detail dicts.

    meta_by_key (from --data) makes the eval file authoritative for each record's metric and
    closed label set, overriding whatever the prediction file recorded.
    """
    per = defaultdict(dict)
    detail = defaultdict(lambda: defaultdict(dict))
    n_empty = defaultdict(int)
    for p in paths:
        for line in open(p):
            r = json.loads(line)
            key = r.get("pid") or r.get("id")
            mk = (meta_by_key or {}).get(key) or {}
            metric = mk.get("metric", r["metric"])
            ch = mk.get("choices") or r.get("choices")
            gold = r["answer"]
            if ref_pool:
                # Key on (task, id), not pid: supplies every official reference for the question
                # and collapses questions that appear under two pids.
                pooled = ref_pool.get((r["task"], r.get("id")))
                if pooled:
                    gold = pooled
                    key = f"{r['task']}::{r.get('id')}"
            # Score truncate_prediction(pred) (strip re-emitted scaffold / echoed labels, cut at the
            # first stop string), identically for every arm.
            s, d = score_one(metric, truncate_prediction(r.get("pred", "")), gold, ch)
            if s is None:
                continue
            per[r["task"]][key] = s
            detail[r["task"]][key] = d
            if not r.get("pred", "").strip():
                n_empty[r["task"]] += 1
    return per, detail, n_empty


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True,
                    help="NAME=GLOB, repeatable. First arm is the reference.")
    ap.add_argument("--detail", action="store_true", help="also print ROUGE-1/2 breakdown")
    ap.add_argument("--refs", default=None,
                    help="dir of official <task>.validation.jsonl. Enables SCROLLS' "
                         "max-over-references protocol (pools every gold for a question id). "
                         "Without it, qasper is scored against one reference of its 2-3.")
    ap.add_argument("--data", default=None,
                    help="eval jsonl (scrolls6_n200.jsonl); authoritative metric and `choices` "
                         "per record")
    a = ap.parse_args()

    cbk = {}
    if a.data:
        for line in open(a.data):
            r = json.loads(line)
            if "__meta__" in r:
                continue
            cbk[r.get("pid") or r.get("id")] = {"metric": r["metric"], "choices": r.get("choices")}
        n_ch = sum(1 for v in cbk.values() if v.get("choices"))
        print(f"authoritative metric for {len(cbk)} records "
              f"({n_ch} with a closed choice set) from {os.path.basename(a.data)}\n")

    pool = load_ref_pool(a.refs)
    if pool:
        multi = sum(1 for v in pool.values() if len(v) > 1)
        print(f"reference pool: {len(pool)} question ids, {multi} with >1 official reference "
              f"(max-over-references is ACTIVE)\n")
    elif a.refs:
        print(f"  !! --refs {a.refs} unreadable -- scoring against single references\n")

    arms = []
    for spec in a.arm:
        name, pat = spec.split("=", 1)
        files = sorted(glob.glob(pat))
        if not files:
            print(f"  !! no files for {name}: {pat}", file=sys.stderr)
            continue
        per, det, empt = load(files, cbk, pool)
        arms.append((name, per, det, empt, len(files)))

    if not arms:
        print("nothing to score"); return

    ref_name, ref_per = arms[0][0], arms[0][1]
    tasks = ["gov_report", "qasper", "qmsum", "quality", "summ_screen_fd", "contract_nli"]
    tasks = [t for t in tasks if t in ref_per] + [t for t in ref_per if t not in tasks]

    for name, per, det, empt, nf in arms:
        tot = sum(len(v) for v in per.values())
        print(f"arm {name:<14} {tot:5d} scored records over {nf} file(s)"
              + (f"   empty predictions: {dict(empt)}" if empt else ""))
    print()

    w = max(14, max(len(n) for n, *_ in arms) + 2)
    hdr = f"{'task':<16}{'n':>5}" + "".join(f"{n:>{w}}" for n, *_ in arms)
    print(hdr); print("-" * len(hdr))
    means = defaultdict(list)
    for t in tasks:
        keys = set(ref_per.get(t, {}))
        for _, per, *_ in arms[1:]:
            keys &= set(per.get(t, {}))
        keys = sorted(keys)
        if not keys:
            continue
        line = f"{t:<16}{len(keys):>5}"
        for name, per, *_ in arms:
            m = sum(per[t][k] for k in keys) / len(keys)
            means[name].append(m)
            line += f"{m:>{w}.2f}"
        print(line)
    print("-" * len(hdr))
    line = f"{'MACRO AVG':<16}{'':>5}"
    for name, *_ in arms:
        line += f"{sum(means[name])/len(means[name]):>{w}.2f}" if means[name] else f"{'--':>{w}}"
    print(line)

    if len(arms) < 2:
        return

    print(f"\npaired difference vs {ref_name}  (positive = the arm is better)")
    print(f"{'task':<16}{'arm':<14}{'n':>5}{'delta':>9}{'se':>8}{'t':>8}{'95% CI':>20}{'wins':>10}")
    for t in tasks:
        keys = set(ref_per.get(t, {}))
        for _, per, *_ in arms[1:]:
            keys &= set(per.get(t, {}))
        keys = sorted(keys)
        if len(keys) < 2:
            continue
        for name, per, *_ in arms[1:]:
            d = [per[t][k] - ref_per[t][k] for k in keys]
            n = len(d); m = sum(d) / n
            sd = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1)) if n > 1 else 0.0
            se = sd / math.sqrt(n) if n else 0.0
            tstat = m / se if se else 0.0
            ci = f"[{m-1.96*se:+.2f}, {m+1.96*se:+.2f}]"
            wins = sum(1 for x in d if x > 0); ties = sum(1 for x in d if x == 0)
            print(f"{t:<16}{name:<14}{n:>5}{m:>+9.2f}{se:>8.2f}{tstat:>+8.2f}{ci:>20}"
                  f"{f'{wins}/{n-ties}':>10}")

    if a.detail:
        print("\nROUGE breakdown")
        for t in tasks:
            if not any(t in d for _, _, d, *_ in arms):
                continue
            keys = sorted(set.intersection(*[set(d.get(t, {})) for _, _, d, *_ in arms]))
            if not keys or "rouge1" not in next(iter(arms[0][2][t].values()), {}):
                continue
            for name, _, det, *_ in arms:
                agg = {k: sum(det[t][x][k] for x in keys) / len(keys) for k in ("rouge1", "rouge2", "rougeL")}
                print(f"  {t:<16}{name:<14}" + "  ".join(f"{k}={v:6.2f}" for k, v in agg.items()))


if __name__ == "__main__":
    main()
