#!/usr/bin/env python3
"""Render the six SCROLLS tasks into one uniform eval file (scrolls6_n200.jsonl).

All six tasks (GovReport, SummScreenFD, QMSum, Qasper, QuALITY, ContractNLI) are rendered from the
official SCROLLS validation splits, 200 records per task by default (deterministic subsample,
seed 1234), with no input truncation. Instruction headers for GovReport, SummScreenFD, QMSum and
Qasper follow the original SCROLLS fine-tuning prompts; QuALITY and ContractNLI headers are written
in the same style (see HEADERS). The same render() is used for the training conversations in
pack_scrolls_mix.py.

Official SCROLLS packs query-based tasks as "<query><sep><document>" in a single `input` field, so
the query is split back out and placed after the document.
"""
import argparse, json, os, random, re, sys

HEADERS = {
    # verbatim from the original SCROLLS fine-tuning prompt file (scrolls_combine_val)
    "gov_report": "Instruction: You are given a report by a government agency. Write a one-page summary of the report.",
    "summ_screen_fd": "You are given a script of a TV episode. Summarize the episode in a paragraph.",
    "qmsum": "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.",
    "qasper": 'You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write "unanswerable". If the question is a yes/no question, answer "yes", "no", or "unanswerable".',
    # NOT verbatim: the original prompt asks for the option letter, but the official SCROLLS gold
    # for QuALITY is the option TEXT ('10 hours'), and the metric matches against it. All four
    # options sit inside the query segment in 2083/2086 records.
    "quality": "You are provided a story and a multiple-choice question with 4 possible answers (marked by A, B, C, D). Choose the best answer and write it out exactly as it appears in the options, without the letter.",
    # ContractNLI has no original prompt. Phrased to match SCROLLS' own three labels exactly.
    "contract_nli": 'You are given a non-disclosure agreement and a hypothesis about it. Answer whether the hypothesis is "Entailment", "Contradiction", or "Not mentioned" with respect to the agreement.',
}

# task -> (document label, query label or None, answer label, query/document separator)
#
# The separator is not uniform and getting it wrong is silent. Official SCROLLS packs QuALITY as
#   "<question>\n\n (A) ..\n (B) ..\n (C) ..\n (D) ..\n\n\n<story>"
# so splitting on the first "\n\n" keeps only the question and pushes all four options into the story
# body. Measured over the full validation splits: "\n\n\n" occurs in 2086/2086 quality records and in
# 0/1726 qasper, 0/272 qmsum, 0/1037 contract_nli. Splitting quality on "\n\n\n" yields a query segment
# of median 237 chars containing all four options (2083/2086); splitting the others on "\n\n" yields
# median 44-90 chars, i.e. the bare question.
LAYOUT = {
    "gov_report":     ("Report",         None,         "Summary", None),
    "summ_screen_fd": ("Episode Script", None,         "Summary", None),
    "qmsum":          ("Transcript",     "Query",      "Answer",  "\n\n"),
    "qasper":         ("Article",        "Question",   "Answer",  "\n\n"),
    "quality":        ("Story",          "Question",   "Answer",  "\n\n\n"),
    "contract_nli":   ("Contract",       "Hypothesis", "Answer",  "\n\n"),
}

# SCROLLS metric per task. rouge/f1/em follow the official SCROLLS leaderboard definitions.
# ContractNLI and QuALITY are closed-label tasks, but an instruct model will not emit a bare label:
# dense answers ContractNLI with 'The hypothesis is "Entailment" with respect to the agreement.' and
# QuALITY with 'D) Completely inhospitable...'. Scored by string equality both read 0.00 even when the
# model is right. em_label scores which of the known choices the prediction actually picks.
METRIC = {
    "gov_report": "rouge", "summ_screen_fd": "rouge", "qmsum": "rouge",
    "qasper": "f1", "quality": "em_label", "contract_nli": "em_label",
}

CONTRACT_NLI_LABELS = ["Entailment", "Contradiction", "Not mentioned"]
_OPT_RE = re.compile(r"^\s*\(([A-D])\)\s*(.+?)\s*$")


def extract_choices(task, rec, prompt):
    """Closed choice set for label tasks, or None."""
    if task == "contract_nli":
        return CONTRACT_NLI_LABELS
    if task == "quality":
        # the query segment is "<question>\n\n (A) ..\n (B) ..\n (C) ..\n (D) .."
        head = rec["input"].split("\n\n\n", 1)[0]
        opts = [m.group(2) for m in (_OPT_RE.match(l) for l in head.split("\n")) if m]
        return opts if len(opts) >= 2 else None
    return None

# generation budget. Summaries need room; the two classification-ish tasks need almost none.
MAXNEW = {
    "gov_report": 512, "summ_screen_fd": 512, "qmsum": 256,
    # contract_nli: instruct models restate the hypothesis before the label, which needs up to ~60
    # tokens; 96 leaves margin while still forbidding an essay.
    "qasper": 128, "quality": 32, "contract_nli": 96,
}


def render(task, rec):
    """Return the full prompt string for one record."""
    doc_label, q_label, a_label, sep = LAYOUT[task]
    raw = rec["input"]
    if q_label is None:
        query, doc = None, raw
    else:
        # official SCROLLS puts the query first; see LAYOUT for why the separator is per-task
        parts = raw.split(sep, 1)
        query, doc = (parts[0], parts[1]) if len(parts) == 2 else (None, raw)
    p = HEADERS[task] + f"\n\n{doc_label}:\n" + doc.strip()
    if query is not None:
        p += f"\n\n{q_label}:\n" + query.strip()
    return p + f"\n\n{a_label}:\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="dir of official <task>.validation.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=200, help="records per task (0 = all)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--tokenizer", default="",
                    help="optional tokenizer.json; only used to record n_tokens per record")
    ap.add_argument("--max_input_tokens", type=int, default=0,
                    help="0 = no cap. Any cap is recorded in meta and applied by middle-drop.")
    a = ap.parse_args()

    tok = None
    if a.tokenizer and os.path.exists(a.tokenizer):
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(a.tokenizer)

    rng = random.Random(a.seed)
    out, meta = [], {}
    for task in LAYOUT:
        f = os.path.join(a.src, f"{task}.validation.jsonl")
        if not os.path.exists(f):
            print(f"  !! missing {f}", file=sys.stderr)
            continue
        rows = [json.loads(l) for l in open(f)]
        # Deterministic subsample. Sampling (not head) because SCROLLS files are grouped by document,
        # so the first N records would cover only a handful of distinct documents.
        idx = list(range(len(rows)))
        if a.n and a.n < len(idx):
            rng.shuffle(idx)
            idx = sorted(idx[:a.n])
        lens = []
        for i in idx:
            r = rows[i]
            prompt = render(task, r)
            nt = len(tok.encode(prompt).ids) if tok else None
            if nt: lens.append(nt)
            out.append({
                "task": task, "id": r.get("id"), "pid": r.get("pid"),
                "prompt": prompt, "answer": r.get("output", ""),
                "metric": METRIC[task], "max_new_tokens": MAXNEW[task],
                "choices": extract_choices(task, r, prompt),
                "n_tokens": nt,
            })
        lens.sort()
        meta[task] = {
            "n": len(idx), "pool": len(rows), "metric": METRIC[task],
            "p50_tokens": lens[len(lens)//2] if lens else None,
            "p90_tokens": lens[int(.9*len(lens))] if lens else None,
            "max_tokens": lens[-1] if lens else None,
            "over_16k": sum(1 for x in lens if x > 16384),
            "over_32k": sum(1 for x in lens if x > 32768),
        }

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        fh.write(json.dumps({"__meta__": {
            "source": "official SCROLLS validation splits",
            "note": ("quality and contract_nli are rendered here, not taken from scrolls_combine_val: "
                     "that file omits contract_nli and its quality split has the question and options "
                     "stripped. Headers for the other four are verbatim from scrolls_combine_val."),
            "seed": a.seed, "per_task": meta,
        }}) + "\n")
        for r in out:
            fh.write(json.dumps(r) + "\n")

    print(f"wrote {len(out)} records -> {a.out}")
    print(f"{'task':<16}{'n':>5}{'pool':>7}{'metric':>8}{'p50':>8}{'p90':>8}{'max':>8}{'>16k':>6}{'>32k':>6}")
    for t, m in meta.items():
        print(f"{t:<16}{m['n']:>5}{m['pool']:>7}{m['metric']:>8}"
              f"{str(m['p50_tokens']):>8}{str(m['p90_tokens']):>8}{str(m['max_tokens']):>8}"
              f"{m['over_16k']:>6}{m['over_32k']:>6}")


if __name__ == "__main__":
    main()
