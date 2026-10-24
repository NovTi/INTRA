#!/usr/bin/env python3
"""SCROLLS metrics, dependency-free.

ROUGE (headline ROUGE-L), token F1 and exact match are implemented against their published
definitions (rouge_score tokenisation for ROUGE; SQuAD normalisation for F1/EM), with max over
references for every task. ROUGE is cross-checked against rouge_score when it is importable.
An empty prediction scores 0; any other scoring error is raised rather than silently scored 0.
"""
import re
import string
from collections import Counter


def normalize_answer(s):
    """SQuAD normalisation: lowercase, strip articles, punctuation and extra whitespace."""
    def remove_articles(t): return re.sub(r"\b(a|an|the)\b", " ", t)
    def white_space_fix(t): return " ".join(t.split())
    def remove_punc(t): return "".join(ch for ch in t if ch not in set(string.punctuation))
    return white_space_fix(remove_articles(remove_punc(s.lower())))


# A prediction that is ONLY an option letter: "D", "(D)", "D.", "D) ". Anchored and short so
# it cannot fire on prose that merely begins with a capital letter.
_BARE_LETTER = re.compile(r"^[\(\[]?([A-Da-d])[\)\].:]?\s*$")
_OPTION_PREFIX = re.compile(r"^\s*[\(\[]?([A-Da-d])[\)\].:]\s+")


def exact_match(pred, golds):
    """Exact match after normalisation, tolerant of a leading option letter.

    SCROLLS' QuALITY gold is the option TEXT, but an instruct model answers "D) Completely
    inhospitable for human life..." -- semantically exactly right, EM 0. Stripping one leading
    enumeration marker is the minimum needed for the metric to measure the model rather than its
    formatting.
    """
    cands = [pred]
    m = _OPTION_PREFIX.match(pred)
    if m:
        cands.append(pred[m.end():])
    norm = {normalize_answer(c) for c in cands}
    return float(any(normalize_answer(g) in norm for g in golds))


def em_label(pred, golds, choices):
    """Accuracy for closed-label tasks: which of `choices` did the model actually pick?

    Needed because instruct models rarely emit a bare label (e.g. 'The hypothesis is "Entailment"
    with respect to the agreement.'). The pick is the choice appearing EARLIEST in the prediction,
    so a model that answers and then discusses the alternatives is scored on its answer.
    Predictions naming no choice score 0.
    """
    p = " " + normalize_answer(pred) + " "
    best, best_at = None, None
    for c in choices:
        at = p.find(" " + normalize_answer(c) + " ")
        if at == -1:
            at = p.find(normalize_answer(c))
        if at != -1 and (best_at is None or at < best_at):
            best, best_at = c, at
    if best is None:
        # Option letter fallback, for predictions that name an option only by its letter ("D",
        # "(B) <paraphrase>"): map the letter back through the option order. Only used when no option
        # TEXT matched, so it can move a record 0 -> 100 but never lower a correct answer (a model can
        # write a letter that disagrees with the option text it then states). Only fires when the
        # letter indexes a real choice and the prediction starts with it.
        m = _OPTION_PREFIX.match(pred.strip()) or _BARE_LETTER.match(pred.strip())
        if m:
            i = ord(m.group(1).upper()) - ord("A")
            if 0 <= i < len(choices):
                best = choices[i]
    if best is None:
        return 0.0
    return float(any(normalize_answer(best) == normalize_answer(g) for g in golds))


def token_f1(pred, golds):
    """SQuAD token-level F1, max over references."""
    def one(p, g):
        pt, gt = normalize_answer(p).split(), normalize_answer(g).split()
        common = Counter(pt) & Counter(gt)
        same = sum(common.values())
        if same == 0:
            return 0.0
        prec, rec = same / len(pt), same / len(gt)
        return 2 * prec * rec / (prec + rec)
    return max((one(pred, g) for g in golds), default=0.0)


def _lcs_len(a, b):
    """Length of the longest common subsequence, O(len(a) * len(b)) time, O(len(b)) space."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            cur[j] = prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def _ngrams(toks, n):
    return Counter(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))


def _f(match, npred, ngold):
    if match == 0 or npred == 0 or ngold == 0:
        return 0.0
    p, r = match / npred, match / ngold
    return 2 * p * r / (p + r)


def rouge_tokens(s):
    """rouge_score's tokenizer: lowercase, non-alphanumeric to space, split. ARTICLES ARE KEPT.

    normalize_answer must NOT be used here: SQuAD normalisation strips a/an/the, which is correct
    for EM and token-F1 but not for ROUGE. Official SCROLLS scores ROUGE through rouge_score, whose
    tokenizer keeps articles.
    """
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).split()


def rouge(pred, golds, cap_tokens=1000):
    """ROUGE-1/2/L F1, max over references. Returns a dict.

    cap_tokens bounds the LCS cost: gov_report references run to several thousand tokens and the
    quadratic LCS over a 512-token generation against a 4000-token reference is the slowest thing in
    the scorer. Truncating both sides to the first cap_tokens matches what the generation budget can
    produce anyway (max_new_tokens is 512).
    """
    pt = rouge_tokens(pred)[:cap_tokens]
    best = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    for g in golds:
        gt = rouge_tokens(g)[:cap_tokens]
        c1 = sum((_ngrams(pt, 1) & _ngrams(gt, 1)).values())
        c2 = sum((_ngrams(pt, 2) & _ngrams(gt, 2)).values())
        r = {
            "rouge1": _f(c1, max(0, len(pt)), max(0, len(gt))),
            "rouge2": _f(c2, max(0, len(pt) - 1), max(0, len(gt) - 1)),
            "rougeL": _f(_lcs_len(pt, gt), len(pt), len(gt)),
        }
        for k in best:
            best[k] = max(best[k], r[k])
    return best


def score_one(metric, pred, golds, choices=None):
    """Returns (headline_score, detail_dict). Headline is what goes in the table."""
    if isinstance(golds, str):
        golds = [golds]
    golds = [g for g in golds if g is not None and str(g).strip() != ""]
    if not golds:
        return None, {}
    if metric == "em_label":
        # closed label set (ContractNLI's 3 labels, QuALITY's 4 options): score which one was picked
        if not choices:
            # 3 of the 2086 official QuALITY records carry no parseable "(A) .." options. Drop the
            # record (score.py skips None), identically for every arm.
            return None, {}
        v = em_label(pred, golds, choices); return v * 100.0, {"acc": v * 100.0}
    if metric == "em":
        v = exact_match(pred, golds); return v * 100.0, {"em": v * 100.0}
    if metric == "f1":
        v = token_f1(pred, golds); return v * 100.0, {"f1": v * 100.0}
    if metric == "rouge":
        d = rouge(pred, golds)
        d = {k: v * 100.0 for k, v in d.items()}
        # SCROLLS reports ROUGE-1/2/L; ROUGE-L is the headline
        return d["rougeL"], d
    raise ValueError(f"unknown metric {metric}")


# Leading scaffold stripped by truncate_prediction. Echoed field labels ("Answer:\nA") are stripped
# as whole units, before the single-character entries; the colon is what marks an echoed label, so
# a summary that merely begins with the word "Answer" is untouched.
_SCAFFOLD = ("assistant", "<|end_header_id|>", "<|start_header_id|>",
             "Answer:", "answer:", "Summary:", "summary:", "Hypothesis:", "hypothesis:",
             ":", "\n", " ", "\t")


def truncate_prediction(text):
    """Cut the generation at the first sign the model has run past its answer.

    Sparse models sometimes re-emit the chat scaffold before answering ("assistant\n\nThe report
    ..."). A stop-string list containing "assistant\n\n" would then truncate the whole answer to
    the empty string, so leading scaffold is stripped first and stop strings only fire at a
    positive index.
    """
    t = text
    # strip re-emitted chat scaffold from the front, repeatedly (it can appear more than once)
    changed = True
    while changed:
        changed = False
        for s in _SCAFFOLD:
            if t.startswith(s):
                t = t[len(s):]
                changed = True
    for stop in ("\n\nQuestion:", "\n\nAnswer:", "\n\nSummary:", "\n\nHypothesis:", "\n\nQuery:",
                 "\nQuestion:", "</s>", "<|eot_id|>", "assistant\n\n"):
        i = t.find(stop)
        if i > 0:            # never at 0 -- that would delete the whole answer
            t = t[:i]
    return t.strip()


if __name__ == "__main__":
    # sanity + cross-check against rouge_score if it is importable
    assert exact_match("Entailment", ["Entailment"]) == 1.0
    assert exact_match("entailment.", ["Entailment"]) == 1.0
    assert exact_match("Contradiction", ["Entailment"]) == 0.0
    # normalize_answer strips articles, so both sides reduce to 3 tokens with 2 in common -> F1 = 2/3
    assert abs(token_f1("the quick brown fox", ["a quick brown dog"]) - 2 / 3) < 1e-9, \
        token_f1("the quick brown fox", ["a quick brown dog"])
    assert token_f1("unanswerable", ["unanswerable"]) == 1.0
    # max over references, not average -- a prediction matching any gold scores full marks
    assert token_f1("yes", ["no", "yes"]) == 1.0
    r = rouge("the cat sat on the mat", ["the cat sat on a mat"])
    print("  self-tests OK |", {k: round(v, 4) for k, v in r.items()})
    try:
        from rouge_score import rouge_scorer
        rs = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
        a, b = "the cat sat on the mat", "the cat sat on a mat"
        ref = rs.score(b, a)
        print("  rouge_score  |", {k: round(ref[k].fmeasure, 4) for k in ref})
        print("  (small gaps are expected: normalize_answer strips articles, rouge_score does not)")
    except ImportError:
        print("  rouge_score not installed -- skipped cross-check")
