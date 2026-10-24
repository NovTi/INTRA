#!/usr/bin/env python3
"""pack_scrolls_mix.py -- the distillation corpus for all reported LoRA adapters ("mixgen").

Blocks of exactly --seq_len tokens, in three kinds (defaults: 400 blocks = 240 / 100 / 60):

  task       (60%) real SCROLLS TRAIN conversations, each rendered through build_data.render() --
             the same function that renders the eval set -- then chat-wrapped and followed by the
             gold answer. Several short conversations are packed end-to-end with an EOS boundary;
             a long document is middle-truncated to fill a block by itself.
  retrieval  (25%) needle blocks (pack_retrieval.build_retrieval_block, 3 needles) over a haystack
             of mixed SCROLLS training documents.
  plain      (rest) plain continuation over the same mixed corpus.

Supervision comes from the dense teacher (train_llama_distill.py), not from the gold answers
(except through the small CE term).

Leakage guard: evaluation uses the SCROLLS validation split, training the train split. Every
training document is additionally hashed and dropped if it collides with a document in the eval
file (--eval_file), with the count reported in meta.json.
"""
import argparse, hashlib, json, os, random, sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_data import HEADERS, LAYOUT, render                      # noqa: E402
from pack_retrieval import build_retrieval_block, template_pieces   # noqa: E402

TASKS = list(LAYOUT)


def doc_of(task, rec):
    """The document half of a SCROLLS record, which is what identifies it across splits.

    Official SCROLLS packs query-based tasks as "<query><sep><document>" in one `input` field, with a
    per-task separator (LAYOUT). Summarisation tasks have no query and the whole input is the doc.
    """
    _, q_label, _, sep = LAYOUT[task]
    if q_label is None:
        return rec["input"]
    parts = rec["input"].split(sep, 1)
    return parts[1] if len(parts) == 2 else rec["input"]


def dochash(s):
    # normalise whitespace before hashing: the same document is not always whitespace-identical
    # across splits, and a hash that misses a real overlap is worse than no check at all.
    return hashlib.sha1(" ".join(s.split()).encode("utf-8", "replace")).hexdigest()


def middle_drop(ids, budget):
    """Truncate to `budget` tokens by removing the MIDDLE, keeping head and tail.

    Head and tail are where the answerable content concentrates in these tasks (a gov_report's
    opening framing and closing findings; a screenplay's setup and resolution), and truncating from
    the tail alone would systematically delete the part a summary has to cover.
    """
    if len(ids) <= budget:
        return ids
    head = budget // 2
    return ids[:head] + ids[len(ids) - (budget - head):]


def conv_ids(tok, task, rec, pre_ids, post_ids, max_doc_tokens):
    """Token ids for one complete conversation: [chat prefix][eval prompt][gen prompt][gold answer].

    The prompt half goes through render() unmodified, so it is exactly what run_eval.py will feed
    the model at eval time -- header, document label, query placement and all. Only the document is
    length-limited, and that happens on the rendered text so the header and question always survive.
    """
    prompt = render(task, rec)
    ans = str(rec.get("output", "")).strip()
    if not ans:
        return None
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    a_ids = tok(ans, add_special_tokens=False)["input_ids"]
    if len(p_ids) > max_doc_tokens:
        # Keep the tail intact: it holds "\n\nQuestion:\n<q>\n\nAnswer:\n", without which the block
        # teaches the model to attend over a document toward nothing in particular.
        tail_n = min(len(p_ids), 512)
        p_ids = middle_drop(p_ids[:-tail_n], max_doc_tokens - tail_n) + p_ids[-tail_n:]
    return pre_ids + p_ids + post_ids + a_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--scrolls_train", required=True, help="dir of <task>.train.jsonl")
    ap.add_argument("--eval_file", default=None,
                    help="the eval jsonl; its documents are excluded from training")
    ap.add_argument("--seq_len", type=int, default=16384)
    ap.add_argument("--n_blocks", type=int, default=400)
    ap.add_argument("--frac_task", type=float, default=0.60)
    ap.add_argument("--frac_retrieval", type=float, default=0.25)
    ap.add_argument("--n_needles", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    assert a.frac_task + a.frac_retrieval <= 1.0, "fractions exceed 1"
    os.makedirs(a.out, exist_ok=True)
    rng = random.Random(a.seed)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    pre_ids, post_ids = template_pieces(tok)
    eos = tok.eos_token_id

    # ---------------------------------------------------------------- leakage guard
    banned = set()
    if a.eval_file:
        n_eval = 0
        for line in open(a.eval_file):
            r = json.loads(line)
            if "__meta__" in r:
                continue
            # the eval file stores the RENDERED prompt, so recover the document by slicing between
            # the document label and the query label -- the same layout render() produced it with
            doc_label, q_label, a_label, _ = LAYOUT[r["task"]]
            p = r["prompt"]
            i = p.find(f"\n{doc_label}:\n")
            j = p.find(f"\n\n{q_label}:\n") if q_label else p.find(f"\n\n{a_label}:\n")
            if i != -1 and j > i:
                banned.add(dochash(p[i + len(doc_label) + 3: j]))
            n_eval += 1
        print(f"[pack] eval file: {n_eval} records -> {len(banned)} distinct documents banned")

    # ---------------------------------------------------------------- load train records
    by_task, dropped = {}, 0
    for t in TASKS:
        f = os.path.join(a.scrolls_train, f"{t}.train.jsonl")
        if not os.path.exists(f):
            print(f"  !! missing {f}, skipping {t}")
            continue
        keep = []
        for line in open(f):
            rec = json.loads(line)
            if dochash(doc_of(t, rec)) in banned:
                dropped += 1
                continue
            keep.append(rec)
        rng.shuffle(keep)
        by_task[t] = keep
        print(f"  {t:<16} {len(keep):>6} train records")
    assert by_task, "no SCROLLS train data found"
    print(f"[pack] leakage guard dropped {dropped} train records overlapping the eval documents")

    # ---------------------------------------------------------------- haystack for needles/plain
    # Mixed SCROLLS documents, NOT PG19. Round-robin across tasks so one 17k-record task cannot own
    # the haystack -- gov_report alone is 17457 records against qmsum's 1257.
    hay_txt, per_task = [], 60
    for t, rows in by_task.items():
        for rec in rows[:per_task]:
            hay_txt.append(doc_of(t, rec))
    rng.shuffle(hay_txt)
    hay = []
    need = a.seq_len * 12
    for txt in hay_txt:
        hay.extend(tok(txt, add_special_tokens=False)["input_ids"])
        if len(hay) >= need:
            break
    print(f"[pack] mixed-domain haystack: {len(hay)} tokens from {len(hay_txt)} documents")
    assert len(hay) >= a.seq_len * 2, f"haystack too short ({len(hay)})"

    # ---------------------------------------------------------------- build
    n_task = int(a.n_blocks * a.frac_task)
    n_retr = int(a.n_blocks * a.frac_retrieval)
    cursor = {t: 0 for t in by_task}
    order = list(by_task)
    made = k_task = k_retr = k_plain = 0
    n_conv = 0
    plain_buf = list(hay)

    def next_rec():
        """Round-robin across tasks, so each of the six is equally represented per block."""
        for _ in range(len(order) * 4):
            t = order[(next_rec.i) % len(order)]
            next_rec.i += 1
            rows = by_task[t]
            if cursor[t] < len(rows):
                rec = rows[cursor[t]]
                cursor[t] += 1
                return t, rec
            cursor[t] = 0                      # wrap; 400 blocks cannot exhaust 25k records anyway
        return None, None
    next_rec.i = 0

    while made < a.n_blocks:
        ids = None
        if k_task < n_task:
            # pack whole conversations end-to-end until the block is full
            buf = []
            while len(buf) < a.seq_len:
                t, rec = next_rec()
                if rec is None:
                    break
                c = conv_ids(tok, t, rec, pre_ids, post_ids, a.seq_len - len(pre_ids) - len(post_ids) - 64)
                if c is None:
                    continue
                buf.extend(c)
                buf.append(eos)
                n_conv += 1
            if len(buf) >= a.seq_len:
                ids = buf[:a.seq_len]
                k_task += 1
        if ids is None and k_retr < n_retr:
            ids = build_retrieval_block(tok, hay, a.seq_len, rng, pre_ids, post_ids, a.n_needles)
            if ids is not None:
                k_retr += 1
        if ids is None:
            if len(plain_buf) < a.seq_len:
                plain_buf = list(hay) + [eos] + plain_buf
            ids = plain_buf[:a.seq_len]
            plain_buf = plain_buf[a.seq_len:]
            k_plain += 1
        assert len(ids) == a.seq_len, f"block {made} is {len(ids)} tokens, expected {a.seq_len}"
        torch.save(torch.tensor(ids, dtype=torch.long), os.path.join(a.out, f"block_{made:05d}.pt"))
        made += 1
        if made % 50 == 0:
            print(f"[pack] {made}/{a.n_blocks}  task={k_task} retr={k_retr} plain={k_plain}", flush=True)

    json.dump({"seq_len": a.seq_len, "n_blocks": made, "n_task": k_task, "n_retrieval": k_retr,
               "n_plain": k_plain, "n_conversations": n_conv, "n_needles": a.n_needles,
               "tasks": list(by_task), "leakage_dropped": dropped, "banned_docs": len(banned),
               "source": "SCROLLS train splits + needle blocks over mixed SCROLLS haystack"},
              open(os.path.join(a.out, "meta.json"), "w"), indent=2)
    print(f"[pack] DONE {made} blocks (task={k_task}/{n_conv} convs, retr={k_retr}, "
          f"plain={k_plain}) -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
