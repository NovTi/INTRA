"""
pack_retrieval.py -- synthetic retrieval blocks for distillation.

Generic long text is almost entirely predictable from local context, so a distillation loss over it
puts little gradient on long-range routing. These blocks plant facts ("The special magic {city}
number is: {num}.") far from a question about one of them, so the teacher's predictions on the
answer tokens are sharp and depend on information beyond a sliding window.

Blocks are exactly `--seq_len` tokens with no padding.

`build_retrieval_block` and `template_pieces` are used by pack_scrolls_mix.py (the corpus used for
all reported adapters). Run standalone, this script packs a PG19-only retrieval/continuation mix:
  python pack_retrieval.py --tokenizer meta-llama/Llama-3.1-8B-Instruct \
      --haystack "parquet:$DATA_DIR/pg19/*.parquet" --seq_len 16384 --n_blocks 400 \
      --frac_retrieval 0.5 --n_needles 3 --out $DATA_DIR/mix_16k
"""
import os, glob, json, random, argparse
import torch

NEEDLE_TMPL = "The special magic {city} number is: {num}."
QUESTION_TMPL = "What is the special magic {city} number?"
ANSWER_TMPL = "The special magic {city} number is: {num}"
CITIES = ["San Francisco", "Tokyo", "Reykjavik", "Nairobi", "Montevideo", "Helsinki",
          "Kyoto", "Lisbon", "Reykjanes", "Ulaanbaatar", "Valparaiso", "Trondheim",
          "Marrakesh", "Queenstown", "Hobart", "Tromso"]


def _rows_from_parquet(pat):
    """Read local parquet with pyarrow, so a haystack file does not require `datasets`."""
    import pyarrow.parquet as pq
    files = sorted(glob.glob(pat))
    if not files:
        raise FileNotFoundError(f"no parquet matched {pat}")
    for f in files:
        t = pq.read_table(f)
        cols = set(t.column_names)
        for batch in t.to_batches():
            d = batch.to_pydict()
            n = len(next(iter(d.values())))
            for i in range(n):
                yield {k: d[k][i] for k in cols}


def load_haystack(tok, spec, need):
    kind, _, pat = spec.partition(":")
    if kind == "parquet":
        try:
            ds = _rows_from_parquet(pat)
        except ImportError:
            from datasets import load_dataset
            ds = load_dataset("parquet", data_files=sorted(glob.glob(pat)), split="train")
    else:
        from datasets import load_dataset
        ds = load_dataset(pat, split="train")
    buf = []
    for ex in ds:
        t = next((ex[k] for k in ("text", "content", "raw_content", "document")
                  if k in ex and ex[k]), None)
        if not t:
            continue
        buf.extend(tok(t, add_special_tokens=False)["input_ids"])
        if len(buf) >= need:
            break
    assert len(buf) >= need, f"haystack too short: {len(buf)} < {need}"
    return buf


def template_pieces(tok):
    """Exact token ids for the chat wrapper, split around the user content."""
    MARK = "§MARK§"
    s = tok.apply_chat_template([{"role": "user", "content": MARK}],
                                tokenize=False, add_generation_prompt=True)
    pre, post = s.split(MARK)
    return (tok(pre, add_special_tokens=False)["input_ids"],
            tok(post, add_special_tokens=False)["input_ids"])


def build_retrieval_block(tok, hay, seq_len, rng, pre_ids, post_ids, n_needles=2):
    """Exactly seq_len tokens: [chat prefix][haystack with needles][question][gen prompt][answer]."""
    picks = rng.sample(CITIES, n_needles)
    nums = [str(rng.randint(1000000, 9999999)) for _ in picks]
    needles = [tok(NEEDLE_TMPL.format(city=c, num=n), add_special_tokens=False)["input_ids"]
               for c, n in zip(picks, nums)]
    qi = rng.randrange(n_needles)                      # ask about one of them
    q_ids = tok("\n\n" + QUESTION_TMPL.format(city=picks[qi]),
                add_special_tokens=False)["input_ids"]
    a_ids = tok(ANSWER_TMPL.format(city=picks[qi], num=nums[qi]),
                add_special_tokens=False)["input_ids"]

    fixed = len(pre_ids) + len(post_ids) + len(q_ids) + len(a_ids) + sum(len(n) for n in needles)
    budget = seq_len - fixed
    if budget <= 0:
        return None
    start = rng.randrange(0, max(1, len(hay) - budget))
    body = hay[start:start + budget]

    # plant needles at random depths, keeping them well before the question
    cuts = sorted(rng.randrange(0, len(body)) for _ in range(n_needles))
    out, prev = [], 0
    for c, nd in zip(cuts, needles):
        out.extend(body[prev:c]); out.extend(nd); prev = c
    out.extend(body[prev:])

    ids = pre_ids + out + q_ids + post_ids + a_ids
    if len(ids) != seq_len:                             # correct any tokenizer drift
        if len(ids) > seq_len:
            ids = pre_ids + out[: len(out) - (len(ids) - seq_len)] + q_ids + post_ids + a_ids
        else:
            pad_from = hay[:seq_len - len(ids)]
            ids = pre_ids + pad_from + out + q_ids + post_ids + a_ids
    return ids[:seq_len]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--haystack", required=True, help="parquet:<glob> or a HF dataset name")
    ap.add_argument("--seq_len", type=int, default=4096)
    ap.add_argument("--n_blocks", type=int, default=1200)
    ap.add_argument("--frac_retrieval", type=float, default=0.5,
                    help="rest are plain packed text, so general LM behaviour is not lost")
    ap.add_argument("--n_needles", type=int, default=2)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = random.Random(a.seed)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    pre_ids, post_ids = template_pieces(tok)
    hay = load_haystack(tok, a.haystack, a.seq_len * 12)
    print(f"[pack] haystack {len(hay)} tok; target {a.n_blocks} x {a.seq_len}", flush=True)

    n_retr = int(a.n_blocks * a.frac_retrieval)
    made = n_r = 0
    plain_buf = list(hay)
    eos = tok.eos_token_id
    while made < a.n_blocks:
        want_retr = n_r < n_retr
        if want_retr:
            ids = build_retrieval_block(tok, hay, a.seq_len, rng, pre_ids, post_ids, a.n_needles)
            if ids is None:
                want_retr = False
            else:
                n_r += 1
        if not want_retr:
            if len(plain_buf) < a.seq_len:              # recycle with an EOS boundary
                plain_buf = list(hay) + [eos] + plain_buf
            ids = plain_buf[:a.seq_len]
            plain_buf = plain_buf[a.seq_len:]
        torch.save(torch.tensor(ids, dtype=torch.long),
                   os.path.join(a.out, f"block_{made:05d}.pt"))
        made += 1
        if made % 200 == 0:
            print(f"[pack] {made}/{a.n_blocks} ({n_r} retrieval)", flush=True)

    json.dump({"seq_len": a.seq_len, "n_blocks": made, "n_retrieval": n_r,
               "frac_retrieval": a.frac_retrieval, "n_needles": a.n_needles},
              open(os.path.join(a.out, "meta.json"), "w"), indent=2)
    print(f"[pack] DONE {made} blocks ({n_r} retrieval) -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
