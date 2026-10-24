#!/usr/bin/env python3
"""Retained attention density of an INTRA layer stack, as installed by llama_intra.py.

Density = attended (query, key) pairs / dense causal pairs, averaged over layers (sink tokens are
not counted), at S = 16384. Compare the "retained density" column of paper Tables 5 and 16.

  python density.py                                    # the paper configurations
  python density.py --pattern SWWWWWWW --groups 2 --n_offsets 2 --window 575
"""
import argparse

import numpy as np


def scatter_density(S, b, off, G):
    """Exact visible-pair / causal-pair count for ONE block-level Scatter layer (block b, offset off).

    With off != 0 the query's own block is visible in addition to the shifted residue class.
    """
    b = max(1, int(b))
    nb = -(-S // b)
    pairs = 0
    for u in range(nb):
        su = min(b, S - u * b)                       # size of the query block
        r = (u + off) % G
        if off == 0:
            n_earlier = u // G                       # v < u, v == u (mod G)
        else:
            n_earlier = ((u - 1 - r) // G + 1) if r <= u - 1 else 0
        pairs += n_earlier * b * su + su * (su + 1) // 2   # earlier blocks + own block (causal)
    return pairs / (S * (S + 1) / 2)


def window_density(S, window):
    """Causal sliding window of `window` keys."""
    i = np.arange(S)
    return float(np.minimum(i + 1, window).sum()) / (S * (S + 1) / 2)


def stack_density(S, pattern, groups, block=256, n_offsets=1, window=575, n_layers=32):
    """Mean density over the stack; mirrors install_intra_llama's layer/offset assignment."""
    pat = [c for c in pattern.upper() if c in "SWD"]
    n_off = n_offsets if n_offsets > 0 else groups
    tot, n_s = 0.0, 0
    for li in range(n_layers):
        kind = pat[li % len(pat)]
        if kind == "S":
            off = (n_s % n_off) if n_off > 1 else 0
            tot += scatter_density(S, block, off, groups)
            n_s += 1
        elif kind == "W":
            tot += window_density(S, window)
        else:
            tot += 1.0
    return tot / n_layers


# arm: (pattern, groups, n_offsets, window, paper value)
PAPER = {
    "wonly575":  ("WWWWWWWW", 2, 1, 575, 0.069),    # Table 5, sliding window
    "g2s4":      ("SWWWWWWW", 2, 2, 575, 0.123),    # Table 5/16, INTRA
    "dense4":    ("DWWWWWWW", 2, 1, 575, 0.185),    # Table 5, dense in the same 4 layers
    "baseb":     ("SSWW",     8, 1, 575, 0.098),    # Table 16, sparse attention in 16 layers
    "wonly2048": ("WWWWWWWW", 2, 1, 2048, 0.234),   # Table 16, wider windows
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--S", type=int, default=16384)
    ap.add_argument("--pattern")
    ap.add_argument("--groups", type=int, default=2)
    ap.add_argument("--n_offsets", type=int, default=1, help="stripe offsets (1 = no stripe)")
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--window", type=int, default=575)
    ap.add_argument("--n_layers", type=int, default=32)
    a = ap.parse_args()
    if a.pattern:
        print(f"{stack_density(a.S, a.pattern, a.groups, a.block, a.n_offsets, a.window, a.n_layers):.5f}")
    else:
        for arm, (pat, g, no, w, ref) in PAPER.items():
            d = stack_density(a.S, pat, g, 256, no, w)
            print(f"{arm:<10} {pat:<9} G={g} offsets={no} w={w:<5} density={d:.5f}  (paper {ref:.3f})")
            # Exact pair counts differ from the paper's rounded values by <= 0.001 (g2s4: the paper
            # uses the nominal per-layer density 0.5, which ignores the stripe's own-block edge).
            assert abs(d - ref) < 0.0015, (arm, d, ref)
