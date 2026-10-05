"""De-risking experiment for the tie-alphabet (generalized ordinal pattern, GOP) paper.

Question R2: does the Borin finite-sample/overlap machinery port to the GOP alphabet?
Test on Delta-quantized Gaussian white noise (still IID, so overlap is the only source
of dependence). We check:
  (i)  the plug-in disequilibrium D on the 75-symbol m=4 GOP alphabet is biased upward
       relative to the true (population) D, exactly as Borin show for permutations;
  (ii) the distinct-pairs estimator U flips to negative bias under overlap;
  (iii) the gap estimator with w=m-1 returns the true D (removes the overlap bias).
And R3: how the plane point (H, D) moves as the quantization step Delta grows (tie
fraction from ~0 to high).

GOP symbol = weak order of the window, encoded by the sign pattern of all pairwise
differences (sign in {-1,0,+1}). For m=4 the alphabet has size the ordered Bell number
a(4)=75. H is normalized by log a(m).
"""
import numpy as np
from itertools import combinations
from math import log

M_GOP = {2: 3, 3: 13, 4: 75, 5: 541}   # ordered Bell (Fubini) numbers


def gop_codes(x, m, tau=1):
    """Weak-order code per overlapping window via pairwise-difference signs."""
    span = (m - 1) * tau + 1
    e = np.lib.stride_tricks.sliding_window_view(np.asarray(x, float), span)[:, ::tau]
    pairs = list(combinations(range(m), 2))
    code = np.zeros(e.shape[0], dtype=np.int64)
    for p, (i, j) in enumerate(pairs):
        s = np.sign(e[:, i] - e[:, j]).astype(np.int64) + 1   # {0,1,2}
        code += s * (3 ** p)
    return code


def coords(code, M):
    c = np.bincount(code, minlength=code.max() + 1).astype(float)
    c = c[c > 0]; n = c.sum(); p = c / n
    H = (-(p * np.log(p)).sum()) / log(M)
    D = (p ** 2).sum() - 1.0 / M
    D_U = (c * (c - 1)).sum() / (n * (n - 1)) - 1.0 / M
    return H, D, D_U


def gap_D(code, M, w):
    n = code.shape[0]
    c = np.bincount(code).astype(float)
    col = (c * (c - 1)).sum() / 2
    for h in range(1, w + 1):
        col -= np.sum(code[:-h] == code[h:])
    return col / ((n - w - 1) * (n - w) / 2) - 1.0 / M


def tie_fraction(x, m, tau=1):
    span = (m - 1) * tau + 1
    e = np.lib.stride_tricks.sliding_window_view(np.asarray(x, float), span)[:, ::tau]
    return float(np.mean(np.any(np.diff(np.sort(e, 1), 1) == 0, 1)))


def quantize(g, Delta):
    return np.round(g / Delta) * Delta if Delta > 0 else g


if __name__ == "__main__":
    m = 4; M = M_GOP[m]; w = m - 1
    rng = np.random.default_rng(20260927)

    # ---- true (population) values by a long simulation, per Delta ----
    print(f"GOP alphabet m={m}: M={M} symbols (a(4)=75)")
    print("\nR3. Quantization sweep (true values, N=4e6):")
    print(f"{'Delta':>6} {'tie%':>6} {'H':>9} {'D':>11} {'active patt':>12}")
    Nbig = 4_000_000
    gbig = rng.standard_normal(Nbig)
    sweep = []
    for Delta in (0.02, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
        xq = quantize(gbig, Delta)
        code = gop_codes(xq, m)
        H, D, _ = coords(code, M)
        tf = tie_fraction(xq, m)
        active = len(np.unique(code))
        sweep.append((Delta, tf, H, D))
        print(f"{Delta:>6} {100*tf:>6.1f} {H:>9.5f} {D:>11.5e} {active:>12d}")

    # ---- R2. overlap bias at short N, for a representative Delta with real ties ----
    Delta = 0.75
    xq_true = quantize(rng.standard_normal(20_000_000), Delta)
    ct = gop_codes(xq_true, m)
    _, D_true, _ = coords(ct, M)
    print(f"\nR2. Overlap bias at Delta={Delta} (true D={D_true:.6e}), short N=200, B=8000 reps:")
    N, B = 200, 8000
    Dp = np.empty(B); Du = np.empty(B); Dg = np.empty(B)
    for b in range(B):
        xq = quantize(rng.standard_normal(N), Delta)
        code = gop_codes(xq, m)
        _, Dp[b], Du[b] = coords(code, M)
        Dg[b] = gap_D(code, M, w)
    se = lambda a: a.std() / np.sqrt(len(a))
    print(f"  true D           = {D_true:+.6e}")
    print(f"  plug-in  E[D]    = {Dp.mean():+.6e}  bias {Dp.mean()-D_true:+.3e} (se {se(Dp):.1e})")
    print(f"  distinct-pairs U = {Du.mean():+.6e}  bias {Du.mean()-D_true:+.3e} (se {se(Du):.1e})")
    print(f"  gap (w={w})       = {Dg.mean():+.6e}  bias {Dg.mean()-D_true:+.3e} (se {se(Dg):.1e})")

    np.save("/home/claude/gop/sweep.npy", np.array(sweep))
    np.save("/home/claude/gop/bias.npy",
            np.array([D_true, Dp.mean(), Du.mean(), Dg.mean(), se(Dp), se(Du), se(Dg)]))
    print("\nsaved sweep.npy, bias.npy")
