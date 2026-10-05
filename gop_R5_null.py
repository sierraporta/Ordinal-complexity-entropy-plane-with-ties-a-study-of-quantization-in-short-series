"""R5: a tie-matched calibrated null for the GOP (tie) plane, and its size/power.

Score (as in the base paper): T = (1-H)^2 + C_JS^2, computed on the GOP alphabet.

Two nulls:
  (A) naive: continuous IID Gaussian white noise (the tie-free null). With ties in the data
      this is MIS-calibrated: quantization shifts the plane point (R3), so a genuinely IID but
      quantized series sits far from the continuous-null cloud and is rejected far too often.
  (B) tie-matched: a permutation (shuffle) null of the observed series. It keeps the exact
      marginal, hence the exact tie propensity, and destroys only the serial order -- the right
      reference for "is there serial structure beyond the marginal?".

Size study: data = quantized Gaussian white noise (serial null TRUE, ties present). A correct
test rejects at ~alpha. Power study: data = quantized AR(1) (serial structure present).
"""
import numpy as np
from math import log, factorial
from itertools import combinations
from scipy.signal import lfilter
from scipy.special import xlogy

m, M = 4, 75

def gop_codes(x):
    e = np.lib.stride_tricks.sliding_window_view(np.asarray(x, float), m)
    code = np.zeros(e.shape[0], dtype=np.int64)
    for p, (i, j) in enumerate(combinations(range(m), 2)):
        code += (np.sign(e[:, i] - e[:, j]).astype(np.int64) + 1) * (3 ** p)
    return code

def _q0(M):
    q = np.full(M, 1 / (2 * M)); q[0] = (1 + 1 / M) / 2
    return 1.0 / (-xlogy(q, q).sum() - 0.5 * log(M))
Q0 = _q0(M)
LOGM = log(M)

def score_T(x):
    counts = np.unique(gop_codes(x), return_counts=True)[1].astype(float)
    n = counts.sum(); p = counts / n; K = len(counts)
    S = -xlogy(p, p).sum(); H = S / LOGM
    qs = (p + 1 / M) / 2
    J = (-xlogy(qs, qs).sum()
         + (M - K) * (-(1 / (2 * M)) * log(1 / (2 * M)))
         - 0.5 * S - 0.5 * LOGM)
    C = Q0 * H * J
    return (1 - H) ** 2 + C ** 2

def quantize(x, D): return np.round(x / D) * D if D else x
def ar1(N, rng, phi, D):
    y = lfilter([1.0], [1.0, -phi], rng.standard_normal(N + 2000))[2000:]
    return quantize(y / y.std(), D)

def pval_permutation(x, B, rng):
    T0 = score_T(x)
    cnt = 1
    for _ in range(B):
        cnt += score_T(rng.permutation(x)) >= T0
    return cnt / (B + 1)

def pval_vs_cloud(x, cloud):
    """Returns (upper-tail p, two-sided p) against a fixed null cloud."""
    T0 = score_T(x); L = len(cloud)
    pu = (1 + np.sum(cloud >= T0)) / (L + 1)
    pl = (1 + np.sum(cloud <= T0)) / (L + 1)
    return pu, min(1.0, 2 * min(pu, pl))

if __name__ == "__main__":
    rng = np.random.default_rng(7)
    N, alpha, Bperm = 300, 0.05, 299

    # naive null-A cloud: continuous Gaussian white noise (computed once)
    cloudA = np.array([score_T(rng.standard_normal(N)) for _ in range(4000)])

    print(f"N={N}, alpha={alpha}. Rejection rates (should be ~{alpha} when the serial null is TRUE).\n")
    print("SIZE  (data = quantized white noise; NO serial structure, ties present):")
    print(f"{'Delta':>6} {'tie%':>6} | {'naiveA 1-sided':>14} {'naiveA 2-sided':>14} {'tie-matched B':>14}")
    R = 500
    size_rows = []
    for D in (0.0, 0.5, 1.0, 1.5):
        rA1 = rA2 = rB = 0; tie = 0.0
        for _ in range(R):
            x = quantize(rng.standard_normal(N), D)
            e = np.lib.stride_tricks.sliding_window_view(x, m)
            tie += np.mean(np.any(np.diff(np.sort(e, 1), 1) == 0, 1))
            pu, p2 = pval_vs_cloud(x, cloudA)
            rA1 += pu <= alpha; rA2 += p2 <= alpha
            rB += pval_permutation(x, Bperm, rng) <= alpha
        size_rows.append((100 * tie / R, rA1 / R, rA2 / R, rB / R))
        print(f"{D:>6} {100*tie/R:>6.1f} | {rA1/R:>14.3f} {rA2/R:>14.3f} {rB/R:>14.3f}")

    print("\nPOWER (data = quantized AR(1); serial structure present) with the tie-matched null B:")
    print(f"{'phi':>6} {'Delta':>6} {'tie%':>6} | {'power B':>9}")
    R = 400
    power_rows = []
    for phi, D in ((0.3, 1.0), (0.5, 1.0), (0.9, 1.0), (0.5, 1.5)):
        rej = 0; tie = 0.0
        for _ in range(R):
            x = ar1(N, rng, phi, D)
            e = np.lib.stride_tricks.sliding_window_view(x, m)
            tie += np.mean(np.any(np.diff(np.sort(e, 1), 1) == 0, 1))
            rej += pval_permutation(x, Bperm, rng) <= alpha
        power_rows.append((phi, D, 100 * tie / R, rej / R))
        print(f"{phi:>6} {D:>6} {100*tie/R:>6.1f} | {rej/R:>9.3f}")

    # ---- figure ----
    from pathlib import Path
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HERE = Path(__file__).resolve().parent
    W = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 9,
        "axes.linewidth": 1.0, "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True,
        "ytick.right": True, "legend.frameon": True, "legend.fontsize": 7.2, "pdf.fonttype": 42})
    fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.3), layout="constrained")
    tievals = [r[0] for r in size_rows]
    xx = np.arange(len(tievals)); wid = 0.26
    a0 = ax[0]
    a0.bar(xx - wid, [r[1] for r in size_rows], wid, color=W[1], edgecolor="k", lw=0.5, label="naive null (1-sided)")
    a0.bar(xx, [r[2] for r in size_rows], wid, color=W[4], edgecolor="k", lw=0.5, label="naive null (2-sided)")
    a0.bar(xx + wid, [r[3] for r in size_rows], wid, color=W[2], edgecolor="k", lw=0.5, label="tie-matched null")
    a0.axhline(alpha, color="k", ls="--", lw=1.0, label=r"nominal $\alpha=0.05$")
    a0.set_xticks(xx); a0.set_xticklabels([f"{t:.0f}%" for t in tievals])
    a0.set_xlabel("tied ordinal windows (%)"); a0.set_ylabel("rejection rate under the serial null")
    a0.set_title("(a) Size: only the tie-matched null stays at $\\alpha$", loc="left")
    a0.legend(loc="upper center"); a0.minorticks_on(); a0.tick_params(which="both", direction="in", top=True, right=True)
    a1 = ax[1]
    labs = [f"$\\phi{{=}}{p}$\n$\\Delta{{=}}{d}$" for p, d, *_ in power_rows]
    a1.bar(range(len(power_rows)), [r[3] for r in power_rows], 0.6, color=W[0], edgecolor="k", lw=0.5)
    a1.axhline(alpha, color="k", ls="--", lw=1.0)
    a1.set_xticks(range(len(power_rows))); a1.set_xticklabels(labs, fontsize=7.5)
    a1.set_ylim(0, 1.05); a1.set_ylabel("power (tie-matched null)")
    a1.set_title("(b) Power vs quantized AR(1)", loc="left")
    a1.minorticks_on(); a1.tick_params(which="both", direction="in", top=True, right=True)
    fig.savefig(HERE / "figure_R5.png", dpi=400, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(HERE / "figure_R5.pdf", bbox_inches="tight", pad_inches=0.05)
    print("\nsaved figure_R5.png/pdf")
