"""R4: ground-truth plane coordinates on the GOP (tie) alphabet for processes beyond
white noise, and the estimator behaviour on a dependent process.

Processes (m=4, M=75):
  - quantized Gaussian white noise      (exact theory available -> validates the simulator)
  - quantized AR(1), phi = 0.5, 0.9     (dependence beyond the window)
  - Poisson counts, lambda = 5, 10      (native integers, ties without a quantization step)
  - quantized logistic map              (deterministic, strongly non-uniform)

True coordinates come from one long simulation (N=6e6). For AR(1) we also run the short-N
estimators (plug-in / distinct-pairs / gap w=m-1): the gap removes the within-window overlap
bias but leaves a residual for strong persistence (phi=0.9), because AR(1) dependence extends
beyond the excluded lags. This is the tie-alphabet analogue of the heliospheric case and
motivates a tie-matched calibrated null (R5).
"""
import numpy as np
from math import factorial
from itertools import combinations
from scipy.signal import lfilter
import gop_bias_theory as T

m, M, w = 4, T.FUBINI[4], 3

def gop_codes(x):
    e = np.lib.stride_tricks.sliding_window_view(np.asarray(x, float), m)
    code = np.zeros(e.shape[0], dtype=np.int64)
    for p, (i, j) in enumerate(combinations(range(m), 2)):
        code += (np.sign(e[:, i] - e[:, j]).astype(np.int64) + 1) * (3 ** p)
    return code

def HD(code):
    c = np.bincount(code); c = c[c > 0].astype(float); n = c.sum(); p = c / n
    H = (-(p * np.log(p)).sum()) / np.log(M)
    D = (p ** 2).sum() - 1 / M
    return H, D

def tie_frac(x):
    e = np.lib.stride_tricks.sliding_window_view(np.asarray(x, float), m)
    return float(np.mean(np.any(np.diff(np.sort(e, 1), 1) == 0, 1)))

def D_estimators(code):
    c = np.bincount(code).astype(float); cc = c[c > 0]; n = cc.sum()
    Dp = (cc / n ** 2 * cc).sum() - 1 / M
    Du = (cc * (cc - 1)).sum() / (n * (n - 1)) - 1 / M
    col = (c * (c - 1)).sum() / 2
    for h in range(1, w + 1):
        col -= np.sum(code[:-h] == code[h:])
    N = code.shape[0]
    Dg = col / ((N - w - 1) * (N - w) / 2) - 1 / M
    return Dp, Du, Dg

# ---- generators (unit-variance where continuous, then quantize) ----
def q(x, Delta):
    return np.round(x / Delta) * Delta if Delta else x

def gen_white(N, rng, Delta):    return q(rng.standard_normal(N), Delta)
def gen_ar1(N, rng, Delta, phi):
    x = lfilter([1.0], [1.0, -phi], rng.standard_normal(N + 2000))[2000:]
    return q(x / x.std(), Delta)
def gen_poisson(N, rng, lam):    return rng.poisson(lam, N).astype(float)
def gen_logistic(N, Delta, x0=0.4):
    x = np.empty(N); x[0] = x0
    for t in range(1, N):
        x[t] = 4 * x[t - 1] * (1 - x[t - 1])
    return q(x, Delta)

if __name__ == "__main__":
    rng = np.random.default_rng(11)
    Nbig = 6_000_000
    rows = []
    def add(name, x):
        H, D = HD(gop_codes(x)); rows.append((name, tie_frac(x), H, D))

    add("white  Delta=0.5", gen_white(Nbig, rng, 0.5))
    add("white  Delta=1.0", gen_white(Nbig, rng, 1.0))
    add("AR(1) phi=0.5 D=0.5", gen_ar1(Nbig, rng, 0.5, 0.5))
    add("AR(1) phi=0.9 D=0.5", gen_ar1(Nbig, rng, 0.5, 0.9))
    add("AR(1) phi=0.9 D=1.0", gen_ar1(Nbig, rng, 1.0, 0.9))
    add("Poisson lambda=5", gen_poisson(Nbig, rng, 5))
    add("Poisson lambda=10", gen_poisson(Nbig, rng, 10))
    add("logistic Delta=0.02", gen_logistic(2_000_000, 0.02))
    add("logistic Delta=0.05", gen_logistic(2_000_000, 0.05))

    print("Ground-truth GOP coordinates (m=4, M=75):")
    print(f"{'process':>22} {'tie%':>7} {'H':>9} {'D':>11}")
    for name, tf, H, D in rows:
        print(f"{name:>22} {100*tf:>7.1f} {H:>9.5f} {D:>11.4e}")

    # cross-check white noise against exact theory
    th = T.theory(0.5, m, 200)
    print(f"\nvalidator: white Delta=0.5 sim D={rows[0][3]:.4e} vs exact D_true={th['D_true']:.4e}")

    # ---- AR(1) estimator demonstration on short series ----
    print("\nAR(1) estimators on short N=300 (B=4000 reps): gap leaves a residual under persistence")
    print(f"{'phi':>5} {'Delta':>6} {'trueD':>10} {'plug':>10} {'U':>10} {'gap':>10} {'gap-true':>10}")
    est = []
    for phi, Delta in ((0.5, 0.5), (0.9, 0.5), (0.9, 1.0)):
        xt = gen_ar1(Nbig, rng, Delta, phi); _, Dtrue = HD(gop_codes(xt))
        B = 4000
        acc = np.zeros((B, 3))
        for b in range(B):
            x = gen_ar1(300, rng, Delta, phi)
            acc[b] = D_estimators(gop_codes(x))
        mp, mu, mg = acc.mean(0)
        est.append((phi, Delta, Dtrue, mp, mu, mg))
        print(f"{phi:>5} {Delta:>6} {Dtrue:>10.4e} {mp:>10.4e} {mu:>10.4e} {mg:>10.4e} {mg-Dtrue:>+10.2e}")

    # ---- figure ----
    from pathlib import Path
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HERE = Path(__file__).resolve().parent
    WONG = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 9,
        "axes.linewidth": 1.0, "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True,
        "ytick.right": True, "legend.frameon": True, "legend.fontsize": 7.0, "pdf.fonttype": 42})
    fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.3), layout="constrained")
    # (a) processes on the GOP plane, colored by family
    axa = ax[0]
    fam = {"white": WONG[0], "AR(1)": WONG[1], "Poisson": WONG[2], "logistic": WONG[3]}
    seen = set()
    for name, tf, H, D in rows:
        key = name.split()[0]
        lab = {"white": "white noise", "AR(1)": "AR(1)", "Poisson": "Poisson", "logistic": "logistic"}[key]
        axa.scatter(H, D, s=48, color=fam[key], edgecolor="k", lw=0.4, zorder=3,
                    label=lab if key not in seen else None)
        seen.add(key)
    axa.set_yscale("log"); axa.set_xlabel(r"normalized entropy $H$ (base $\log M$)")
    axa.set_ylabel(r"disequilibrium $D$ (log)")
    axa.set_title("(a) Tie-alphabet ground truth by process", loc="left")
    axa.legend(loc="lower left"); axa.minorticks_on()
    axa.tick_params(which="both", direction="in", top=True, right=True)
    # (b) AR(1) estimators vs true: gap recovers for weak dep, residual for strong
    axb = ax[1]
    labels = [f"$\\phi{{=}}{p}$\n$\\Delta{{=}}{d}$" for p, d, *_ in est]
    x = np.arange(len(est)); wid = 0.22
    axb.bar(x - 1.5*wid, [e[3] for e in est], wid, color=WONG[0], edgecolor="k", lw=0.5, label="plug-in")
    axb.bar(x - 0.5*wid, [e[4] for e in est], wid, color=WONG[1], edgecolor="k", lw=0.5, label="$U$")
    axb.bar(x + 0.5*wid, [e[5] for e in est], wid, color=WONG[2], edgecolor="k", lw=0.5, label="gap")
    axb.plot(x + 1.2*wid, [e[2] for e in est], "k_", ms=16, mew=2, label="true $D$")
    axb.set_yscale("log"); axb.set_xticks(x); axb.set_xticklabels(labels, fontsize=7.5)
    axb.set_ylabel(r"disequilibrium $D$ (log)")
    axb.set_title("(b) AR(1), $N{=}300$: gap residual grows with persistence", loc="left")
    axb.legend(loc="upper left", ncol=2); axb.minorticks_on()
    axb.tick_params(which="both", direction="in", top=True, right=True)
    fig.savefig(HERE / "figure_R4.png", dpi=400, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(HERE / "figure_R4.pdf", bbox_inches="tight", pad_inches=0.05)
    print("\nsaved figure_R4.png/pdf")
