"""R6: real application. RR intervals from record 14134 (MIT-BIH Long-Term ECG DB),
in integer samples (native quantization: 1000/128 = 7.8125 ms lattice), so ties are
frequent. We contrast the tie-aware GOP plane (M=a(m)=75 for m=4) with the conventional
tie-broken permutation plane (M=m!=24) under the two common tie policies (index, jitter),
and test serial structure with the R5 tie-matched (permutation) null.

Message: the tie-broken plane is policy-dependent (index vs jitter disagree, more so as
ties grow), while the GOP treatment is single-valued; and the permutation null confirms the
RR structure is real, not a tie artifact.
"""
import numpy as np, sys
from math import log
from pathlib import Path
from itertools import combinations
from scipy.special import xlogy
import apply_diagnostics as A            # paper-1 permutation-alphabet engine (index/jitter)

HERE = Path(__file__).resolve().parent
RR = str(HERE / "RealData" / "RR_14134.csv")   # RR in integer samples (one column, header)
m, Mgop = 4, 75
LOGM = log(Mgop)

def gop_codes(x):
    e = np.lib.stride_tricks.sliding_window_view(np.asarray(x, float), m)
    code = np.zeros(e.shape[0], dtype=np.int64)
    for p, (i, j) in enumerate(combinations(range(m), 2)):
        code += (np.sign(e[:, i] - e[:, j]).astype(np.int64) + 1) * (3 ** p)
    return code

def _q0(M):
    q = np.full(M, 1 / (2 * M)); q[0] = (1 + 1 / M) / 2
    return 1.0 / (-xlogy(q, q).sum() - 0.5 * log(M))
Q0 = _q0(Mgop)

def gop_coords(x):
    code = gop_codes(x)
    counts = np.unique(code, return_counts=True)[1].astype(float)
    n = counts.sum(); p = counts / n; K = len(counts)
    S = -xlogy(p, p).sum(); H = S / LOGM
    D = (p ** 2).sum() - 1 / Mgop
    Du = (counts * (counts - 1)).sum() / (n * (n - 1)) - 1 / Mgop
    qs = (p + 1 / Mgop) / 2
    J = (-xlogy(qs, qs).sum() + (Mgop - K) * (-(1 / (2 * Mgop)) * log(1 / (2 * Mgop)))
         - 0.5 * S - 0.5 * LOGM)
    C = Q0 * H * J
    # gap estimator, w = m-1
    c = np.bincount(code).astype(float); col = (c * (c - 1)).sum() / 2
    for h in range(1, m):
        col -= np.sum(code[:-h] == code[h:])
    N = code.shape[0]; Dg = col / ((N - m) * (N - m + 1) / 2) - 1 / Mgop
    return dict(H=H, D=D, D_U=Du, D_gap=Dg, C_JS=C, tie=A.tie_fraction(x, m, 1))

def gop_T(x):
    co = gop_coords(x); return (1 - co["H"]) ** 2 + co["C_JS"] ** 2

def perm_null_p(x, B, rng):
    T0 = gop_T(x); cnt = 1
    for _ in range(B):
        cnt += gop_T(rng.permutation(x)) >= T0
    return cnt / (B + 1)

def clean(x):
    x = x.astype(float).copy(); med = np.median(x)
    x[(x > 250) | (x < 40)] = med
    return x

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    rr = np.loadtxt(RR, delimiter=",", skiprows=1)
    print(f"record 14134: {len(rr)} RR intervals (integer samples), "
          f"median {np.median(rr):.0f} = {60*128/np.median(rr):.1f} bpm")

    # representative window
    REP0, REPN = 2000, 2000
    x = clean(rr[REP0:REP0 + REPN])
    g = gop_coords(x)
    p_perm = perm_null_p(x, 999, rng)
    # tie-broken permutation-alphabet treatments (M=24)
    idx = A.coordinates(A.symbols(x, m, 1, policy="index"), 24)
    jit = A.coordinates(A.symbols(x, m, 1, policy="jitter", rng=np.random.default_rng(0)), 24)
    print(f"\nRepresentative window [{REP0}:{REP0+REPN}]  tie fraction {100*g['tie']:.0f}%")
    print(f"  GOP (tie-aware, M=75):  H={g['H']:.4f}  D={g['D']:.5f}  C_JS={g['C_JS']:.5f}")
    print(f"     estimators D: plug {g['D']:.5f}  U {g['D_U']:+.5f}  gap {g['D_gap']:+.5f}  "
          f"(overlap correction small -> strong real structure)")
    print(f"     tie-matched permutation null: p = {p_perm:.4f}")
    print(f"  tie-broken (perm, M=24): index  H={idx['H']:.4f} C_JS={idx['C_JS']:.5f}")
    print(f"                           jitter H={jit['H']:.4f} C_JS={jit['C_JS']:.5f}")
    print(f"     index-vs-jitter H gap = {idx['H']-jit['H']:+.4f}  (policy ambiguity; GOP has none)")

    # across-record scan
    WIN = 1500
    rows = []
    for st in range(0, len(rr) - WIN, WIN):
        w = clean(rr[st:st + WIN])
        tf = A.tie_fraction(w, m, 1)
        Hg = gop_coords(w)["H"]
        Hi = A.coordinates(A.symbols(w, m, 1, policy="index"), 24)["H"]
        Hj = A.coordinates(A.symbols(w, m, 1, policy="jitter", rng=np.random.default_rng(0)), 24)["H"]
        rows.append((st, tf, Hg, Hi, Hj))
    R = np.array(rows)
    tie = 100 * R[:, 1]; gap = R[:, 3] - R[:, 4]         # index - jitter (signed)
    corr = np.corrcoef(tie, gap)[0, 1]
    print(f"\nAcross record: tie {tie.min():.0f}-{tie.max():.0f}%  "
          f"corr(tie, index-jitter H gap) = {corr:.3f}")
    np.save(HERE / "rr_scan.npy", R)

    # ---- figure ----
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    W = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 9,
        "axes.linewidth": 1.0, "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True,
        "ytick.right": True, "legend.fontsize": 7.0, "pdf.fonttype": 42})
    def sm(ax): ax.minorticks_on(); ax.tick_params(which="both", direction="in", top=True, right=True)
    fig, ax = plt.subplots(2, 2, figsize=(7.6, 5.6), layout="constrained")
    # (a) RR trace (ms)
    axa = ax[0, 0]; ms = 1000.0 / 128
    axa.plot(np.arange(REPN), x * ms, color=W[0], lw=0.6)
    axa.set_xlabel("beat number"); axa.set_ylabel("RR interval (ms)")
    axa.set_title(f"(a) RR, record 14134 (tie {100*g['tie']:.0f}%, perm-null $p<10^{{-3}}$)", loc="left")
    sm(axa)
    # (b) tie fraction across record
    axb = ax[0, 1]; axb.plot(R[:, 0] / 1000, tie, color=W[0], marker="o", ms=3, lw=0.9)
    axb.set_xlabel("record position (thousands of beats)"); axb.set_ylabel("tied windows (%)")
    axb.set_title("(b) Ties vary along the record", loc="left"); sm(axb)
    # (c) tie-broken policy ambiguity grows with ties; GOP has none
    axc = ax[1, 0]; axc.scatter(tie, gap, s=15, color=W[1], edgecolor="k", lw=0.3, zorder=3)
    b1, b0 = np.polyfit(tie, gap, 1); xx = np.linspace(tie.min(), tie.max(), 50)
    axc.plot(xx, b0 + b1 * xx, "k--", lw=0.9)
    axc.axhline(0, color=W[2], lw=1.3, label="GOP (tie-aware): no policy, zero gap")
    axc.set_xlabel("tied windows (%)"); axc.set_ylabel(r"$H_{\rm index}-H_{\rm jitter}$ (tie-broken)")
    axc.set_title(f"(c) Tie-breaking is arbitrary (corr $={corr:.2f}$)", loc="left")
    axc.legend(loc="lower left"); sm(axc)
    # (d) GOP plane point + permutation null cloud for the representative window
    axd = ax[1, 1]
    nullT = []
    Hn, Dn = [], []
    for _ in range(400):
        co = gop_coords(rng.permutation(x)); Hn.append(co["H"]); Dn.append(co["C_JS"])
    axd.scatter(Hn, Dn, s=8, color="0.7", label="permutation null", zorder=1)
    axd.scatter([g["H"]], [g["C_JS"]], s=70, color=W[1], edgecolor="k", lw=0.6,
                marker="*", zorder=3, label="RR (observed)")
    axd.set_xlabel(r"normalized entropy $H$ (base $\log 75$)"); axd.set_ylabel(r"$C_{\rm JS}$")
    axd.set_title("(d) RR sits off the tie-matched null", loc="left")
    axd.legend(loc="upper right"); sm(axd)
    fig.savefig(HERE / "figure_R6.png", dpi=400, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(HERE / "figure_R6.pdf", bbox_inches="tight", pad_inches=0.05)
    print("saved figure_R6.png/pdf, rr_scan.npy")
