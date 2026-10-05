"""Driver: exact bias theory vs Monte Carlo (m=3 and m=4), and the exact plane
trajectory vs simulation. Saves results_theory.txt and figure_theory.(png|pdf)."""
import numpy as np
from math import factorial
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gop_bias_theory as T

HERE = Path(__file__).resolve().parent
WONG = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]

def style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 9,
        "axes.linewidth": 1.0, "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True, "legend.frameon": True,
        "legend.framealpha": 0.9, "legend.fontsize": 7.4, "pdf.fonttype": 42, "ps.fonttype": 42})
def sm(ax):
    ax.minorticks_on(); ax.tick_params(which="both", direction="in", top=True, right=True)

def tie_frac_exact(m, P):
    return 1.0 - factorial(m) * P((1,) * m)   # 1 - Pr(all m distinct)

lines = []
def log(s=""):
    print(s); lines.append(s)

# ---- 1. theory vs MC, m=3 and m=4 ----
for m in (3, 4):
    N = 200
    log(f"Exact theory vs Monte Carlo (m={m}, N={N}, M={T.FUBINI[m]}):")
    log(f"{'Delta':>6} {'D_true':>11} {'plug thy':>11} {'plug MC':>11} {'U thy':>11} {'U MC':>11}")
    for Delta in (0.5, 0.75, 1.0, 1.5):
        th = T.theory(Delta, m, N)
        mp, mu, sp, su = T.mc_biases(Delta, m, N, B=8000, seed=7)
        log(f"{Delta:>6} {th['D_true']:>11.4e} {th['plug_bias']:>+11.3e} {mp-th['D_true']:>+11.3e} "
            f"{th['U_bias']:>+11.3e} {mu-th['D_true']:>+11.3e}")
    log("")

# ---- 2. exact plane trajectory (m=4) on a fine grid ----
m = 4
grid = np.geomspace(0.05, 3.0, 40)
Hs, Ds, ties = [], [], []
for D in grid:
    q = T.q_levels(D); P = T.make_P(q)
    s2, Dtrue, tot, Htrue = T.s2_and_D(m, P)
    Hs.append(Htrue); Ds.append(Dtrue); ties.append(tie_frac_exact(m, P))
Hs, Ds, ties = map(np.array, (Hs, Ds, ties))

# simulated sweep from the de-risking run, if present
sweep = np.load(HERE / "sweep.npy") if (HERE / "sweep.npy").exists() else None

# ---- figure ----
style()
fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.3), layout="constrained")

# (a) parity: theory vs MC bias, both estimators, m=4, several Delta
axa = ax[0]
m = 4; N = 200
tp, mp_, tu, mu_ = [], [], [], []
for Delta in (0.5, 0.75, 1.0, 1.5, 2.0):
    th = T.theory(Delta, m, N)
    a, b, _, _ = T.mc_biases(Delta, m, N, B=8000, seed=3)
    tp.append(th["plug_bias"]); mp_.append(a - th["D_true"])
    tu.append(th["U_bias"]);    mu_.append(b - th["D_true"])
axa.scatter(tp, mp_, s=34, color=WONG[0], edgecolor="k", lw=0.4, label="plug-in", zorder=3)
axa.scatter(tu, mu_, s=34, color=WONG[1], marker="s", edgecolor="k", lw=0.4, label="distinct-pairs $U$", zorder=3)
lim = [min(tu + mu_) * 1.1, max(tp + mp_) * 1.1]
axa.plot(lim, lim, "k--", lw=0.8, zorder=1)
axa.set_xlabel("exact bias (theory)"); axa.set_ylabel("bias (Monte Carlo)")
axa.set_title(r"(a) Exact bias identity vs simulation ($m{=}4$)", loc="left")
axa.legend(loc="upper left"); sm(axa)

# (b) exact plane trajectory vs simulated sweep
axb = ax[1]
sc = axb.scatter(Hs, Ds, c=100 * ties, cmap="viridis", s=16, zorder=2)
axb.plot(Hs, Ds, "-", color="0.7", lw=0.8, zorder=1)
if sweep is not None:
    axb.scatter(sweep[:, 2], sweep[:, 3], facecolor="none", edgecolor="k",
                s=55, lw=1.0, zorder=3, label="simulation")
    axb.legend(loc="upper left")
axb.set_yscale("log")
cb = fig.colorbar(sc, ax=axb, pad=0.02); cb.set_label("tied windows (%)", fontsize=8)
axb.set_xlabel(r"normalized entropy $H$ (base $\log M$)")
axb.set_ylabel(r"disequilibrium $D$ (log)")
axb.set_title(r"(b) Exact plane trajectory vs $\Delta$", loc="left")
sm(axb)

fig.savefig(HERE / "figure_theory.png", dpi=400, bbox_inches="tight", pad_inches=0.05)
fig.savefig(HERE / "figure_theory.pdf", bbox_inches="tight", pad_inches=0.05)
log("saved figure_theory.png/pdf")
(HERE / "results_theory.txt").write_text("\n".join(lines) + "\n")
