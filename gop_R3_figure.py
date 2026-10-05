"""Figure for R3: leading-order quantization law for the plane coordinates."""
import numpy as np
from math import factorial, log, comb
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gop_R3_sheppard as R3
import gop_bias_theory as T

HERE = Path(__file__).resolve().parent
WONG = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
def style():
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 9,
        "axes.linewidth": 1.0, "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True,
        "ytick.right": True, "legend.frameon": True, "legend.framealpha": 0.9, "legend.fontsize": 7.2,
        "pdf.fonttype": 42, "ps.fonttype": 42})
def sm(ax): ax.minorticks_on(); ax.tick_params(which="both", direction="in", top=True, right=True)

m = 4; M = T.FUBINI[m]
H0 = log(factorial(m)) / log(M); D0 = 1 / factorial(m) - 1 / M
grid = np.geomspace(0.01, 0.5, 30)

# Gaussian: exact vs leading-order law
cdf, I = R3.DENS["Gaussian"]
c = np.array([R3.coords(D, cdf, m)[0] for D in grid])
Dx = np.array([R3.coords(D, cdf, m)[2] for D in grid])
Hx = np.array([R3.coords(D, cdf, m)[3] for D in grid])
D_lead = D0 - c / factorial(m - 2)
H_lead = H0 + comb(m, 2) / log(M) * c * np.log(1 / c)

style()
fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.3), layout="constrained")

# (a) D and H displacement vs Delta: exact vs leading law
axa = ax[0]
axa.plot(grid, Dx, "o", color=WONG[0], ms=4, label="$D$ exact")
axa.plot(grid, D_lead, "-", color=WONG[0], lw=1.2, label=r"$D_0-c/(m-2)!$")
axa.set_xlabel(r"quantization step $\Delta$"); axa.set_ylabel(r"disequilibrium $D$", color=WONG[0])
axa.tick_params(axis="y", labelcolor=WONG[0])
axb2 = axa.twinx()
axb2.plot(grid, Hx, "s", color=WONG[1], ms=4, label="$H$ exact")
axb2.plot(grid, H_lead, "--", color=WONG[1], lw=1.2, label=r"$H_0+\binom{m}{2}c\ln(1/c)/\ln M$")
axb2.set_ylabel(r"normalized entropy $H$", color=WONG[1]); axb2.tick_params(axis="y", labelcolor=WONG[1])
axa.set_title(r"(a) $D\sim\Delta$, $H\sim\Delta\ln(1/\Delta)$ ($m{=}4$, Gaussian)", loc="left")
h1, l1 = axa.get_legend_handles_labels(); h2, l2 = axb2.get_legend_handles_labels()
axa.legend(h1 + h2, l1 + l2, loc="center right", fontsize=6.6)
axa.minorticks_on(); axa.tick_params(which="both", direction="in", top=True)

# (b) universality: (D0-D)/c -> 1/(m-2)! and (H-H0)logM/(c ln 1/c) -> C(m,2), three densities
axc = ax[1]
mk = {"Gaussian": "o", "Laplace": "s", "Uniform": "^"}
for name, (cdf, I) in R3.DENS.items():
    cc = np.array([R3.coords(D, cdf, m)[0] for D in grid])
    DD = np.array([R3.coords(D, cdf, m)[2] for D in grid])
    HH = np.array([R3.coords(D, cdf, m)[3] for D in grid])
    axc.plot(grid, (D0 - DD) / cc, mk[name], color=WONG[0], ms=3.4, lw=0)
    axc.plot(grid, (HH - H0) * log(M) / (cc * np.log(1 / cc)) / comb(m, 2), mk[name],
             color=WONG[1], ms=3.4, lw=0, label=name)
axc.axhline(1 / factorial(m - 2), color=WONG[0], ls="-", lw=1.0)
axc.axhline(1.0, color=WONG[1], ls="--", lw=1.0)
axc.text(0.012, 1 / factorial(m - 2) + 0.01, r"$(D_0-D)/c\to 1/(m-2)!$", color=WONG[0], fontsize=7)
axc.text(0.012, 1.02, r"$(H-H_0)\ln M/[c\ln(1/c)]\,/\,\binom{m}{2}\to 1$", color=WONG[1], fontsize=7)
axc.set_xscale("log"); axc.set_xlabel(r"quantization step $\Delta$")
axc.set_ylabel("normalized leading coefficient")
axc.set_title("(b) Coefficients universal; density enters only via $\\int f^2$", loc="left")
axc.legend(loc="lower right", title="noise density", title_fontsize=7); sm(axc)

fig.savefig(HERE / "figure_R3.png", dpi=400, bbox_inches="tight", pad_inches=0.05)
fig.savefig(HERE / "figure_R3.pdf", bbox_inches="tight", pad_inches=0.05)
print("saved figure_R3.png/pdf")
