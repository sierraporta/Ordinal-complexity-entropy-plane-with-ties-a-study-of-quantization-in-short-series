"""R3: leading-order quantization (Sheppard-style) law for the plane coordinates.

Continuous white noise realizes only the m! strict orders (uniform), so on the GOP
alphabet its coordinates are H0 = log(m!)/log(M), D0 = 1/m! - 1/M. Quantizing with step
Delta activates tie patterns. The controlling small quantity is the per-pair coincidence
probability  c(Delta) = sum_k q_k^2  (q_k the lattice-bin probability), and c ~ Delta * I
with I = integral f(x)^2 dx the density collision integral.

Claims (derived, validated below):
  (D)  D0 - D(Delta) = c/(m-2)! + O(c^2)         -- D shifts LINEARLY in Delta
  (c)  c(Delta)/Delta -> I = integral f^2         -- density enters only through I
  (H)  (H(Delta) - H0) * log(M) / (c * log(1/c)) -> C(m,2)
                                                  -- H shifts like Delta*log(1/Delta), steeper

The coefficient 1/(m-2)! for D and C(m,2) for H are universal; the density sets only I.
"""
import numpy as np
from math import factorial, log, comb, sqrt, pi
from scipy.stats import norm, laplace, uniform
import gop_bias_theory as T

def q_from_cdf(cdf, Delta, K=18):
    K = max(K, int(np.ceil(9.0 / Delta)))
    k = np.arange(-K, K + 1)
    q = cdf((k + 0.5) * Delta) - cdf((k - 0.5) * Delta)
    return q / q.sum()

# unit-variance densities and their exact collision integrals I = int f^2
b = 1 / sqrt(2)                      # Laplace scale for unit variance
a = sqrt(3.0)                        # uniform half-width for unit variance
DENS = {
    "Gaussian": (norm.cdf,                       1 / (2 * sqrt(pi))),   # 0.282095
    "Laplace":  (laplace(scale=b).cdf,           1 / (4 * b)),          # 0.353553
    "Uniform":  (uniform(loc=-a, scale=2*a).cdf, 1 / (2 * a)),          # 0.288675
}

def coords(Delta, cdf, m):
    q = q_from_cdf(cdf, Delta)
    P = T.make_P(q)
    s2, D, tot, H = T.s2_and_D(m, P)
    c = float((q ** 2).sum())
    return c, s2, D, H, tot

if __name__ == "__main__":
    for m in (3, 4):
        M = T.FUBINI[m]
        H0 = log(factorial(m)) / log(M)
        D0 = 1 / factorial(m) - 1 / M
        cD_coeff = 1 / factorial(m - 2)      # predicted D slope vs c
        cH_coeff = comb(m, 2)                 # predicted H coeff
        print(f"\n===== m={m} (M={M}) =====")
        print(f"H0={H0:.6f}  D0={D0:.6f}   predicted: D0-D = c/{factorial(m-2)},  "
              f"(H-H0)logM/(c log 1/c) -> {cH_coeff}")
        for name, (cdf, I) in DENS.items():
            print(f"\n  density={name}   I=int f^2={I:.6f}")
            print(f"    {'Delta':>7} {'c':>10} {'c/Delta':>9} {'(D0-D)/c':>10} "
                  f"{'(H-H0)logM/(c ln1/c)':>22}")
            for Delta in (0.16, 0.08, 0.04, 0.02, 0.01):
                c, s2, D, H, tot = coords(Delta, cdf, m)
                rD = (D0 - D) / c
                rH = (H - H0) * log(M) / (c * log(1 / c))
                print(f"    {Delta:>7} {c:>10.6f} {c/Delta:>9.5f} {rD:>10.5f} {rH:>22.5f}")
        print(f"\n  -> c/Delta approaches I (density), (D0-D)/c approaches {cD_coeff} (=1/(m-2)!), "
              f"(H..)/.. approaches {cH_coeff} (=C(m,2)), all densities.")
