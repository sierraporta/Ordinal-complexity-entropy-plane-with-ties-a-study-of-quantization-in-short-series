"""Exact tie-alphabet coordinates of a quantized Gaussian AR(1) process, via
multivariate-normal box probabilities.

A length-m window of a stationary AR(1) with lag-1 correlation phi is m-variate normal
with correlation phi^{|i-j|} and unit variance. Quantized to step Delta, the probability of
a generalized-ordinal pattern pi is the sum, over lattice-level tuples (k_0,...,k_{m-1}) whose
weak order equals pi, of the box probability Pr(X_i in bin k_i for all i).

AR(1) is Markov, so the box probability factorizes through a transfer operator. We place
Gauss-Legendre nodes inside each Delta-bin, build the one-step transition weight
T[g,g'] = N(x_{g'}; phi x_g, 1-phi^2) w_{g'}, start from the stationary weights
s[g] = phi(x_g) w_g, and accumulate pattern probabilities by a depth-m pass that restricts each
position to a bin and propagates. This is exact up to the quadrature, and reduces to independence
(the white-noise theory) at phi=0.

Validates against the R4 simulation. Requires numpy, scipy.
"""
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.stats import norm
from itertools import combinations
from math import log

MFUB = 75  # a(4)

def _code(levels):
    c = 0
    for p, (i, j) in enumerate(combinations(range(len(levels)), 2)):
        c += ((levels[i] > levels[j]) - (levels[i] < levels[j]) + 1) * (3 ** p)
    return c

def ar1_exact(phi, Delta, m=4, q=8, nsig=5.5):
    K = int(np.ceil(nsig / Delta)) + 1
    sig = np.sqrt(1 - phi ** 2) if phi != 0 else 1.0
    tl, wl = leggauss(q)
    xs, ws, levs = [], [], []
    for k in range(-K, K + 1):
        a, b = (k - 0.5) * Delta, (k + 0.5) * Delta
        xs.append((a + b) / 2 + (b - a) / 2 * tl)
        ws.append((b - a) / 2 * wl)
        levs.append(np.full(q, k))
    x = np.concatenate(xs); w = np.concatenate(ws); lev = np.concatenate(levs)
    G = len(x)
    s = norm.pdf(x) * w
    if phi == 0:
        Trow = norm.pdf(x) * w
        T = np.broadcast_to(Trow, (G, G))
    else:
        T = norm.pdf(x[None, :], loc=phi * x[:, None], scale=sig) * w[None, :]
    Ks = np.arange(-K, K + 1)
    masks = {int(k): (lev == k) for k in Ks}

    p = np.zeros(3 ** (m * (m - 1) // 2))
    tie = np.zeros(1)
    def rec(pos, dist, labels):
        for k in Ks:
            db = dist * masks[int(k)]
            tot = db.sum()
            if tot < 1e-14:
                continue
            lab = labels + [int(k)]
            if pos == m - 1:
                p[_code(lab)] += tot
                if len(set(lab)) < m:
                    tie[0] += tot
            else:
                rec(pos + 1, db @ T, lab)
    rec(0, s, [])
    p = p / p.sum()
    pos = p[p > 0]
    H = -(pos * np.log(pos)).sum() / log(MFUB)
    D = (p ** 2).sum() - 1.0 / MFUB
    return H, D, float(tie[0] / p.sum() if False else tie[0])


if __name__ == "__main__":
    # R4 simulation values, for comparison
    sim = {(0.0, 0.5): (0.9929, 7.63e-4), (0.0, 1.0): (0.9561, 4.98e-3),
           (0.5, 0.5): (0.9800, 2.38e-3), (0.9, 0.5): (0.8518, 1.88e-2),
           (0.9, 1.0): (0.6139, 1.04e-1)}
    print("Exact quantized-AR(1) coordinates (MVN box probabilities), m=4:")
    print(f"{'phi':>5} {'Delta':>6} {'tie%':>7} {'H exact':>9} {'H sim':>8} {'D exact':>11} {'D sim':>10}")
    for (phi, D) in [(0.0, 0.5), (0.0, 1.0), (0.5, 0.5), (0.9, 0.5), (0.9, 1.0)]:
        H, Dd, tf = ar1_exact(phi, D)
        Hs, Ds = sim[(phi, D)]
        print(f"{phi:>5} {D:>6} {100*tf:>7.1f} {H:>9.5f} {Hs:>8.4f} {Dd:>11.4e} {Ds:>10.2e}")
