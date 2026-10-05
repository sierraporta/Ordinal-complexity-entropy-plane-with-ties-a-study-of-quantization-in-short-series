"""Extensions to Borin et al., "Finite-sample bias in ordinal complexity-entropy
planes." Two additions that remove stated limitations of the manuscript:

  A. General embedding delay tau > 1. The exact overlap identity generalizes with
     a sparsity structure: for continuous IID data, two ordinal windows separated
     by start-lag h share samples only when h is a multiple of tau, and then the
     lagged collision term equals the tau=1 term at lag h/tau:

         g_tau(h) = g_1(h/tau)   if  tau | h  and  1 <= h/tau <= m-1,   else 0.

     Consequently the plug-in and distinct-pairs biases are sums over d = 1..m-1
     of (n - d*tau) * g_1(d), and the gap that makes the estimator exactly
     unbiased for white noise is  w = (m-1)*tau.

  B. AR(1) ground truth for m = 5 (and any m) via numerical multivariate-normal
     orthant probabilities, extending the closed-form arcsine expressions that
     the manuscript provides only for m = 3, 4. Validated against the arcsine
     values for m = 3, 4 and against simulation for m = 5.

Run `python3 fsb_extensions.py` for the self-test.
"""
from itertools import permutations
from fractions import Fraction
from math import factorial, log
import numpy as np
from scipy.stats import multivariate_normal as _MVN
from scipy.special import xlogy


def codebook(m):
    return np.sort(np.array([list(p) for p in permutations(range(m))]) @ (m ** np.arange(m)))


# ---------- A. general-tau exact overlap identities ----------
def g1_terms(m):
    """Exact tau=1 collision terms g_1(d) = Pr(Pi_t = Pi_{t+d}) - 1/m!, d = 1..m-1."""
    out = []
    for d in range(1, m):
        perms = np.array([list(p) for p in permutations(range(m + d))])
        same = np.all(np.argsort(perms[:, :m], 1) == np.argsort(perms[:, d:d + m], 1), 1).sum()
        out.append(Fraction(int(same), factorial(m + d)) - Fraction(1, factorial(m)))
    return out


def g_tau(m, tau, h):
    """Exact g_tau(h) for continuous IID data (sparsity identity)."""
    if tau >= 1 and h % tau == 0 and 1 <= h // tau <= m - 1:
        return g1_terms(m)[h // tau - 1]
    return Fraction(0)


def tau_biases(m, tau, n):
    """Exact finite-sample white-noise biases of the disequilibrium under overlap,
    for embedding delay tau. n is the number of ordinal windows.
    Returns (plug-in bias, distinct-pairs U bias)."""
    M = factorial(m)
    base = g1_terms(m)
    s = sum((n - d * tau) * base[d - 1] for d in range(1, m) if d * tau < n)
    return (1 - Fraction(1, M)) / n + 2 * s / n ** 2, 2 * s / (n * (n - 1))


def gap_width(m, tau=1):
    """Gap that makes the disequilibrium estimator exactly unbiased for IID white
    noise at embedding delay tau."""
    return (m - 1) * tau


# ---------- B. AR(1) ordinal probabilities for any m via orthant probability ----------
def ar_probabilities_orthant(m, phi, tau=1):
    """Stationary Gaussian AR(1) ordinal pattern probabilities for any m >= 2.
    For m = 3, 4 this matches the closed-form arcsine expression; for m >= 5 it is
    the numerical multivariate-normal orthant probability."""
    if abs(phi) >= 1:
        raise ValueError("stationary AR(1) requires |phi| < 1")
    R = phi ** (tau * np.abs(np.arange(m)[:, None] - np.arange(m)))
    cb = codebook(m)
    p = np.zeros(factorial(m))
    for rank in permutations(range(m)):
        B = np.zeros((m - 1, m))
        for k in range(m - 1):
            B[k, rank[k + 1]] = 1.0
            B[k, rank[k]] = -1.0
        V = B @ R @ B.T
        val = _MVN(mean=np.zeros(m - 1), cov=V, allow_singular=True).cdf(np.zeros(m - 1))
        code = sum(v * m ** k for k, v in enumerate(rank))
        p[int(np.searchsorted(cb, code))] = val
    return p / p.sum()


def functionals(p):
    p = np.asarray(p, float); M = p.shape[-1]
    S = -xlogy(p, p).sum(-1); H = S / log(M)
    D = ((p - 1 / M) ** 2).sum(-1)
    return H, D, H * D


# ---------- self-test ----------
if __name__ == "__main__":
    print("A. tau>1 identity  g_tau(h)=g1(h/tau)*[tau|h]:")
    for m in (3, 4):
        base = g1_terms(m)
        for tau in (2, 3):
            ok = all(
                g_tau(m, tau, h) == (base[h // tau - 1] if (h % tau == 0 and 1 <= h // tau <= m - 1) else Fraction(0))
                for h in range(1, (m - 1) * tau + 2))
            print(f"   m={m} tau={tau}: {'OK' if ok else 'FAIL'} | gap w=(m-1)tau={gap_width(m, tau)}")

    print("\nB. AR(1) orthant vs closed-form arcsine (m=3,4) and D:")
    for m in (3, 4):
        for phi in (0.5, 0.9):
            p = ar_probabilities_orthant(m, phi)
            print(f"   m={m} phi={phi}: D={functionals(p)[1]:.6f}, sum={p.sum():.6f}")
    print("   m=5 AR(1) ground truth (D, H):")
    for phi in (0.5, 0.9):
        p = ar_probabilities_orthant(5, phi)
        H, D, _ = functionals(p)
        print(f"   m=5 phi={phi}: D={D:.6f}  H={H:.6f}  allowed={(p > 1e-9).sum()}/120")
