"""Exact finite-sample overlap-bias theory for the generalized-ordinal-pattern (GOP,
tie-inclusive) alphabet, for Delta-quantized Gaussian white noise.

Each value falls on lattice level k with probability q_k = Phi((k+.5)D) - Phi((k-.5)D).
The values are IID, so the probability that u of them realize a given weak order omega
(ordered set partition with group sizes b_0,...,b_{g-1} at increasing levels) is

    P(omega) = sum_{k_0 < k_1 < ... < k_{g-1}}  prod_j q_{k_j}^{b_j},

computed by a level-by-level DP. From this:
  s2      = sum_pi p_pi^2                      (p_pi = P of an m-element weak order)
  r(h)    = Pr(two windows at lag h share a GOP pattern)
          = sum over weak orders of the (m+h)-element union with equal sub-window patterns
  g(h)    = r(h) - s2

and the finite-sample white-noise biases of the disequilibrium follow with the SAME
identities as the permutation case (only s2 and g change):

  plug-in bias = (1 - s2)/n + (2/n^2)   sum_{h=1}^{m-1} (n-h) g(h)
  U       bias = (2/(n(n-1)))           sum_{h=1}^{m-1} (n-h) g(h)
  gap (w=m-1)  = 0                        (all overlapping lags excluded; IID beyond)

Validated below against direct Monte Carlo. Requires numpy, scipy.
"""
import numpy as np
from math import factorial
from itertools import product
from functools import lru_cache
from scipy.stats import norm

FUBINI = {2: 3, 3: 13, 4: 75, 5: 541, 6: 4683, 7: 47293}


def q_levels(Delta, K=18):
    K = max(K, int(np.ceil(7.0 / Delta)))   # cover ~7 sigma whatever the step
    k = np.arange(-K, K + 1)
    q = norm.cdf((k + 0.5) * Delta) - norm.cdf((k - 0.5) * Delta)
    return q / q.sum()                      # renormalize away the truncated tails


def weak_orders(u):
    """All weak-order rank vectors on u labelled positions (surjective onto 0..g-1)."""
    out = []
    for a in product(range(u), repeat=u):
        mx = max(a)
        if set(a) == set(range(mx + 1)):
            out.append(a)
    return out


def dense(sub):
    """Canonical dense-rank tuple of a sub-window (equal values share a rank)."""
    order = sorted(set(sub))
    idx = {v: i for i, v in enumerate(order)}
    return tuple(idx[v] for v in sub)


def _composition(rankvec):
    g = max(rankvec) + 1
    b = [0] * g
    for r in rankvec:
        b[r] += 1
    return tuple(b)


def make_P(q):
    """P(omega) as a function of the ordered group-size composition, memoized."""
    L = len(q)

    @lru_cache(maxsize=None)
    def P(comp):
        g = len(comp)
        dp = np.zeros(g + 1)
        dp[0] = 1.0
        for k in range(L):
            qk = q[k]
            # place the next group (index j-1, exponent comp[j-1]) at level k
            for j in range(g, 0, -1):
                dp[j] += dp[j - 1] * qk ** comp[j - 1]
        return dp[g]
    return P


def s2_and_D(m, P):
    M = FUBINI[m]
    s2 = 0.0; tot = 0.0; S = 0.0
    for w in weak_orders(m):
        p = P(_composition(w)); s2 += p * p; tot += p
        if p > 0:
            S -= p * np.log(p)
    H = S / np.log(M)
    return s2, s2 - 1.0 / M, tot, H       # (s2, true D, sanity sum ~ 1, true H)


def r_of_h(m, h, P):
    """Pr(pattern(window_0) == pattern(window_h)) over the (m+h)-var union."""
    u = m + h
    acc = 0.0
    for w in weak_orders(u):
        if dense(w[0:m]) == dense(w[h:h + m]):
            acc += P(_composition(w))
    return acc


def theory(Delta, m=4, N=200, K=18):
    q = q_levels(Delta, K)
    P = make_P(q)
    s2, D_true, tot, H_true = s2_and_D(m, P)
    g = {h: r_of_h(m, h, P) - s2 for h in range(1, m)}
    n = N - (m - 1)
    S = sum((n - h) * g[h] for h in g)
    plug_bias = (1 - s2) / n + 2 * S / n ** 2
    U_bias = 2 * S / (n * (n - 1))
    return dict(s2=s2, D_true=D_true, H_true=H_true, sum_p=tot, g=g, n=n,
                plug_bias=plug_bias, U_bias=U_bias)


# ---------------- Monte Carlo cross-check ----------------
from itertools import combinations
def gop_codes(x, m):
    e = np.lib.stride_tricks.sliding_window_view(np.asarray(x, float), m)
    code = np.zeros(e.shape[0], dtype=np.int64)
    for p, (i, j) in enumerate(combinations(range(m), 2)):
        code += (np.sign(e[:, i] - e[:, j]).astype(np.int64) + 1) * (3 ** p)
    return code

def mc_biases(Delta, m=4, N=200, B=8000, seed=1):
    M = FUBINI[m]; rng = np.random.default_rng(seed)
    Dp = np.empty(B); Du = np.empty(B)
    for b in range(B):
        x = np.round(rng.standard_normal(N) / Delta) * Delta
        c = np.bincount(gop_codes(x, m)); c = c[c > 0].astype(float)
        n = c.sum(); p = c / n
        Dp[b] = (p ** 2).sum() - 1 / M
        Du[b] = (c * (c - 1)).sum() / (n * (n - 1)) - 1 / M
    return Dp.mean(), Du.mean(), Dp.std() / np.sqrt(B), Du.std() / np.sqrt(B)


if __name__ == "__main__":
    m, N = 4, 200
    # sanity: Delta -> 0 must recover Borin's continuous g1(d)
    print("Sanity: g^tie(h) -> continuous permutation g1(d) [Borin] as Delta -> 0")
    import fsb_extensions as E  # continuous g1 terms (Fractions)
    g1c = [float(x) for x in E.g1_terms(m)]
    print("   continuous g1(d):     ", [f"{v:+.6f}" for v in g1c])
    for Delta in (0.2, 0.1, 0.05):
        th0 = theory(Delta, m, N)
        print(f"   g^tie (Delta={Delta:>4}):   ", [f"{th0['g'][h]:+.6f}" for h in (1, 2, 3)],
              f" tie-driven s2={th0['s2']:.6f} (cont. 1/24={1/24:.6f})")

    print(f"\nExact theory vs Monte Carlo (m={m}, N={N}):")
    print(f"{'Delta':>6} {'s2':>9} {'D_true':>11} {'sum_p':>8} | "
          f"{'g(1)':>9} {'g(2)':>9} {'g(3)':>9} | "
          f"{'plug thy':>10} {'plug MC':>10} | {'U thy':>10} {'U MC':>10}")
    for Delta in (0.5, 0.75, 1.0):
        th = theory(Delta, m, N)
        mp, mu, sp, su = mc_biases(Delta, m, N, B=8000, seed=7)
        # MC biases are E[Dhat]-D_true
        print(f"{Delta:>6} {th['s2']:>9.5f} {th['D_true']:>11.4e} {th['sum_p']:>8.5f} | "
              f"{th['g'][1]:>+9.5f} {th['g'][2]:>+9.5f} {th['g'][3]:>+9.5f} | "
              f"{th['plug_bias']:>+10.3e} {mp-th['D_true']:>+10.3e} | "
              f"{th['U_bias']:>+10.3e} {mu-th['D_true']:>+10.3e}")
    print("  (MC standard errors ~1e-5)")
