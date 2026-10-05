"""Real short-series diagnostic battery for the ordinal complexity-entropy plane,
following Borin et al. Given a 1D series it reports the sampling scheme, the tie
fraction under a declared policy, the plane coordinates under the plug-in,
distinct-pairs and gap estimators, and a calibrated Monte Carlo rank test against
continuous IID white noise. This is the engine for the real-data application the
manuscript flags as its natural next step (a physiological RR record and a
heliospheric/Forbush series).

Estimator definitions are identical to the reproduction notebook.
"""
from itertools import permutations
from math import factorial, log, ceil
import numpy as np
from scipy.special import xlogy, digamma

def codebook(m):
    return np.sort(np.array([list(p) for p in permutations(range(m))]) @ (m ** np.arange(m)))

def _windows(x, m, tau, stride=1):
    span = (m - 1) * tau + 1
    return np.lib.stride_tricks.sliding_window_view(np.asarray(x, float), span)[::stride, ::tau]

def tie_fraction(x, m, tau=1, stride=1):
    e = _windows(x, m, tau, stride)
    return float(np.mean(np.any(np.diff(np.sort(e, 1), 1) == 0, 1)))

def symbols(x, m, tau=1, stride=1, policy="index", jitter_scale=1e-9, rng=None):
    """policy: 'strict' (raise on ties), 'index' (ties broken by position, stable
    argsort), or 'jitter' (add uniform noise below the quantization step)."""
    e = _windows(x, m, tau, stride)
    if policy == "jitter":
        rng = rng or np.random.default_rng(0)
        e = e + rng.uniform(-jitter_scale, jitter_scale, e.shape)
    r = np.argsort(e, axis=1, kind="stable")
    if policy == "strict":
        ordered = np.take_along_axis(e, r, 1)
        if np.any(np.diff(ordered, axis=1) == 0):
            raise ValueError("ties present; choose policy='index' or 'jitter'")
    return np.searchsorted(codebook(m), r @ (m ** np.arange(m)))

def _hist(s, M):
    return np.bincount(s, minlength=M)

def q0(M):
    q = np.full(M, 1 / (2 * M)); q[0] = (1 + 1 / M) / 2
    return 1 / (-xlogy(q, q).sum() - .5 * log(M))

def coordinates(s, M):
    c = _hist(s, M).astype(float); n = c.sum(); p = c / n; K = int((c > 0).sum())
    S = -xlogy(p, p).sum(); H = S / log(M)
    D = ((p - 1 / M) ** 2).sum()
    q = (p + 1 / M) / 2
    J = -xlogy(q, q).sum() - .5 * S - .5 * log(M)
    Hmm = H + (K - 1) / (2 * n * log(M))
    Du = (c * (c - 1)).sum() / (n * (n - 1)) - 1 / M
    return dict(n=int(n), missing=M - K, H=H, H_MM=Hmm, D=D, D_U=Du,
                C_LMC=H * D, C_JS=q0(M) * H * J)

def gap_D(s, M, w):
    n = s.shape[-1]; c = _hist(s, M).astype(float)
    col = (c * (c - 1)).sum() / 2
    for h in range(1, w + 1):
        col -= np.sum(s[:-h] == s[h:])
    return col / ((n - w - 1) * (n - w) / 2) - 1 / M

def score(s, M):
    co = coordinates(s, M)
    return (1 - co["H"]) ** 2 + co["C_JS"] ** 2

def rank_test(x, m, tau=1, policy="index", B_null=9999, seed=0):
    """Calibrated one-sided rank test against continuous IID white noise.
    Returns (T_obs, p_value). p = (1 + #{T_null >= T_obs}) / (B+1)."""
    M = factorial(m); N = len(x)
    s = symbols(x, m, tau, policy=policy, rng=np.random.default_rng(seed))
    T = score(s, M)
    rng = np.random.default_rng(seed + 1)
    span = (m - 1) * tau + 1
    Tn = np.empty(B_null)
    for b in range(B_null):
        sw = symbols(rng.standard_normal(N), m, tau, policy="index")
        Tn[b] = score(sw, M)
    p = (1 + int(np.sum(Tn >= T))) / (B_null + 1)
    return float(T), float(p)

def report(x, m=4, tau=1, label="series", policies=("index", "jitter"), B_null=9999, seed=0):
    M = factorial(m); w = (m - 1) * tau
    print(f"=== {label} | N={len(x)} | m={m} tau={tau} | gap w=(m-1)tau={w} ===")
    print(f"tie fraction (strict windows): {tie_fraction(x, m, tau):.3f}")
    for pol in policies:
        s = symbols(x, m, tau, policy=pol, rng=np.random.default_rng(seed))
        co = coordinates(s, M); Dg = gap_D(s, M, w)
        T, p = rank_test(x, m, tau, policy=pol, B_null=B_null, seed=seed)
        print(f"  policy={pol:7s} windows n={co['n']} missing={co['missing']}/{M}")
        print(f"     H={co['H']:.4f} (MM {co['H_MM']:.4f}) | "
              f"D: plug-in {co['D']:.5f}  U {co['D_U']:+.5f}  gap {Dg:+.5f}")
        print(f"     C_LMC={co['C_LMC']:.5f}  C_JS={co['C_JS']:.5f} | rank test vs white noise: T={T:.4f} p={p:.4f}")


if __name__ == "__main__":
    rng = np.random.default_rng(20260922)
    # synthetic Forbush-like: integer neutron-monitor counts, high baseline with a
    # non-stationary dip (the Forbush decrease) -> real quantization + short + drift
    N = 360
    base = 8000 + 40 * np.sin(np.arange(N) / 20)
    dip = -600 * np.exp(-((np.arange(N) - 140) ** 2) / (2 * 25 ** 2))
    counts = rng.poisson(np.clip(base + dip, 1, None)).astype(float)
    report(counts, m=4, tau=1, label="synthetic Forbush counts (integer)")

    print()
    # synthetic RR-like: ms-quantized inter-beat intervals, short record
    N = 500
    rr = 800 + np.cumsum(rng.standard_normal(N) * 6)          # slow wander
    rr += 25 * np.sin(np.arange(N) / 8)                        # respiratory-like
    rr_ms = np.round(rr)                                       # quantize to 1 ms
    report(rr_ms, m=4, tau=1, label="synthetic RR intervals (ms-quantized)")
