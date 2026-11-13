"""Bernoulli KL divergence and KL-UCB index solver.

Book reference: Chapter 10 (Bernoulli noise / KL-UCB), Algorithm 8.
Source verification: verified against online book PDF (accessed this session).
"""

from __future__ import annotations

import math

import numpy as np


def bernoulli_kl(p: float, q: float) -> float:
    """Binary relative entropy d(p, q) with correct endpoint behaviour.

    Conventions (standard, matching book usage):
    - d(0, 0) = d(1, 1) = 0
    - d(0, q) = log(1/(1-q)) for q in [0, 1); d(0, 1) = +inf
    - d(1, q) = log(1/q) for q in (0, 1]; d(1, 0) = +inf
    - d(p, 0) = +inf for p in (0, 1]; d(p, 1) = +inf for p in [0, 1)
    """
    p = float(p)
    q = float(q)
    if not (0.0 <= p <= 1.0 and 0.0 <= q <= 1.0):
        raise ValueError(f"p, q must be in [0, 1], got p={p}, q={q}")

    if abs(p - q) < 1e-15:
        return 0.0
    if q <= 0.0:
        return math.inf if p > 0.0 else 0.0
    if q >= 1.0:
        return math.inf if p < 1.0 else 0.0

    # Safe logs for p in {0,1}
    term = 0.0
    if p > 0.0:
        term += p * math.log(p / q)
    if p < 1.0:
        term += (1.0 - p) * math.log((1.0 - p) / (1.0 - q))
    return term


def kl_ucb_index(
    empiric_mean: float,
    n_pulls: int,
    threshold: float,
    *,
    tol: float = 1e-8,
    max_iter: int = 64,
) -> float:
    """Solve max {q in [mu_hat, 1] : d(mu_hat, q) <= threshold}.

    Uses bounded monotone bisection on q. If n_pulls == 0, returns +inf
    (caller should treat unpulled arms separately). If threshold <= 0, returns
    empiric_mean clipped to [0, 1].

    Numerical approximation: bisection tolerance ``tol`` (default 1e-8).
    """
    if n_pulls <= 0:
        return math.inf
    mu = min(1.0, max(0.0, float(empiric_mean)))
    thr = float(threshold)
    if thr <= 0.0 or mu >= 1.0:
        return mu

    # Upper endpoint: if d(mu, 1) <= thr then answer is 1
    if bernoulli_kl(mu, 1.0) <= thr:
        return 1.0

    lo, hi = mu, 1.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if bernoulli_kl(mu, mid) <= thr:
            lo = mid
        else:
            hi = mid
        if hi - lo <= tol:
            break
    return lo
