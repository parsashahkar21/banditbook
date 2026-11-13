"""Educational demonstration: concentration of Gaussian sample means.

Book intuition: Chapter 5 (concentration) / subgaussian tails.
This is a Monte Carlo illustration — not a proof and not a finite-time lower bound.
"""

from __future__ import annotations

import numpy as np


def empirical_tail_probability(n: int = 50, delta_threshold: float = 0.5, reps: int = 5000, seed: int = 0):
    """P(|μ̂_n| >= t) for N(0,1) samples vs subgaussian bound 2 exp(-n t² / 2)."""
    rng = np.random.default_rng(seed)
    means = rng.normal(0, 1, size=(reps, n)).mean(axis=1)
    empirical = float(np.mean(np.abs(means) >= delta_threshold))
    bound = float(2 * np.exp(-n * delta_threshold**2 / 2))
    return {"empirical": empirical, "subgaussian_bound": bound, "n": n, "t": delta_threshold}


def bernoulli_kl_vs_pinsker(p: float = 0.2, q: float = 0.5):
    from banditbook.utils.kl import bernoulli_kl

    d = bernoulli_kl(p, q)
    pinsker_lhs = 2 * (p - q) ** 2  # Pinsker: d >= 2 (p-q)^2 for Bernoulli (book Lemma 10.2 style)
    return {"kl": d, "two_delta_sq": pinsker_lhs, "kl_ge_two_delta_sq": d >= pinsker_lhs - 1e-12}


if __name__ == "__main__":
    print(empirical_tail_probability())
    print(bernoulli_kl_vs_pinsker())
