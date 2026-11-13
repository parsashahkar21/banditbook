"""Finite-action approximate G/D-optimal design (Frank–Wolfe / Fedorov–Wynn).

Book reference: Chapter 21 (Kiefer–Wolfowitz), used by Algorithm 12 (Ch. 22).
Source verification: verified against online book PDF (accessed this session).

Restriction: finite action sets only. Not claimed for continuous/arbitrary sets.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DesignResult:
    weights: np.ndarray  # shape (k,), sums to 1
    V: np.ndarray  # Σ π(a) a a^T
    max_leverage: float  # max_a ||a||_{V^{-1}}^2
    optimality_gap: float  # max_leverage - dim (Kiefer–Wolfowitz: =0 at optimum)
    n_iter: int


def g_optimal_design(
    actions: np.ndarray,
    *,
    max_iter: int = 500,
    tol: float = 1e-4,
    ridge: float = 1e-10,
) -> DesignResult:
    """Approximate G-optimal design on a finite action set via Frank–Wolfe.

    Maximises log det V(π) over distributions π on the rows of ``actions``
    (shape k×d). At a G-optimum, max_a a^T V(π)^{-1} a = d.

    Returns a numerical optimality diagnostic ``optimality_gap = max_leverage - d``.
    Sampling/rounding of π into integer pull counts is left to the caller
    (see PhasedElimination).
    """
    A = np.asarray(actions, dtype=float)
    if A.ndim != 2:
        raise ValueError("actions must be 2-D (k, d)")
    k, d = A.shape
    if k < 1:
        raise ValueError("need at least one action")

    # Initialise with uniform design
    pi = np.full(k, 1.0 / k)
    V = (A.T * pi) @ A + ridge * np.eye(d)

    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        try:
            V_inv = np.linalg.inv(V)
        except np.linalg.LinAlgError:
            V_inv = np.linalg.pinv(V)
        leverages = np.einsum("ij,jk,ik->i", A, V_inv, A)
        j = int(np.argmax(leverages))
        gap = float(leverages[j] - d)
        if gap <= tol:
            break
        # Frank–Wolfe step toward vertex e_j; step size 1/(n_iter+1) (Wynn–Fedorov style)
        gamma = 2.0 / (n_iter + 2.0)
        pi = (1.0 - gamma) * pi
        pi[j] += gamma
        V = (1.0 - gamma) * (V - ridge * np.eye(d)) + gamma * np.outer(A[j], A[j]) + ridge * np.eye(d)

    try:
        V_inv = np.linalg.inv(V)
    except np.linalg.LinAlgError:
        V_inv = np.linalg.pinv(V)
    leverages = np.einsum("ij,jk,ik->i", A, V_inv, A)
    max_lev = float(np.max(leverages))
    return DesignResult(
        weights=pi,
        V=V - ridge * np.eye(d),
        max_leverage=max_lev,
        optimality_gap=max_lev - d,
        n_iter=n_iter,
    )


def round_design_to_counts(weights: np.ndarray, total: int) -> np.ndarray:
    """Largest-remainder rounding of a design to integer sample counts summing to ``total``."""
    weights = np.asarray(weights, dtype=float)
    if total < 0:
        raise ValueError("total must be >= 0")
    if total == 0:
        return np.zeros(len(weights), dtype=int)
    raw = weights * total
    counts = np.floor(raw).astype(int)
    rem = total - int(counts.sum())
    order = np.argsort(-(raw - counts))
    counts[order[:rem]] += 1
    return counts
