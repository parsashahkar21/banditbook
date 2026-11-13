"""Regularised least squares and confidence-ellipsoid utilities.

Book reference: Chapter 19–20 (stochastic linear bandits / least-squares confidence).
Source verification: verified against online book PDF (accessed this session).
"""

from __future__ import annotations

import numpy as np


class RegularizedLeastSquares:
    """Online RLS: V_0 = λI, V_t = V_0 + Σ a_s a_s^T, θ̂_t = V_t^{-1} Σ a_s X_s.

    Exact closed form (Eq. 19.4–19.6) using Cholesky solves for numerical stability.
    """

    def __init__(self, dim: int, lam: float = 1.0):
        if dim < 1:
            raise ValueError("dim must be >= 1")
        if lam < 0:
            raise ValueError("lam must be >= 0")
        self.dim = int(dim)
        self.lam = float(lam)
        self.V = self.lam * np.eye(self.dim)
        self.b = np.zeros(self.dim)
        self._chol = None
        self._theta = None
        self._dirty = True

    def update(self, action: np.ndarray, reward: float) -> None:
        a = np.asarray(action, dtype=float).reshape(-1)
        if a.shape != (self.dim,):
            raise ValueError(f"action shape {a.shape} != ({self.dim},)")
        self.V += np.outer(a, a)
        self.b += float(reward) * a
        self._dirty = True

    def _factorize(self) -> None:
        if not self._dirty:
            return
        # Add tiny jitter if λ=0 and V is singular in early rounds.
        try:
            self._chol = np.linalg.cholesky(self.V)
        except np.linalg.LinAlgError:
            jitter = 1e-8 * np.eye(self.dim)
            self._chol = np.linalg.cholesky(self.V + jitter)
        # Solve V θ = b
        y = np.linalg.solve(self._chol, self.b)
        self._theta = np.linalg.solve(self._chol.T, y)
        self._dirty = False

    @property
    def theta_hat(self) -> np.ndarray:
        self._factorize()
        return self._theta.copy()

    def mahalanobis_norm_sq(self, x: np.ndarray) -> float:
        """||x||_{V^{-1}}^2 = x^T V^{-1} x."""
        self._factorize()
        x = np.asarray(x, dtype=float).reshape(-1)
        y = np.linalg.solve(self._chol, x)
        return float(np.dot(y, y))

    def ucb_linear(self, action: np.ndarray, beta_sqrt: float) -> float:
        """⟨θ̂, a⟩ + β^{1/2} ||a||_{V^{-1}} (OFUL/LinUCB index form)."""
        self._factorize()
        a = np.asarray(action, dtype=float).reshape(-1)
        mean = float(np.dot(self._theta, a))
        width = float(beta_sqrt) * math_sqrt(self.mahalanobis_norm_sq(a))
        return mean + width


def math_sqrt(x: float) -> float:
    return float(np.sqrt(max(0.0, x)))


def beta_oful(delta: float, d: int, lam: float, n: int, L: float, m2: float) -> float:
    """β_n^{1/2} from Eq. (19.8) (square-root of the book β_n).

    Book: √β_n = √(λ m²) + √(2 log(1/δ) + d log((dλ + n L²)/(d λ)))
    We return √β_n for use in the UCB index.
    """
    if not 0 < delta < 1:
        raise ValueError("delta must be in (0, 1)")
    inner = 2.0 * np.log(1.0 / delta) + d * np.log((d * lam + n * L**2) / (d * lam))
    return float(np.sqrt(lam * m2) + np.sqrt(max(0.0, inner)))
