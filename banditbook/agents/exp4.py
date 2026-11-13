"""Exp4 — exponential weighting for exploration and exploitation with experts.

Book reference: Chapter 18, Algorithm 11.
Source verification: verified against online book PDF (accessed this session).

Finite expert/policy interface: each round the caller supplies E(t) of shape
(M, K) with rows as probability distributions over arms.
Default γ=0 (original Exp4). η = sqrt(2 log(M)/(n k)) from Theorem 18.1.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from banditbook.utils.validation import check_positive_int

from .base import ContextualAgent, Feedback


class Exp4Agent(ContextualAgent):
    name = "exp4"
    source = "Book Ch. 18, Algorithm 11 (exact; γ=0 default)"

    def __init__(
        self,
        n_arms: int,
        n_experts: int,
        horizon: int,
        rng: np.random.Generator | None = None,
        *,
        eta: float | None = None,
        gamma: float = 0.0,
    ):
        super().__init__(n_arms, rng)
        self.n_experts = check_positive_int("n_experts", n_experts)
        self.horizon = check_positive_int("horizon", horizon)
        self.gamma = float(gamma)
        if self.gamma < 0:
            raise ValueError("gamma must be >= 0")
        if eta is None:
            self.eta = float(
                np.sqrt(2.0 * np.log(self.n_experts) / (self.horizon * self.n_arms))
            )
        else:
            if eta <= 0:
                raise ValueError("eta must be > 0")
            self.eta = float(eta)
        self.Q = np.full(self.n_experts, 1.0 / self.n_experts)
        self._last_P: np.ndarray | None = None
        self._last_E: np.ndarray | None = None

    def _select_with_context(self, context: Any) -> int:
        E = np.asarray(context, dtype=float)
        if E.ndim != 2 or E.shape != (self.n_experts, self.n_arms):
            raise ValueError(
                f"Exp4 context must be advice matrix shape {(self.n_experts, self.n_arms)}, got {E.shape}"
            )
        # Rows should be probability vectors; renormalise lightly for numerics
        row_sums = E.sum(axis=1, keepdims=True)
        if np.any(row_sums <= 0):
            raise ValueError("each expert row must be a non-zero distribution")
        E = E / row_sums
        P = self.Q @ E
        P = np.maximum(P, 0.0)
        s = P.sum()
        if s <= 0:
            P = np.full(self.n_arms, 1.0 / self.n_arms)
        else:
            P = P / s
        self._last_P = P
        self._last_E = E
        return int(self.rng.choice(self.n_arms, p=P))

    def update(self, action: int, feedback: Feedback) -> None:
        r = float(feedback.reward)
        if r < 0.0 or r > 1.0:
            raise ValueError("Exp4 requires rewards in [0, 1]")
        if self._last_P is None or self._last_E is None:
            raise RuntimeError("select_action must be called before update")
        P = self._last_P
        E = self._last_E
        # X̂_ti = 1 - I{At=i}(1-Xt)/(P_ti + γ)
        X_hat = np.ones(self.n_arms)
        X_hat[action] = 1.0 - (1.0 - r) / (P[action] + self.gamma)
        # Propagate to experts
        X_tilde = E @ X_hat
        # Qt+1,i ∝ Qt,i exp(η X̃_ti)
        log_Q = np.log(np.maximum(self.Q, 1e-300)) + self.eta * X_tilde
        log_Q -= log_Q.max()
        w = np.exp(log_Q)
        self.Q = w / w.sum()
        self.t += 1
        self._last_P = None
        self._last_E = None

    def _reset_state(self) -> None:
        self.Q[:] = 1.0 / self.n_experts
        self._last_P = None
        self._last_E = None
