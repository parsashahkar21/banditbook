"""Exp3-IX for adversarial bandits with rewards in [0, 1].

Book reference: Chapter 12, Algorithm 10.
Source verification: verified against online book PDF (accessed this session).

Works in the loss formulation: Y_t = 1 - X_t,
  L̂_ti ← L̂_{t-1,i} + I{At=i} (1-X_t) / (P_ti + γ)
  P_ti ∝ exp(-η L̂_{t-1,i})

Default (δ-independent): η = sqrt(2 log(k+1)/(n k)), γ = η/2 (Theorem 12.1 part 1).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from banditbook.utils.validation import check_positive_int

from .base import Agent, Feedback


class Exp3IXAgent(Agent):
    name = "exp3ix"
    source = "Book Ch. 12, Algorithm 10 (exact)"

    def __init__(
        self,
        n_arms: int,
        horizon: int,
        rng: np.random.Generator | None = None,
        *,
        eta: float | None = None,
        gamma: float | None = None,
        delta: float | None = None,
    ):
        super().__init__(n_arms, rng)
        self.horizon = check_positive_int("horizon", horizon)
        n, k = self.horizon, self.n_arms
        if eta is None:
            if delta is None:
                # η1 from Theorem 12.1
                self.eta = float(np.sqrt(2.0 * np.log(k + 1) / (n * k)))
            else:
                if not 0 < delta < 1:
                    raise ValueError("delta must be in (0, 1)")
                self.eta = float(
                    np.sqrt((np.log(k) + np.log((k + 1) / delta)) / (n * k))
                )
        else:
            if eta <= 0:
                raise ValueError("eta must be > 0")
            self.eta = float(eta)
        self.gamma = float(self.eta / 2.0) if gamma is None else float(gamma)
        if self.gamma < 0:
            raise ValueError("gamma must be >= 0")
        self.L = np.zeros(self.n_arms, dtype=float)
        self._last_P: np.ndarray | None = None

    def _probs(self) -> np.ndarray:
        # softmax(-η L)
        logits = -self.eta * self.L
        logits -= logits.max()
        w = np.exp(logits)
        return w / w.sum()

    def select_action(self, context: Any = None) -> int:
        P = self._probs()
        self._last_P = P
        return int(self.rng.choice(self.n_arms, p=P))

    def update(self, action: int, feedback: Feedback) -> None:
        r = float(feedback.reward)
        if r < 0.0 or r > 1.0:
            raise ValueError(
                f"Exp3-IX requires rewards in [0, 1]; got {r}."
            )
        if self._last_P is None:
            raise RuntimeError("select_action must be called before update")
        P = self._last_P
        loss = 1.0 - r
        self.L[action] += loss / (P[action] + self.gamma)
        self.t += 1
        self._last_P = None

    def _reset_state(self) -> None:
        self.L[:] = 0.0
        self._last_P = None
