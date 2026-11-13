"""Exp3 for oblivious adversarial bandits with rewards in [0, 1].

Book reference: Chapter 11, Algorithm 9.
Source verification: verified against online book PDF (accessed this session).

Reward/loss conversion (book): losses y = 1 - x.
Update uses the loss-based importance-weighted estimator rewritten as rewards:
  Ŝ_ti = Ŝ_{t-1,i} + 1 - I{At=i}(1-Xt)/P_ti
which equals adding the estimator X̂_ti from Eq. (11.6).

Requires bounded rewards in [0, 1]. Must NOT be applied to unbounded Gaussians
while claiming the book's bounded-loss guarantee.
Default η = sqrt(log(K)/(n K)) from Theorem 11.1.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from banditbook.utils.validation import check_positive_int

from .base import Agent, Feedback


class Exp3Agent(Agent):
    name = "exp3"
    source = "Book Ch. 11, Algorithm 9 (exact)"

    def __init__(
        self,
        n_arms: int,
        horizon: int,
        rng: np.random.Generator | None = None,
        *,
        eta: float | None = None,
    ):
        super().__init__(n_arms, rng)
        self.horizon = check_positive_int("horizon", horizon)
        if eta is None:
            # Theorem 11.1: η = sqrt(log(k)/(n k))
            self.eta = float(np.sqrt(np.log(self.n_arms) / (self.horizon * self.n_arms)))
        else:
            if eta <= 0:
                raise ValueError("eta must be > 0")
            self.eta = float(eta)
        self.S = np.zeros(self.n_arms, dtype=float)
        self._last_P: np.ndarray | None = None
        self._last_action: int | None = None

    def _probs(self) -> np.ndarray:
        # Stable log-weight normalisation: softmax(η S)
        logits = self.eta * self.S
        logits -= logits.max()
        w = np.exp(logits)
        P = w / w.sum()
        return P

    def select_action(self, context: Any = None) -> int:
        P = self._probs()
        self._last_P = P
        a = int(self.rng.choice(self.n_arms, p=P))
        self._last_action = a
        return a

    def update(self, action: int, feedback: Feedback) -> None:
        r = float(feedback.reward)
        if r < 0.0 or r > 1.0:
            raise ValueError(
                f"Exp3 requires rewards in [0, 1]; got {r}. "
                "Do not apply Exp3 to unbounded Gaussian rewards under the book guarantee."
            )
        if self._last_P is None:
            raise RuntimeError("select_action must be called before update")
        P = self._last_P
        # Algorithm 9 line 6: Ŝ_ti = Ŝ_{t-1,i} + 1 - I{At=i}(1-Xt)/P_ti
        # Equivalent: add 1 to every arm, then subtract (1-X)/P for chosen arm.
        # Numerically: X̂_i = 1 - I_i (1-X)/P_i  (Eq. 11.6)
        X_hat = np.ones(self.n_arms, dtype=float)
        X_hat[action] = 1.0 - (1.0 - r) / P[action]
        self.S += X_hat
        self.t += 1
        self._last_P = None
        self._last_action = None

    def _reset_state(self) -> None:
        self.S[:] = 0.0
        self._last_P = None
        self._last_action = None
