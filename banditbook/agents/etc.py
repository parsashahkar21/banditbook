"""Explore-Then-Commit (ETC).

Book reference: Chapter 6, Algorithm 1.
Source verification: verified against online book PDF (accessed this session).

Exact implementation of the book's round-robin explore then commit rule,
with an explicit educational policy for horizons shorter than K*m.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from banditbook.utils.validation import check_positive_int

from .base import Agent, Feedback


class ETCAgent(Agent):
    """Explore-then-commit (Algorithm 1).

    For t = 1..Km (1-indexed in the book): play arm ((t-1) mod K).
    After Km pulls: commit to argmax of empirical means from the exploration
    phase (ties broken by smallest index).

    Short-horizon policy (T < K*m or commitment never reached):
        Continue the round-robin exploration schedule for the whole horizon.
        No commitment occurs. Documented as an educational edge-case policy;
        the book's Theorem 6.1 assumes 1 ≤ m ≤ n/k.
    """

    name = "etc"
    source = "Book Ch. 6, Algorithm 1 (exact)"

    def __init__(
        self,
        n_arms: int,
        m: int,
        rng: np.random.Generator | None = None,
        *,
        horizon: int | None = None,
    ):
        super().__init__(n_arms, rng)
        self.m = check_positive_int("m", m)
        self.horizon = horizon
        self.counts = np.zeros(n_arms, dtype=int)
        self.sums = np.zeros(n_arms, dtype=float)
        self.best_arm: int | None = None
        self._explore_budget = self.n_arms * self.m

    def select_action(self, context: Any = None) -> int:
        # Book uses 1-indexed t; we use number of completed pulls as t_next-1.
        t_next = int(self.counts.sum()) + 1
        if t_next <= self._explore_budget:
            return int((t_next - 1) % self.n_arms)
        if self.best_arm is None:
            means = np.divide(
                self.sums,
                self.counts,
                out=np.full(self.n_arms, -np.inf),
                where=self.counts > 0,
            )
            self.best_arm = int(np.argmax(means))
        return self.best_arm

    def update(self, action: int, feedback: Feedback) -> None:
        self.counts[action] += 1
        self.sums[action] += float(feedback.reward)
        self.t += 1

    def _reset_state(self) -> None:
        self.counts[:] = 0
        self.sums[:] = 0.0
        self.best_arm = None
