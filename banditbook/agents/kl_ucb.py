"""Bernoulli KL-UCB.

Book reference: Chapter 10, Algorithm 8.
Source verification: verified against online book PDF (accessed this session).

Requires Bernoulli (or [0,1]-Bernoulli-mean) rewards. Index solved by bisection
(numerical approximation with documented tolerance).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from banditbook.utils.kl import kl_ucb_index
from banditbook.utils.validation import check_arm

from .base import Agent, Feedback
from .ucb import _MeanTrackerMixin, _f_asymptotic


class KLUCBAgent(_MeanTrackerMixin, Agent):
    name = "kl_ucb"
    source = "Book Ch. 10, Algorithm 8 (exact index definition; bisection solver)"

    def __init__(
        self,
        n_arms: int,
        rng: np.random.Generator | None = None,
        *,
        tol: float = 1e-8,
    ):
        super().__init__(n_arms, rng)
        self.tol = float(tol)
        self._init_means()

    def select_action(self, context: Any = None) -> int:
        if np.any(self.counts == 0):
            return int(np.argmin(self.counts))
        t = int(self.counts.sum()) + 1
        thr_base = np.log(_f_asymptotic(t))
        means = self._means()
        # Clip empirical means to [0,1] for Bernoulli KL (rewards should be in [0,1])
        means = np.clip(means, 0.0, 1.0)
        indices = np.empty(self.n_arms, dtype=float)
        for i in range(self.n_arms):
            indices[i] = kl_ucb_index(means[i], int(self.counts[i]), thr_base / self.counts[i], tol=self.tol)
        return int(np.argmax(indices))

    def update(self, action: int, feedback: Feedback) -> None:
        r = float(feedback.reward)
        if r < 0.0 or r > 1.0:
            raise ValueError(
                f"KL-UCB requires rewards in [0, 1] (Bernoulli setting); got {r}. "
                "Do not apply KL-UCB to unbounded Gaussian rewards."
            )
        check_arm(action, self.n_arms)
        self._update_means(action, r)

    def _reset_state(self) -> None:
        self._init_means()
