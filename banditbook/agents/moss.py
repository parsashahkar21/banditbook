"""MOSS — minimax optimal strategy in the stochastic case.

Book reference: Chapter 9, Algorithm 7.
Source verification: verified against online book PDF (accessed this session).

Index: μ̂_i + σ * sqrt( (4 / T_i) * log⁺(n / (k T_i)) ), log⁺(x)=log(max{1,x}).
Requires horizon n. Exact for 1-subgaussian with σ=1; σ-scale documented.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from banditbook.utils.validation import check_positive_int

from .base import Agent, Feedback
from .ucb import _MeanTrackerMixin


class MOSSAgent(_MeanTrackerMixin, Agent):
    name = "moss"
    source = "Book Ch. 9, Algorithm 7 (exact; σ-scale extension documented)"

    def __init__(
        self,
        n_arms: int,
        horizon: int,
        rng: np.random.Generator | None = None,
        *,
        sigma: float = 1.0,
    ):
        super().__init__(n_arms, rng)
        self.horizon = check_positive_int("horizon", horizon)
        if sigma <= 0:
            raise ValueError("sigma must be > 0")
        self.sigma = float(sigma)
        self._init_means()

    def select_action(self, context: Any = None) -> int:
        if np.any(self.counts == 0):
            return int(np.argmin(self.counts))
        # log⁺(n / (k T_i))
        ratio = self.horizon / (self.n_arms * self.counts.astype(float))
        log_plus = np.log(np.maximum(1.0, ratio))
        bonus = self.sigma * np.sqrt((4.0 / self.counts) * log_plus)
        return int(np.argmax(self._means() + bonus))

    def update(self, action: int, feedback: Feedback) -> None:
        self._update_means(action, float(feedback.reward))

    def _reset_state(self) -> None:
        self._init_means()
