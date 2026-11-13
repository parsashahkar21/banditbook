"""Stochastic linear bandit with hidden θ and public action features.

Book reference: Chapter 19.
Restriction: finite action set provided each round (or fixed). Continuous
action-set optimisation is not claimed.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from banditbook.utils.validation import check_arm

from .base import BanditEnvironment, EvaluatorInfo, StepResult


class StochasticLinearBandit(BanditEnvironment):
    """Xt = ⟨θ, a_t⟩ + η_t with η_t ~ N(0, σ²).

    ``actions`` is a fixed finite set of shape (K, d). Learner sees action
    features; θ is evaluator-only.
    """

    def __init__(
        self,
        theta: np.ndarray,
        actions: np.ndarray,
        noise_std: float = 1.0,
        rng: np.random.Generator | None = None,
    ):
        theta = np.asarray(theta, dtype=float).reshape(-1)
        actions = np.asarray(actions, dtype=float)
        if actions.ndim != 2:
            raise ValueError("actions must be (K, d)")
        if actions.shape[1] != theta.shape[0]:
            raise ValueError("actions dim must match theta")
        if noise_std < 0:
            raise ValueError("noise_std must be >= 0")
        super().__init__(actions.shape[0], rng)
        self._theta = theta  # hidden
        self.actions = actions  # public
        self.noise_std = float(noise_std)
        self._means = actions @ theta
        self._optimal = int(np.argmax(self._means))

    @property
    def dim(self) -> int:
        return int(self._theta.shape[0])

    def _reset_env(self) -> None:
        pass

    def prepare_round(self) -> dict:
        # Public: finite action feature matrix
        return {"actions": self.actions}

    def step(self, action: int) -> StepResult:
        action = check_arm(action, self.n_arms)
        a = self.actions[action]
        mean = float(np.dot(self._theta, a))
        noise = float(self.rng.normal(0.0, self.noise_std)) if self.noise_std > 0 else 0.0
        reward = mean + noise
        self._record(action, reward)
        return StepResult(
            reward=reward,
            context={"actions": self.actions},
            action_features=a.copy(),
            evaluator=EvaluatorInfo(
                true_means=self._means.copy(),
                optimal_action=self._optimal,
                optimal_mean=float(self._means[self._optimal]),
                extras={"theta": self._theta.copy()},  # evaluator-only
            ),
        )
