"""Stochastic finite-armed bandit environments.

Gaussian convention: ``stds`` are standard deviations (passed to Normal(mean, std)).
For subgaussian analyses in the book, a Gaussian N(μ, σ²) is σ-subgaussian.
Bernoulli means must lie in [0, 1].

Preserves distinct mean entries as provided (do not collapse ``0, 1`` into ``0.1``).
"""

from __future__ import annotations

import numpy as np

from banditbook.utils.validation import as_1d_float_array, check_arm

from .base import BanditEnvironment, EvaluatorInfo, StepResult


class BernoulliBandit(BanditEnvironment):
    def __init__(
        self,
        probabilities,
        rng: np.random.Generator | None = None,
    ):
        probs = as_1d_float_array("probabilities", probabilities)
        if np.any(probs < 0) or np.any(probs > 1):
            raise ValueError("Bernoulli probabilities must lie in [0, 1]")
        super().__init__(len(probs), rng)
        self._probabilities = probs
        self._optimal = int(np.argmax(probs))
        self._optimal_mean = float(probs[self._optimal])

    @property
    def means(self) -> np.ndarray:
        return self._probabilities.copy()

    def _reset_env(self) -> None:
        pass

    def step(self, action: int) -> StepResult:
        action = check_arm(action, self.n_arms)
        reward = float(self.rng.random() < self._probabilities[action])
        self._record(action, reward)
        return StepResult(
            reward=reward,
            evaluator=EvaluatorInfo(
                true_means=self._probabilities.copy(),
                optimal_action=self._optimal,
                optimal_mean=self._optimal_mean,
            ),
        )


class GaussianBandit(BanditEnvironment):
    """Gaussian rewards with means and standard deviations (not variances)."""

    def __init__(
        self,
        means,
        stds=None,
        rng: np.random.Generator | None = None,
    ):
        means_arr = as_1d_float_array("means", means)
        if stds is None:
            stds_arr = np.ones_like(means_arr)
        else:
            stds_arr = as_1d_float_array("stds", stds)
            if len(stds_arr) != len(means_arr):
                raise ValueError("means and stds must have equal length")
            if np.any(stds_arr <= 0):
                raise ValueError("stds must be > 0")
        super().__init__(len(means_arr), rng)
        self._means = means_arr
        self._stds = stds_arr
        self._optimal = int(np.argmax(means_arr))
        self._optimal_mean = float(means_arr[self._optimal])

    @property
    def means(self) -> np.ndarray:
        return self._means.copy()

    @property
    def stds(self) -> np.ndarray:
        return self._stds.copy()

    def _reset_env(self) -> None:
        pass

    def step(self, action: int) -> StepResult:
        action = check_arm(action, self.n_arms)
        reward = float(self.rng.normal(self._means[action], self._stds[action]))
        self._record(action, reward)
        return StepResult(
            reward=reward,
            evaluator=EvaluatorInfo(
                true_means=self._means.copy(),
                optimal_action=self._optimal,
                optimal_mean=self._optimal_mean,
            ),
        )
