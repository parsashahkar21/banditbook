"""UCB family for 1-subgaussian / σ-subgaussian finite-armed bandits.

Book references:
- Algorithm 3 (Ch. 7): UCB(δ) with bonus sqrt(2 log(1/δ) / T_i) for 1-subgaussian.
- Algorithm 6 (Ch. 8): asymptotically optimal UCB with f(t)=1+t log^2(t).
- TunableUCBAgent: educational configurable form; NOT the book's asymptotic variant.

Noise-scale convention: for σ-subgaussian rewards, multiply the bonus by σ
(Exercise 7.2). Gaussian N(μ, σ²) is σ-subgaussian. Default σ=1 matches the
book's 1-subgaussian analyses.

Source verification: verified against online book PDF (accessed this session).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from banditbook.utils.validation import check_positive_int

from .base import Agent, Feedback


def _f_asymptotic(t: int) -> float:
    """f(t) = 1 + t log^2(t) from Algorithm 6. For t=1, log(1)=0 so f=1."""
    if t <= 1:
        return 1.0
    log_t = np.log(t)
    return 1.0 + t * (log_t**2)


class _MeanTrackerMixin:
    def _init_means(self) -> None:
        self.counts = np.zeros(self.n_arms, dtype=int)
        self.sums = np.zeros(self.n_arms, dtype=float)

    def _update_means(self, action: int, reward: float) -> None:
        self.counts[action] += 1
        self.sums[action] += reward
        self.t += 1

    def _means(self) -> np.ndarray:
        out = np.zeros(self.n_arms, dtype=float)
        mask = self.counts > 0
        out[mask] = self.sums[mask] / self.counts[mask]
        return out


class UCBDeltaAgent(_MeanTrackerMixin, Agent):
    """UCB(δ) — Book Algorithm 3 (Chapter 7). Exact for 1-subgaussian with σ=1.

    Index: ∞ if T_i=0 else μ̂_i + σ * sqrt(2 log(1/δ) / T_i).
    Typical book choice: δ = 1/n² (requires horizon).
    """

    name = "ucb_delta"
    source = "Book Ch. 7, Algorithm 3 (exact; σ-scale extension documented)"

    def __init__(
        self,
        n_arms: int,
        delta: float,
        rng: np.random.Generator | None = None,
        *,
        sigma: float = 1.0,
    ):
        super().__init__(n_arms, rng)
        if not 0.0 < delta < 1.0:
            raise ValueError(f"delta must be in (0, 1), got {delta}")
        if sigma <= 0:
            raise ValueError("sigma must be > 0")
        self.delta = float(delta)
        self.sigma = float(sigma)
        self._init_means()

    def select_action(self, context: Any = None) -> int:
        # Unpulled arms have index +∞; pull them first (smallest index on ties).
        if np.any(self.counts == 0):
            return int(np.argmin(self.counts))
        bonus = self.sigma * np.sqrt(2.0 * np.log(1.0 / self.delta) / self.counts)
        index = self._means() + bonus
        return int(np.argmax(index))

    def update(self, action: int, feedback: Feedback) -> None:
        self._update_means(action, float(feedback.reward))

    def _reset_state(self) -> None:
        self._init_means()


class AsymptoticallyOptimalUCB(_MeanTrackerMixin, Agent):
    """Asymptotically optimal UCB — Book Algorithm 6 (Chapter 8). Exact.

    At = argmax_i μ̂_i(t-1) + σ * sqrt(2 log f(t) / T_i(t-1)),
    f(t) = 1 + t log²(t). Choose each arm once first.
    """

    name = "ucb_asymptotic"
    source = "Book Ch. 8, Algorithm 6 (exact; σ-scale extension documented)"

    def __init__(
        self,
        n_arms: int,
        rng: np.random.Generator | None = None,
        *,
        sigma: float = 1.0,
    ):
        super().__init__(n_arms, rng)
        if sigma <= 0:
            raise ValueError("sigma must be > 0")
        self.sigma = float(sigma)
        self._init_means()

    def select_action(self, context: Any = None) -> int:
        if np.any(self.counts == 0):
            return int(np.argmin(self.counts))
        t = int(self.counts.sum()) + 1  # upcoming round index (1-based)
        bonus = self.sigma * np.sqrt(2.0 * np.log(_f_asymptotic(t)) / self.counts)
        return int(np.argmax(self._means() + bonus))

    def update(self, action: int, feedback: Feedback) -> None:
        self._update_means(action, float(feedback.reward))

    def _reset_state(self) -> None:
        self._init_means()


class TunableUCBAgent(_MeanTrackerMixin, Agent):
    """Configurable UCB: μ̂ + c * σ * sqrt(log(t) / T_i).

    Educational / sensitivity-analysis tool. This is NOT Algorithm 6 and must
    not be labelled as the book's asymptotically optimal UCB.
    """

    name = "ucb_tunable"
    source = "Educational configurable UCB (NOT Book Algorithm 6)"

    def __init__(
        self,
        n_arms: int,
        c: float = 2.0,
        rng: np.random.Generator | None = None,
        *,
        sigma: float = 1.0,
    ):
        super().__init__(n_arms, rng)
        if c < 0:
            raise ValueError("c must be >= 0")
        if sigma <= 0:
            raise ValueError("sigma must be > 0")
        self.c = float(c)
        self.sigma = float(sigma)
        self._init_means()

    def select_action(self, context: Any = None) -> int:
        if np.any(self.counts == 0):
            return int(np.argmin(self.counts))
        t = max(1, int(self.counts.sum()))
        bonus = self.c * self.sigma * np.sqrt(np.log(t) / self.counts)
        return int(np.argmax(self._means() + bonus))

    def update(self, action: int, feedback: Feedback) -> None:
        self._update_means(action, float(feedback.reward))

    def _reset_state(self) -> None:
        self._init_means()


# Preserve the historical class name from the original repo as an alias of the tunable form.
UCBAgent = TunableUCBAgent
