"""Environment interfaces with learner/evaluator information separation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import numpy as np

from banditbook.utils.validation import check_positive_int


@dataclass
class EvaluatorInfo:
    """Information available only to the evaluator — never passed to agents."""

    true_means: Optional[np.ndarray] = None  # per-arm expected rewards this round
    reward_vector: Optional[np.ndarray] = None  # full reward vector (adversarial)
    optimal_action: Optional[int] = None
    optimal_mean: Optional[float] = None
    extras: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class StepResult:
    """Outcome of one interaction round."""

    reward: float  # learner-visible
    context: Any = None  # learner-visible public context
    action_features: Optional[np.ndarray] = None
    evaluator: EvaluatorInfo = field(default_factory=EvaluatorInfo)


class BanditEnvironment(ABC):
    """Base bandit environment.

    Standard adversarial protocol: the current reward/loss vector is fixed
    before the sampled action is revealed (via ``prepare_round`` / internal
    state). History-dependent adversaries may use previous rounds only.
    """

    def __init__(self, n_arms: int, rng: np.random.Generator | None = None):
        self.n_arms = check_positive_int("n_arms", n_arms)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.t = 0
        self.history: list[tuple[int, float]] = []
        self.record_history = True

    def reset(self, rng: np.random.Generator | None = None) -> None:
        if rng is not None:
            self.rng = rng
        self.t = 0
        self.history = []
        self._reset_env()

    @abstractmethod
    def _reset_env(self) -> None:
        ...

    def prepare_round(self) -> Any:
        """Optional: reveal public context and fix this round's latent rewards.

        Returns learner-visible context (or None).
        """
        return None

    @abstractmethod
    def step(self, action: int) -> StepResult:
        """Execute action and return learner reward + evaluator-only info."""

    def pull(self, action: int) -> float:
        """Back-compat: return only the learner-visible reward."""
        return self.step(action).reward

    def _record(self, action: int, reward: float) -> None:
        if self.record_history:
            self.history.append((int(action), float(reward)))
        self.t += 1
