"""Agent interfaces.

Learner-visible feedback must not include true means, hidden parameters,
unselected rewards, or future outcomes. Evaluator-only info stays outside agents.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import numpy as np

from banditbook.utils.validation import check_positive_int


@dataclass
class Feedback:
    """Learner-visible observation after an action.

    Attributes
    ----------
    reward :
        Observed scalar reward for the chosen action (bandit feedback).
    context :
        Optional public context / features available before or with the reward.
    action_features :
        Optional features of the chosen action (linear / combinatorial settings).
    info :
        Optional learner-visible side information (never evaluator secrets).
    """

    reward: float
    context: Any = None
    action_features: Optional[np.ndarray] = None
    info: Mapping[str, Any] = field(default_factory=dict)


class Agent(ABC):
    """Finite-armed (or discrete-action) bandit learner interface."""

    def __init__(self, n_arms: int, rng: np.random.Generator | None = None):
        self.n_arms = check_positive_int("n_arms", n_arms)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.t = 0  # number of updates completed

    @abstractmethod
    def select_action(self, context: Any = None) -> int:
        """Choose an action index in {0, ..., K-1}."""

    @abstractmethod
    def update(self, action: int, feedback: Feedback) -> None:
        """Update internal state from learner-visible feedback."""

    def reset(self, rng: np.random.Generator | None = None) -> None:
        """Reset learner state; optionally replace the RNG stream."""
        if rng is not None:
            self.rng = rng
        self.t = 0
        self._reset_state()

    @abstractmethod
    def _reset_state(self) -> None:
        ...

    # Back-compat aliases used in early code
    def select_arm(self, context: Any = None) -> int:
        return self.select_action(context)


class ContextualAgent(Agent):
    """Marker for agents that require a public context each round."""

    def select_action(self, context: Any = None) -> int:
        if context is None:
            raise ValueError(f"{type(self).__name__} requires a context")
        return self._select_with_context(context)

    @abstractmethod
    def _select_with_context(self, context: Any) -> int:
        ...
