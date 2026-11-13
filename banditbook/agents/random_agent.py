"""Uniform random baseline (educational / sanity check; not a book-named algorithm)."""

from __future__ import annotations

from typing import Any

from .base import Agent, Feedback


class UniformRandomAgent(Agent):
    """Play each arm with probability 1/K every round."""

    name = "uniform_random"
    source = "baseline (not a named book algorithm)"

    def select_action(self, context: Any = None) -> int:
        return int(self.rng.integers(0, self.n_arms))

    def update(self, action: int, feedback: Feedback) -> None:
        self.t += 1

    def _reset_state(self) -> None:
        pass
