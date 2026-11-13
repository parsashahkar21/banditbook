"""Per-context finite-arm baseline (Chapter 18.1 educational).

Runs an independent UCB instance per discrete context.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import ContextualAgent, Feedback
from .ucb import AsymptoticallyOptimalUCB


class PerContextUCBAgent(ContextualAgent):
    name = "per_context_ucb"
    source = "Book Ch. 18.1 idea (one bandit per context) + Algorithm 6 UCB"

    def __init__(
        self,
        n_arms: int,
        n_contexts: int,
        rng: np.random.Generator | None = None,
        *,
        sigma: float = 1.0,
    ):
        super().__init__(n_arms, rng)
        if n_contexts < 1:
            raise ValueError("n_contexts must be >= 1")
        self.n_contexts = int(n_contexts)
        self.sigma = float(sigma)
        self._agents = [
            AsymptoticallyOptimalUCB(n_arms, rng=rng, sigma=sigma) for _ in range(n_contexts)
        ]
        self._last_ctx: int | None = None

    def _select_with_context(self, context: Any) -> int:
        ctx = int(context)
        if ctx < 0 or ctx >= self.n_contexts:
            raise ValueError(f"context {ctx} out of range")
        self._last_ctx = ctx
        return self._agents[ctx].select_action()

    def update(self, action: int, feedback: Feedback) -> None:
        ctx = self._last_ctx
        if feedback.context is not None:
            ctx = int(feedback.context) if not isinstance(feedback.context, dict) else self._last_ctx
        if ctx is None:
            raise RuntimeError("missing context for update")
        self._agents[ctx].update(action, feedback)
        self.t += 1
        self._last_ctx = None

    def _reset_state(self) -> None:
        for a in self._agents:
            a.reset(self.rng)
        self._last_ctx = None
