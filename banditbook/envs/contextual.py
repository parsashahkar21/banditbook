"""Finite-context bandit: one stochastic bandit per context.

Book reference: Chapter 18.1 (one bandit per context) — educational baseline.
"""

from __future__ import annotations

import numpy as np

from banditbook.utils.validation import check_arm, check_positive_int

from .base import BanditEnvironment, EvaluatorInfo, StepResult


class FiniteContextBandit(BanditEnvironment):
    """Contexts in {0,...,C-1}; each context has its own Bernoulli mean vector."""

    def __init__(
        self,
        mean_table: np.ndarray,
        context_probs: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ):
        table = np.asarray(mean_table, dtype=float)
        if table.ndim != 2:
            raise ValueError("mean_table must be (C, K)")
        if np.any(table < 0) or np.any(table > 1):
            raise ValueError("means must be in [0, 1] for Bernoulli contexts")
        n_contexts, n_arms = table.shape
        super().__init__(n_arms, rng)
        self.n_contexts = check_positive_int("n_contexts", n_contexts)
        self._means = table
        if context_probs is None:
            self._context_probs = np.full(n_contexts, 1.0 / n_contexts)
        else:
            p = np.asarray(context_probs, dtype=float)
            if p.shape != (n_contexts,) or p.sum() <= 0:
                raise ValueError("context_probs invalid")
            self._context_probs = p / p.sum()
        self._pending_context: int | None = None

    def _reset_env(self) -> None:
        self._pending_context = None

    def prepare_round(self) -> int:
        self._pending_context = int(self.rng.choice(self.n_contexts, p=self._context_probs))
        return self._pending_context

    def step(self, action: int) -> StepResult:
        action = check_arm(action, self.n_arms)
        if self._pending_context is None:
            self.prepare_round()
        ctx = int(self._pending_context)
        means = self._means[ctx]
        reward = float(self.rng.random() < means[action])
        self._pending_context = None
        self._record(action, reward)
        return StepResult(
            reward=reward,
            context=ctx,
            evaluator=EvaluatorInfo(
                true_means=means.copy(),
                optimal_action=int(np.argmax(means)),
                optimal_mean=float(np.max(means)),
                extras={"context": ctx},
            ),
        )
