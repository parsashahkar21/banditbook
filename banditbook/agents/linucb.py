"""LinUCB / OFUL for finite-action stochastic linear bandits.

Book reference: Chapter 19 (Eq. 19.2–19.8).
Source verification: verified against online book PDF (accessed this session).

Clearly specified variant: ellipsoid confidence set with β from Eq. (19.8),
finite action set maximisation only (not continuous action sets).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from banditbook.utils.linalg import RegularizedLeastSquares, beta_oful
from banditbook.utils.validation import check_positive_int

from .base import ContextualAgent, Feedback


class LinUCBAgent(ContextualAgent):
    name = "linucb"
    source = "Book Ch. 19 LinUCB/OFUL (finite-action maximisation; Eq. 19.8 β)"

    def __init__(
        self,
        n_arms: int,
        dim: int,
        horizon: int,
        rng: np.random.Generator | None = None,
        *,
        lam: float = 1.0,
        delta: float | None = None,
        L: float = 1.0,
        m2: float = 1.0,
        fixed_actions: np.ndarray | None = None,
    ):
        super().__init__(n_arms, rng)
        self.dim = check_positive_int("dim", dim)
        self.horizon = check_positive_int("horizon", horizon)
        self.lam = float(lam)
        self.L = float(L)
        self.m2 = float(m2)
        self.delta = 1.0 / horizon if delta is None else float(delta)
        self.rls = RegularizedLeastSquares(self.dim, self.lam)
        self.fixed_actions = None if fixed_actions is None else np.asarray(fixed_actions, dtype=float)
        self._beta_sqrt = beta_oful(self.delta, self.dim, self.lam, self.horizon, self.L, self.m2)
        self._last_action_vec: np.ndarray | None = None

    def _actions_from_context(self, context: Any) -> np.ndarray:
        if context is None and self.fixed_actions is not None:
            return self.fixed_actions
        if isinstance(context, dict) and "actions" in context:
            return np.asarray(context["actions"], dtype=float)
        if isinstance(context, np.ndarray):
            return np.asarray(context, dtype=float)
        if self.fixed_actions is not None:
            return self.fixed_actions
        raise ValueError("LinUCB requires action features in context or fixed_actions")

    def _select_with_context(self, context: Any) -> int:
        actions = self._actions_from_context(context)
        if actions.shape != (self.n_arms, self.dim):
            raise ValueError(f"expected actions shape {(self.n_arms, self.dim)}, got {actions.shape}")
        indices = np.array(
            [self.rls.ucb_linear(actions[i], self._beta_sqrt) for i in range(self.n_arms)]
        )
        a = int(np.argmax(indices))
        self._last_action_vec = actions[a].copy()
        return a

    def update(self, action: int, feedback: Feedback) -> None:
        if feedback.action_features is not None:
            a = np.asarray(feedback.action_features, dtype=float)
        elif self._last_action_vec is not None:
            a = self._last_action_vec
        else:
            raise RuntimeError("missing action features for LinUCB update")
        self.rls.update(a, float(feedback.reward))
        self.t += 1
        self._last_action_vec = None

    def _reset_state(self) -> None:
        self.rls = RegularizedLeastSquares(self.dim, self.lam)
        self._last_action_vec = None
