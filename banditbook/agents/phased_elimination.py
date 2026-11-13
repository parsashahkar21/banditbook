"""Phased elimination with G-optimal exploration (finite actions).

Book reference: Chapter 22, Algorithm 12.
Source verification: verified against online book PDF (accessed this session).

Restriction: fixed finite action set. Approximate G-optimal design via
Frank–Wolfe; integer pull counts via largest-remainder rounding.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from banditbook.utils.design import g_optimal_design, round_design_to_counts
from banditbook.utils.validation import check_positive_int

from .base import ContextualAgent, Feedback


class PhasedEliminationAgent(ContextualAgent):
    name = "phased_elimination"
    source = "Book Ch. 22, Algorithm 12 (finite actions; approx G-design)"

    def __init__(
        self,
        n_arms: int,
        dim: int,
        rng: np.random.Generator | None = None,
        *,
        delta: float = 0.1,
        fixed_actions: np.ndarray | None = None,
        max_phases: int = 20,
    ):
        super().__init__(n_arms, rng)
        self.dim = check_positive_int("dim", dim)
        if not 0 < delta < 1:
            raise ValueError("delta must be in (0, 1)")
        self.delta = float(delta)
        self.max_phases = int(max_phases)
        self.fixed_actions = None if fixed_actions is None else np.asarray(fixed_actions, dtype=float)
        self.active: list[int] = list(range(n_arms))
        self.phase = 1
        self._schedule: list[int] = []
        self._phase_actions: list[int] = []
        self._phase_rewards: list[float] = []
        self._T_counts: np.ndarray | None = None
        self.last_design_gap: float | None = None

    def _actions_matrix(self, context: Any) -> np.ndarray:
        if isinstance(context, dict) and "actions" in context:
            return np.asarray(context["actions"], dtype=float)
        if self.fixed_actions is not None:
            return self.fixed_actions
        raise ValueError("PhasedElimination requires action features")

    def _build_phase(self, actions: np.ndarray) -> None:
        A_active = actions[self.active]
        design = g_optimal_design(A_active)
        self.last_design_gap = design.optimality_gap
        ell = self.phase
        eps = 2.0 ** (-ell)
        k_ell = max(1, len(self.active))
        # T_ℓ(a) = ceil( 2 d π(a) / ε² * log(k_ℓ (ℓ+1) / δ) )  — book uses exact formula;
        # we use rounding of the real-valued allocation with a minimum of 1 for support.
        log_term = np.log(k_ell * (ell + 1) / self.delta)
        raw = 2.0 * self.dim * design.weights / (eps**2) * log_term
        # Convert to integer counts: round each up at least for positive weights
        counts_active = np.maximum(1, np.ceil(raw).astype(int))
        # Zero-out near-zero weights for sparsity (support <= d(d+1)/2 ideally)
        counts_active[design.weights < 1e-12] = 0
        if counts_active.sum() == 0:
            counts_active[:] = 1
        self._T_counts = counts_active
        schedule: list[int] = []
        for local_i, global_i in enumerate(self.active):
            schedule.extend([global_i] * int(counts_active[local_i]))
        self.rng.shuffle(schedule)
        self._schedule = schedule
        self._phase_actions = []
        self._phase_rewards = []

    def _select_with_context(self, context: Any) -> int:
        actions = self._actions_matrix(context)
        if not self._schedule:
            if len(self.active) <= 1 or self.phase > self.max_phases:
                return int(self.active[0])
            self._build_phase(actions)
        if not self._schedule:
            return int(self.active[0])
        return int(self._schedule.pop(0))

    def update(self, action: int, feedback: Feedback) -> None:
        self._phase_actions.append(int(action))
        self._phase_rewards.append(float(feedback.reward))
        self.t += 1
        # End of phase when schedule emptied and we have samples
        if not self._schedule and self._phase_actions and len(self.active) > 1:
            self._eliminate(feedback)

    def _eliminate(self, feedback: Feedback) -> None:
        # Reconstruct context actions from fixed or last feedback
        if self.fixed_actions is not None:
            actions = self.fixed_actions
        elif feedback.context is not None and isinstance(feedback.context, dict):
            actions = np.asarray(feedback.context["actions"], dtype=float)
        else:
            return
        A_active = actions[self.active]
        # V = Σ T(a) a a^T ; θ̂ = V^{-1} Σ T-weighted... actually book: sum over plays
        V = np.zeros((self.dim, self.dim))
        b = np.zeros(self.dim)
        for a_idx, r in zip(self._phase_actions, self._phase_rewards):
            a = actions[a_idx]
            V += np.outer(a, a)
            b += r * a
        try:
            theta = np.linalg.solve(V + 1e-8 * np.eye(self.dim), b)
        except np.linalg.LinAlgError:
            theta = np.linalg.pinv(V) @ b
        eps = 2.0 ** (-self.phase)
        scores = A_active @ theta
        best = float(np.max(scores))
        keep = [self.active[i] for i, s in enumerate(scores) if best - float(s) <= 2.0 * eps]
        if not keep:
            keep = [self.active[int(np.argmax(scores))]]
        self.active = keep
        self.phase += 1
        self._phase_actions = []
        self._phase_rewards = []
        self._schedule = []

    def _reset_state(self) -> None:
        self.active = list(range(self.n_arms))
        self.phase = 1
        self._schedule = []
        self._phase_actions = []
        self._phase_rewards = []
        self.last_design_gap = None
