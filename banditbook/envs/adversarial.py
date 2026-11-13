"""Adversarial bandit environments.

Standard protocol: the reward vector for round t is constructed before the
learner's action is revealed. Oblivious sequences may be pre-generated;
history-dependent adversaries may use previous rounds only.

Current-action-aware adversaries (omniscient / count-based counterexamples)
are explicitly labeled and sit outside the standard Exp3 guarantees.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from banditbook.utils.validation import check_arm, check_positive_int

from .base import BanditEnvironment, EvaluatorInfo, StepResult


class ObliviousAdversarialBandit(BanditEnvironment):
    """Oblivious adversarial bandit from a fixed reward table of shape (T, K).

    The table is fixed at construction / reset and does not depend on actions.
    For fair algorithm comparisons, share the same table across agents within
    a repetition.
    """

    def __init__(
        self,
        reward_table: np.ndarray,
        rng: np.random.Generator | None = None,
    ):
        table = np.asarray(reward_table, dtype=float)
        if table.ndim != 2:
            raise ValueError("reward_table must be 2-D (T, K)")
        if table.size == 0:
            raise ValueError("reward_table must be non-empty")
        if np.any(table < 0) or np.any(table > 1):
            raise ValueError(
                "Standard adversarial experiments require rewards in [0, 1]; "
                f"got range [{table.min()}, {table.max()}]"
            )
        super().__init__(table.shape[1], rng)
        self._table = table
        self.horizon = table.shape[0]
        self._pending: Optional[np.ndarray] = None

    @property
    def reward_table(self) -> np.ndarray:
        return self._table.copy()

    def _reset_env(self) -> None:
        self._pending = None

    def prepare_round(self) -> None:
        if self.t >= self.horizon:
            raise RuntimeError(f"horizon {self.horizon} exhausted")
        self._pending = self._table[self.t]
        return None

    def step(self, action: int) -> StepResult:
        action = check_arm(action, self.n_arms)
        if self._pending is None:
            self.prepare_round()
        rewards = self._pending
        assert rewards is not None
        reward = float(rewards[action])
        self._pending = None
        self._record(action, reward)
        return StepResult(
            reward=reward,
            evaluator=EvaluatorInfo(
                reward_vector=rewards.copy(),
                true_means=rewards.copy(),  # oblivious: realised = expected
                optimal_action=int(np.argmax(rewards)),
                optimal_mean=float(np.max(rewards)),
            ),
        )


class SwitchingObliviousBandit(ObliviousAdversarialBandit):
    """Deterministic switching sequence: one arm is 1 for ``switch_interval`` rounds.

    Nonstationary: use dynamic regret separately from fixed-arm external regret.
    """

    def __init__(
        self,
        n_arms: int,
        horizon: int,
        switch_interval: int = 100,
        rng: np.random.Generator | None = None,
    ):
        n_arms = check_positive_int("n_arms", n_arms)
        horizon = check_positive_int("horizon", horizon)
        switch_interval = check_positive_int("switch_interval", switch_interval)
        table = np.zeros((horizon, n_arms), dtype=float)
        for t in range(horizon):
            best = (t // switch_interval) % n_arms
            table[t, best] = 1.0
        super().__init__(table, rng)
        self.switch_interval = switch_interval
        # Time-dependent optimal means for dynamic regret
        self.dynamic_means = table.copy()


def make_oblivious_bernoulli_table(
    means: np.ndarray,
    horizon: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample an oblivious Bernoulli reward table with given per-arm means."""
    means = np.asarray(means, dtype=float)
    return (rng.random((horizon, len(means))) < means).astype(float)


class OmniscientAdversaryBandit(BanditEnvironment):
    """COUNTEREXAMPLE: action-aware adversary that sets chosen arm reward to 0.

    Outside standard adversarial guarantees. Retained for educational contrast
    with oblivious / history-only adversaries.
    """

    name = "omniscient_counterexample"
    label = "counterexample: current-action-aware (not standard protocol)"

    def __init__(self, n_arms: int, rng: np.random.Generator | None = None):
        super().__init__(n_arms, rng)

    def _reset_env(self) -> None:
        pass

    def step(self, action: int) -> StepResult:
        action = check_arm(action, self.n_arms)
        rewards = np.ones(self.n_arms, dtype=float)
        rewards[action] = 0.0  # depends on current action — counterexample
        reward = float(rewards[action])
        self._record(action, reward)
        return StepResult(
            reward=reward,
            evaluator=EvaluatorInfo(
                reward_vector=rewards,
                optimal_action=int(np.argmax(rewards)),
                optimal_mean=1.0,
                extras={"counterexample": True, "action_aware": True},
            ),
        )


class CountBasedAdversaryBandit(BanditEnvironment):
    """COUNTEREXAMPLE / history-dependent: punishes the most-played arm so far.

    Uses previous rounds' counts only when building the vector *before* the
    current action is applied — so this variant is history-dependent but not
    current-action-aware. Still labeled as outside standard oblivious guarantees.
    """

    name = "count_based_reactive"
    label = "reactive history-dependent (not oblivious)"

    def __init__(self, n_arms: int, rng: np.random.Generator | None = None):
        super().__init__(n_arms, rng)
        self._counts = np.zeros(n_arms, dtype=int)
        self._pending: Optional[np.ndarray] = None

    def _reset_env(self) -> None:
        self._counts[:] = 0
        self._pending = None

    def prepare_round(self) -> None:
        rewards = np.ones(self.n_arms, dtype=float)
        # Use previous counts only (before current action)
        if self._counts.sum() > 0:
            rewards[int(np.argmax(self._counts))] = 0.0
        self._pending = rewards
        return None

    def step(self, action: int) -> StepResult:
        action = check_arm(action, self.n_arms)
        if self._pending is None:
            self.prepare_round()
        rewards = self._pending
        assert rewards is not None
        reward = float(rewards[action])
        self._pending = None
        self._counts[action] += 1
        self._record(action, reward)
        return StepResult(
            reward=reward,
            evaluator=EvaluatorInfo(
                reward_vector=rewards.copy(),
                optimal_action=int(np.argmax(rewards)),
                optimal_mean=float(np.max(rewards)),
                extras={"reactive": True},
            ),
        )


# Legacy-compatible wrappers matching the original Omniscient/Switching/Minimizing APIs
class OmniscientAdversary:
    """Legacy strategy object — prefer OmniscientAdversaryBandit."""

    def get_reward(self, arm_index, history, round_num, num_arms):
        rewards = np.ones(num_arms)
        rewards[arm_index] = 0
        return rewards


class SwitchingAdversary:
    def __init__(self, switch_interval=100):
        self.switch_interval = switch_interval

    def get_reward(self, arm_index, history, round_num, num_arms):
        best_arm = (round_num // self.switch_interval) % num_arms
        rewards = np.zeros(num_arms)
        rewards[best_arm] = 1
        return rewards


class MinimizingAdversary:
    def get_reward(self, arm_index, history, round_num, num_arms):
        counts = np.zeros(num_arms)
        for h_arm, _ in history:
            counts[h_arm] += 1
        rewards = np.ones(num_arms)
        rewards[np.argmax(counts)] = 0
        return rewards
