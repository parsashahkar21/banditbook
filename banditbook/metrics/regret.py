"""Regret metrics with distinct definitions (Milestone 2).

Do not clamp negative values. Do not replace adversarial external regret's
fixed-arm comparator with a per-round maximum (that is dynamic regret).

Note on adaptive adversaries: E[max_i ...] vs max_i E[...] ordering matters;
a fixed-comparator expected-regret theorem does not automatically bound
expected realised external regret under adaptive adversaries.
"""

from __future__ import annotations

import numpy as np


def pathwise_pseudo_regret(actions: np.ndarray, means: np.ndarray) -> np.ndarray:
    """Stochastic pathwise pseudo-regret cumulative curve.

    sum_{s<=t} (max_i μ_i - μ[a_s]).
    Averaging independent runs estimates expected regret under the model.
    ``means`` may be 1-D (stationary) or 2-D (T, K) time-varying expected means.
    """
    actions = np.asarray(actions, dtype=int)
    means = np.asarray(means, dtype=float)
    T = len(actions)
    if means.ndim == 1:
        opt = float(np.max(means))
        inst = opt - means[actions]
    else:
        if means.shape[0] != T:
            raise ValueError("time-varying means must have shape (T, K)")
        opt = np.max(means, axis=1)
        inst = opt - means[np.arange(T), actions]
    return np.cumsum(inst)


def reward_shortfall(rewards: np.ndarray, means: np.ndarray) -> float:
    """Optional stochastic reward-based shortfall: T * max μ - sum observed rewards.

    Labelled separately from pseudo-regret. Negative values are retained.
    """
    rewards = np.asarray(rewards, dtype=float)
    means = np.asarray(means, dtype=float)
    if means.ndim != 1:
        raise ValueError("reward_shortfall expects stationary 1-D means")
    T = len(rewards)
    return float(T * np.max(means) - np.sum(rewards))


def adversarial_external_regret(
    actions: np.ndarray,
    reward_table: np.ndarray,
) -> float:
    """Realised external regret in reward notation:

    max_i sum_t reward[t, i] - sum_t reward[t, action_t].

    Maintains cumulative reward per arm; comparator is a single fixed arm,
    not the per-round maximum. Negative values are not clamped.
    """
    actions = np.asarray(actions, dtype=int)
    table = np.asarray(reward_table, dtype=float)
    T = len(actions)
    if table.shape[0] < T:
        raise ValueError("reward_table shorter than actions")
    table = table[:T]
    cum_per_arm = table.sum(axis=0)
    learner = float(table[np.arange(T), actions].sum())
    return float(np.max(cum_per_arm) - learner)


def dynamic_regret(
    actions: np.ndarray,
    means_or_rewards: np.ndarray,
) -> float:
    """Nonstationary dynamic regret vs time-dependent comparator.

    sum_t (max_i x[t,i] - x[t, a_t]) where x is expected means or realised rewards.
    Reported separately from external regret.
    """
    actions = np.asarray(actions, dtype=int)
    x = np.asarray(means_or_rewards, dtype=float)
    T = len(actions)
    if x.ndim != 2 or x.shape[0] < T:
        raise ValueError("dynamic_regret expects array of shape (T, K)")
    x = x[:T]
    return float(np.sum(np.max(x, axis=1) - x[np.arange(T), actions]))


def linear_pseudo_regret_increment(
    action_features: np.ndarray,
    theta: np.ndarray,
    available_actions: np.ndarray,
) -> float:
    """Linear/contextual pseudo-regret for one round:

    max_{a in A_t} ⟨θ, a⟩ - ⟨θ, a_t⟩.
    """
    theta = np.asarray(theta, dtype=float)
    A = np.asarray(available_actions, dtype=float)
    a_t = np.asarray(action_features, dtype=float)
    best = float(np.max(A @ theta))
    return best - float(np.dot(theta, a_t))
