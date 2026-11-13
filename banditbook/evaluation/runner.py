"""Episode and multi-run experiment evaluation.

Agents receive only learner-visible Feedback. EvaluatorInfo never reaches agents.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from banditbook.agents.base import Agent, Feedback
from banditbook.envs.base import BanditEnvironment
from banditbook.metrics.regret import (
    adversarial_external_regret,
    dynamic_regret,
    pathwise_pseudo_regret,
    reward_shortfall,
)
from banditbook.utils.rng import make_rng_pair, spawn_run_seeds


@dataclass
class EpisodeLog:
    actions: np.ndarray
    rewards: np.ndarray
    contexts: list
    reward_vectors: Optional[np.ndarray]
    true_means_path: Optional[np.ndarray]
    cumulative_pseudo_regret: Optional[np.ndarray]
    reward_shortfall: Optional[float]
    external_regret: Optional[float]
    dynamic_regret: Optional[float]
    pull_counts: np.ndarray
    cumulative_reward: float
    runtime_sec: float
    extras: dict = field(default_factory=dict)


@dataclass
class ExperimentResult:
    config: dict
    seeds: list[int]
    regrets: np.ndarray  # (n_runs, T) primary metric curve
    final_regrets: np.ndarray
    pull_counts: np.ndarray  # (n_runs, K)
    cumulative_rewards: np.ndarray
    runtimes: np.ndarray
    metric_name: str
    extras: dict = field(default_factory=dict)

    @property
    def mean_regret_curve(self) -> np.ndarray:
        return self.regrets.mean(axis=0)

    @property
    def stderr_regret_curve(self) -> np.ndarray:
        n = self.regrets.shape[0]
        if n <= 1:
            return np.zeros(self.regrets.shape[1])
        return self.regrets.std(axis=0, ddof=1) / np.sqrt(n)


def run_episode(
    agent: Agent,
    env: BanditEnvironment,
    horizon: int,
    *,
    record_reward_vectors: bool = False,
) -> EpisodeLog:
    """Run one episode of length ``horizon``.

    Protocol each round:
      1. context = env.prepare_round()  # may fix reward vector before action
      2. action = agent.select_action(context)
      3. step = env.step(action)
      4. agent.update(action, Feedback(...))  # no evaluator fields
    """
    t0 = time.perf_counter()
    actions = np.empty(horizon, dtype=int)
    rewards = np.empty(horizon, dtype=float)
    contexts: list = []
    means_rows: list[np.ndarray] = []
    reward_rows: list[np.ndarray] = []
    has_means = False
    has_vectors = False

    for t in range(horizon):
        context = env.prepare_round()
        action = agent.select_action(context)
        step = env.step(action)
        agent.update(
            action,
            Feedback(
                reward=step.reward,
                context=step.context if step.context is not None else context,
                action_features=step.action_features,
                info={},  # never forward evaluator extras
            ),
        )
        actions[t] = action
        rewards[t] = step.reward
        contexts.append(context if context is not None else step.context)
        if step.evaluator.true_means is not None:
            means_rows.append(np.asarray(step.evaluator.true_means, dtype=float))
            has_means = True
        if step.evaluator.reward_vector is not None:
            reward_rows.append(np.asarray(step.evaluator.reward_vector, dtype=float))
            has_vectors = True

    runtime = time.perf_counter() - t0
    K = env.n_arms
    pull_counts = np.bincount(actions, minlength=K).astype(int)

    means_path = np.vstack(means_rows) if has_means else None
    reward_table = np.vstack(reward_rows) if has_vectors else None

    cum_pseudo = None
    shortfall = None
    ext = None
    dyn = None

    if means_path is not None:
        # Stationary if all rows equal
        if np.allclose(means_path, means_path[0]):
            cum_pseudo = pathwise_pseudo_regret(actions, means_path[0])
            shortfall = reward_shortfall(rewards, means_path[0])
        else:
            cum_pseudo = pathwise_pseudo_regret(actions, means_path)
            dyn = dynamic_regret(actions, means_path)

    if reward_table is not None:
        ext = adversarial_external_regret(actions, reward_table)
        if dyn is None:
            dyn = dynamic_regret(actions, reward_table)

    return EpisodeLog(
        actions=actions,
        rewards=rewards,
        contexts=contexts,
        reward_vectors=reward_table if record_reward_vectors else None,
        true_means_path=means_path,
        cumulative_pseudo_regret=cum_pseudo,
        reward_shortfall=shortfall,
        external_regret=ext,
        dynamic_regret=dyn,
        pull_counts=pull_counts,
        cumulative_reward=float(rewards.sum()),
        runtime_sec=runtime,
    )


def run_repetitions(
    agent_factory: Callable[[np.random.Generator, int], Agent],
    env_factory: Callable[[np.random.Generator, int], BanditEnvironment],
    *,
    horizon: int,
    n_runs: int,
    seed: int = 0,
    metric: str = "auto",
    shared_oblivious_table_factory: Callable[[np.random.Generator, int], np.ndarray] | None = None,
    checkpoint_every: int | None = None,
    on_checkpoint: Callable[[int, ExperimentResult], None] | None = None,
) -> ExperimentResult:
    """Run independent repetitions with separated agent/env RNG streams.

    If ``shared_oblivious_table_factory`` is provided, each run builds one reward
    table shared across... (caller should bake the table into env_factory via
    closure). Document coupling in experiment configs.

    metric: 'pseudo' | 'external' | 'dynamic' | 'auto'
    """
    seeds = spawn_run_seeds(seed, n_runs)
    regret_curves = []
    finals = []
    pulls = []
    cum_rewards = []
    runtimes = []
    metric_name = metric
    extras: dict[str, Any] = {"external": [], "dynamic": [], "shortfall": []}

    for r, run_seed in enumerate(seeds):
        agent_rng, env_rng, _ = make_rng_pair(run_seed)
        env = env_factory(env_rng, run_seed)
        agent = agent_factory(agent_rng, run_seed)
        agent.reset(agent_rng)
        env.reset(env_rng)
        log = run_episode(agent, env, horizon)

        if metric == "auto":
            if log.cumulative_pseudo_regret is not None and log.external_regret is None:
                curve = log.cumulative_pseudo_regret
                metric_name = "pathwise_pseudo_regret"
            elif log.external_regret is not None:
                # Build cumulative external regret curve via prefix
                if log.true_means_path is not None and log.reward_vectors is None:
                    # shouldn't happen often
                    curve = log.cumulative_pseudo_regret
                    metric_name = "pathwise_pseudo_regret"
                else:
                    # For adversarial, store final only in curve as constant fill unless vectors recorded
                    # Recompute cumulative external along prefixes if we have vectors in means path
                    table = log.true_means_path  # for oblivious, true_means == rewards
                    if table is not None:
                        curve = _cumulative_external(log.actions, table)
                        metric_name = "adversarial_external_regret"
                    else:
                        curve = log.cumulative_pseudo_regret
                        metric_name = "pathwise_pseudo_regret"
            else:
                curve = np.cumsum(np.zeros(horizon))
                metric_name = "none"
        elif metric == "pseudo":
            curve = log.cumulative_pseudo_regret
            metric_name = "pathwise_pseudo_regret"
        elif metric == "external":
            table = log.true_means_path
            curve = _cumulative_external(log.actions, table)
            metric_name = "adversarial_external_regret"
        elif metric == "dynamic":
            table = log.true_means_path
            curve = _cumulative_dynamic(log.actions, table)
            metric_name = "dynamic_regret"
        else:
            raise ValueError(f"unknown metric {metric}")

        if curve is None:
            curve = np.zeros(horizon)

        regret_curves.append(curve)
        finals.append(float(curve[-1]))
        pulls.append(log.pull_counts)
        cum_rewards.append(log.cumulative_reward)
        runtimes.append(log.runtime_sec)
        extras["external"].append(log.external_regret)
        extras["dynamic"].append(log.dynamic_regret)
        extras["shortfall"].append(log.reward_shortfall)

        if checkpoint_every and on_checkpoint and (r + 1) % checkpoint_every == 0:
            partial = ExperimentResult(
                config={"horizon": horizon, "n_runs": r + 1, "seed": seed},
                seeds=seeds[: r + 1],
                regrets=np.asarray(regret_curves),
                final_regrets=np.asarray(finals),
                pull_counts=np.asarray(pulls),
                cumulative_rewards=np.asarray(cum_rewards),
                runtimes=np.asarray(runtimes),
                metric_name=metric_name,
                extras=extras,
            )
            on_checkpoint(r + 1, partial)

    return ExperimentResult(
        config={
            "horizon": horizon,
            "n_runs": n_runs,
            "seed": seed,
            "metric": metric_name,
        },
        seeds=seeds,
        regrets=np.asarray(regret_curves),
        final_regrets=np.asarray(finals),
        pull_counts=np.asarray(pulls),
        cumulative_rewards=np.asarray(cum_rewards),
        runtimes=np.asarray(runtimes),
        metric_name=metric_name,
        extras=extras,
    )


def _cumulative_external(actions: np.ndarray, table: np.ndarray) -> np.ndarray:
    T, K = table.shape[0], table.shape[1]
    cum_arm = np.zeros(K)
    learner = 0.0
    out = np.empty(T)
    for t in range(T):
        cum_arm += table[t]
        learner += table[t, actions[t]]
        out[t] = np.max(cum_arm) - learner
    return out


def _cumulative_dynamic(actions: np.ndarray, table: np.ndarray) -> np.ndarray:
    inst = np.max(table, axis=1) - table[np.arange(len(actions)), actions]
    return np.cumsum(inst)
