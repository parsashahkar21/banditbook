"""Build agents and environments from configuration dictionaries."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from banditbook import agents as A
from banditbook import envs as E


AGENT_REGISTRY: dict[str, Callable[..., Any]] = {
    "uniform_random": A.UniformRandomAgent,
    "etc": A.ETCAgent,
    "ucb_delta": A.UCBDeltaAgent,
    "ucb_asymptotic": A.AsymptoticallyOptimalUCB,
    "ucb_tunable": A.TunableUCBAgent,
    "moss": A.MOSSAgent,
    "kl_ucb": A.KLUCBAgent,
    "exp3": A.Exp3Agent,
    "exp3ix": A.Exp3IXAgent,
    "exp4": A.Exp4Agent,
    "linucb": A.LinUCBAgent,
    "phased_elimination": A.PhasedEliminationAgent,
    "per_context_ucb": A.PerContextUCBAgent,
}


def build_agent(name: str, n_arms: int, rng: np.random.Generator, params: dict | None = None):
    params = dict(params or {})
    if name not in AGENT_REGISTRY:
        raise KeyError(f"Unknown agent {name!r}. Choose from {sorted(AGENT_REGISTRY)}")
    cls = AGENT_REGISTRY[name]
    return cls(n_arms=n_arms, rng=rng, **params)


def build_env(name: str, rng: np.random.Generator, params: dict | None = None):
    params = dict(params or {})
    if name == "bernoulli":
        return E.BernoulliBandit(params["probabilities"], rng=rng)
    if name == "gaussian":
        return E.GaussianBandit(params["means"], params.get("stds"), rng=rng)
    if name == "oblivious":
        return E.ObliviousAdversarialBandit(params["reward_table"], rng=rng)
    if name == "switching":
        return E.SwitchingObliviousBandit(
            params["n_arms"],
            params["horizon"],
            params.get("switch_interval", 100),
            rng=rng,
        )
    if name == "linear":
        return E.StochasticLinearBandit(
            params["theta"],
            params["actions"],
            params.get("noise_std", 1.0),
            rng=rng,
        )
    if name == "contextual":
        return E.FiniteContextBandit(
            params["mean_table"],
            params.get("context_probs"),
            rng=rng,
        )
    if name == "omniscient_counterexample":
        return E.OmniscientAdversaryBandit(params["n_arms"], rng=rng)
    raise KeyError(f"Unknown env {name!r}")
