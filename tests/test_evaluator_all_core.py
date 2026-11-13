"""Every finite-armed core agent runs through the shared evaluator."""

import numpy as np

from banditbook.agents import (
    AsymptoticallyOptimalUCB,
    ETCAgent,
    Exp3Agent,
    Exp3IXAgent,
    KLUCBAgent,
    MOSSAgent,
    TunableUCBAgent,
    UCBDeltaAgent,
    UniformRandomAgent,
)
from banditbook.envs.stochastic import BernoulliBandit
from banditbook.evaluation.runner import run_episode
from banditbook.utils.rng import make_rng_pair


def test_all_core_agents_on_bernoulli():
    horizon = 60
    means = [0.2, 0.5, 0.7]
    K = len(means)
    makers = [
        lambda rng: UniformRandomAgent(K, rng),
        lambda rng: ETCAgent(K, m=5, rng=rng),
        lambda rng: UCBDeltaAgent(K, delta=1.0 / horizon**2, rng=rng),
        lambda rng: AsymptoticallyOptimalUCB(K, rng=rng),
        lambda rng: TunableUCBAgent(K, c=2.0, rng=rng),
        lambda rng: MOSSAgent(K, horizon=horizon, rng=rng),
        lambda rng: KLUCBAgent(K, rng=rng),
        lambda rng: Exp3Agent(K, horizon=horizon, rng=rng),
        lambda rng: Exp3IXAgent(K, horizon=horizon, rng=rng),
    ]
    for i, make in enumerate(makers):
        ar, er, _ = make_rng_pair(10 + i)
        env = BernoulliBandit(means, rng=er)
        agent = make(ar)
        log = run_episode(agent, env, horizon)
        assert len(log.actions) == horizon
        assert log.cumulative_pseudo_regret is not None
        assert log.pull_counts.sum() == horizon
