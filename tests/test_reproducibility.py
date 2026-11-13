import numpy as np

from banditbook.agents.kl_ucb import KLUCBAgent
from banditbook.agents.ucb import AsymptoticallyOptimalUCB
from banditbook.envs.stochastic import BernoulliBandit
from banditbook.evaluation.runner import run_episode, run_repetitions
from banditbook.utils.rng import make_rng_pair


def test_identical_seeds_reproduce():
    def once(seed):
        ar, er, _ = make_rng_pair(seed)
        env = BernoulliBandit([0.2, 0.5, 0.8], rng=er)
        agent = AsymptoticallyOptimalUCB(3, rng=ar)
        return run_episode(agent, env, 100)

    a = once(123)
    b = once(123)
    assert np.array_equal(a.actions, b.actions)
    assert np.allclose(a.rewards, b.rewards)
    assert np.allclose(a.cumulative_pseudo_regret, b.cumulative_pseudo_regret)


def test_repetitions_reproduce():
    def factory_agent(rng, seed):
        return KLUCBAgent(3, rng=rng)

    def factory_env(rng, seed):
        return BernoulliBandit([0.2, 0.5, 0.7], rng=rng)

    r1 = run_repetitions(factory_agent, factory_env, horizon=80, n_runs=3, seed=7)
    r2 = run_repetitions(factory_agent, factory_env, horizon=80, n_runs=3, seed=7)
    assert np.allclose(r1.regrets, r2.regrets)
