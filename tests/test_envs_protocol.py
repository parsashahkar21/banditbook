import numpy as np
import pytest

from banditbook.agents.base import Feedback
from banditbook.agents.ucb import AsymptoticallyOptimalUCB
from banditbook.envs.adversarial import (
    ObliviousAdversarialBandit,
    OmniscientAdversaryBandit,
    SwitchingObliviousBandit,
)
from banditbook.envs.stochastic import BernoulliBandit, GaussianBandit
from banditbook.evaluation.runner import run_episode
from banditbook.utils.rng import make_rng_pair


def test_means_preserve_0_and_1_not_0_1():
    # Historical main.py ambiguity: [0.1, 0.5, 0.8, 0.1, 0.1, 0, 1]
    means = [0.1, 0.5, 0.8, 0.1, 0.1, 0, 1]
    env = GaussianBandit(means, [1.0] * 7, rng=np.random.default_rng(0))
    assert len(env.means) == 7
    assert env.means[-2] == pytest.approx(0.0)
    assert env.means[-1] == pytest.approx(1.0)


def test_gaussian_uses_std_not_variance():
    env = GaussianBandit([0.0], [2.0], rng=np.random.default_rng(0))
    samples = [env.step(0).reward for _ in range(2000)]
    assert np.std(samples, ddof=1) == pytest.approx(2.0, rel=0.15)


def test_oblivious_vector_before_action():
    table = np.array([[0.0, 1.0], [1.0, 0.0]])
    env = ObliviousAdversarialBandit(table, rng=np.random.default_rng(0))
    env.prepare_round()
    assert env._pending is not None
    # Action does not change the pending vector
    pending = env._pending.copy()
    step = env.step(0)
    assert np.allclose(step.evaluator.reward_vector, pending)


def test_omniscient_labeled_counterexample():
    env = OmniscientAdversaryBandit(3, rng=np.random.default_rng(0))
    assert "counterexample" in env.label
    step = env.step(1)
    assert step.reward == 0.0
    assert step.evaluator.extras.get("action_aware") is True


def test_agent_does_not_receive_evaluator_info():
    agent_rng, env_rng, _ = make_rng_pair(0)
    env = BernoulliBandit([0.1, 0.9], rng=env_rng)
    agent = AsymptoticallyOptimalUCB(2, rng=agent_rng)

    leaked = []

    class Probe(AsymptoticallyOptimalUCB):
        def update(self, action, feedback: Feedback):
            leaked.append(feedback)
            super().update(action, feedback)

    probe = Probe(2, rng=agent_rng)
    run_episode(probe, env, horizon=20)
    for f in leaked:
        assert not hasattr(f, "true_means")
        assert "theta" not in f.info
        assert "true_means" not in f.info


def test_switching_dynamic_means():
    env = SwitchingObliviousBandit(3, horizon=250, switch_interval=100)
    assert env.dynamic_means.shape == (250, 3)
    assert env.dynamic_means[0, 0] == 1.0
    assert env.dynamic_means[100, 1] == 1.0
