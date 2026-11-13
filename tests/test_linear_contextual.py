import numpy as np
import pytest

from banditbook.agents.base import Feedback
from banditbook.agents.exp4 import Exp4Agent
from banditbook.agents.linucb import LinUCBAgent
from banditbook.envs.linear import StochasticLinearBandit
from banditbook.evaluation.runner import run_episode
from banditbook.utils.design import g_optimal_design
from banditbook.utils.linalg import RegularizedLeastSquares
from banditbook.utils.rng import make_rng_pair


def test_rls_recovers_simple_theta():
    rls = RegularizedLeastSquares(2, lam=1e-3)
    theta = np.array([1.0, -1.0])
    rng = np.random.default_rng(0)
    for _ in range(200):
        a = rng.normal(size=2)
        r = float(np.dot(theta, a))
        rls.update(a, r)
    assert rls.theta_hat == pytest.approx(theta, abs=0.15)


def test_g_optimal_design_diagnostic():
    # Simplex vertices / identity basis in 2D: two orthogonal unit vectors + negatives
    actions = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    res = g_optimal_design(actions, max_iter=200, tol=1e-3)
    assert res.weights.sum() == pytest.approx(1.0)
    # At optimum max leverage ≈ d = 2
    assert res.max_leverage == pytest.approx(2.0, abs=0.2)


def test_linucb_episode_runs():
    ar, er, _ = make_rng_pair(0)
    theta = np.array([0.5, -0.2])
    actions = np.array([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]])
    env = StochasticLinearBandit(theta, actions, noise_std=0.1, rng=er)
    agent = LinUCBAgent(
        n_arms=3, dim=2, horizon=50, rng=ar, fixed_actions=actions, lam=1.0
    )
    log = run_episode(agent, env, 50)
    assert log.cumulative_pseudo_regret is not None
    assert log.cumulative_pseudo_regret[-1] >= 0


def test_exp4_with_constant_experts():
    rng = np.random.default_rng(0)
    agent = Exp4Agent(n_arms=2, n_experts=2, horizon=20, rng=rng, eta=0.5)
    # Expert 0 always picks arm 0; expert 1 always picks arm 1
    E = np.array([[1.0, 0.0], [0.0, 1.0]])
    for _ in range(10):
        a = agent.select_action(E)
        agent.update(a, Feedback(reward=1.0 if a == 1 else 0.0))
    # Expert 1 should get more weight
    assert agent.Q[1] > agent.Q[0]
