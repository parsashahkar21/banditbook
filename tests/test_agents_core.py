"""Deterministic hand-worked tests for selection / updates."""

import numpy as np
import pytest

from banditbook.agents.base import Feedback
from banditbook.agents.etc import ETCAgent
from banditbook.agents.exp3 import Exp3Agent
from banditbook.agents.exp3ix import Exp3IXAgent
from banditbook.agents.moss import MOSSAgent
from banditbook.agents.ucb import AsymptoticallyOptimalUCB, UCBDeltaAgent


def fb(r):
    return Feedback(reward=r)


def test_etc_round_robin_then_commit():
    agent = ETCAgent(n_arms=3, m=2, rng=np.random.default_rng(0))
    # Explore 6 rounds: 0,1,2,0,1,2
    expected = [0, 1, 2, 0, 1, 2]
    for i, a_exp in enumerate(expected):
        a = agent.select_action()
        assert a == a_exp
        # Make arm 1 clearly best
        reward = 1.0 if a == 1 else 0.0
        agent.update(a, fb(reward))
    # Commit to arm 1
    for _ in range(5):
        assert agent.select_action() == 1
        agent.update(1, fb(0.0))


def test_etc_short_horizon_no_commit():
    agent = ETCAgent(n_arms=3, m=5, rng=np.random.default_rng(0))
    # Only 4 pulls < 15 explore budget: stay in round-robin
    seen = [agent.select_action() for _ in range(4)]
    # Don't update between selects in this check — actually need update to advance counts
    agent2 = ETCAgent(n_arms=3, m=5, rng=np.random.default_rng(0))
    seq = []
    for _ in range(4):
        a = agent2.select_action()
        seq.append(a)
        agent2.update(a, fb(0.0))
    assert seq == [0, 1, 2, 0]
    assert agent2.best_arm is None


def test_ucb_delta_unpulled_priority():
    agent = UCBDeltaAgent(n_arms=3, delta=0.01, rng=np.random.default_rng(0))
    assert agent.select_action() == 0
    agent.update(0, fb(0.0))
    assert agent.select_action() == 1
    agent.update(1, fb(0.0))
    assert agent.select_action() == 2


def test_asymptotic_ucb_formula_step():
    agent = AsymptoticallyOptimalUCB(n_arms=2, rng=np.random.default_rng(0), sigma=1.0)
    agent.update(0, fb(0.0))  # bypass select to set counts — need both pulled
    # Force state: one pull each with means 0 and 1
    agent.counts[:] = [1, 1]
    agent.sums[:] = [0.0, 1.0]
    agent.t = 2
    # Upcoming t=3; f(3)=1+3*log(3)^2
    a = agent.select_action()
    # Arm 1 has higher mean; with equal counts should pick 1
    assert a == 1


def test_moss_requires_horizon_and_handles_unpulled():
    agent = MOSSAgent(n_arms=2, horizon=100, rng=np.random.default_rng(0))
    assert agent.select_action() == 0
    agent.update(0, fb(0.5))
    assert agent.select_action() == 1


def test_exp3_importance_weighted_update():
    rng = np.random.default_rng(0)
    agent = Exp3Agent(n_arms=2, horizon=100, rng=rng, eta=0.1)
    # Force probabilities by zeroing S then selecting
    a = agent.select_action()
    P = agent._last_P.copy()
    agent.update(a, fb(1.0))
    # X̂_a = 1 - (1-1)/P_a = 1; X̂_other = 1
    # So S increases by 1 for both — relative unchanged
    assert agent.S[0] == pytest.approx(1.0)
    assert agent.S[1] == pytest.approx(1.0)

    agent2 = Exp3Agent(n_arms=2, horizon=100, rng=np.random.default_rng(1), eta=0.1)
    a = agent2.select_action()
    P = agent2._last_P.copy()
    agent2.update(a, fb(0.0))
    # X̂_a = 1 - 1/P_a; X̂_j = 1
    expected = np.ones(2)
    expected[a] = 1.0 - 1.0 / P[a]
    assert agent2.S == pytest.approx(expected)


def test_exp3_rejects_unbounded():
    agent = Exp3Agent(n_arms=2, horizon=10, rng=np.random.default_rng(0))
    agent.select_action()
    with pytest.raises(ValueError):
        agent.update(0, fb(1.5))


def test_exp3ix_loss_update():
    agent = Exp3IXAgent(n_arms=2, horizon=100, rng=np.random.default_rng(0), eta=0.2, gamma=0.1)
    a = agent.select_action()
    P = agent._last_P.copy()
    agent.update(a, fb(0.0))  # loss=1
    expected = np.zeros(2)
    expected[a] = 1.0 / (P[a] + 0.1)
    assert agent.L == pytest.approx(expected)


def test_k1_and_ties():
    agent = ETCAgent(n_arms=1, m=3, rng=np.random.default_rng(0))
    for _ in range(5):
        assert agent.select_action() == 0
        agent.update(0, fb(0.5))
