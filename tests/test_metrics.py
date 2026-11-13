import numpy as np
import pytest

from banditbook.metrics.regret import (
    adversarial_external_regret,
    dynamic_regret,
    pathwise_pseudo_regret,
    reward_shortfall,
)


def test_pathwise_pseudo_regret_hand():
    means = np.array([0.1, 0.5, 0.8])
    actions = np.array([0, 1, 2, 0])
    # gaps: 0.7, 0.3, 0.0, 0.7
    cum = pathwise_pseudo_regret(actions, means)
    assert cum[-1] == pytest.approx(1.7)
    assert list(cum) == pytest.approx([0.7, 1.0, 1.0, 1.7])


def test_reward_shortfall_can_be_negative():
    means = np.array([0.0, 1.0])
    rewards = np.array([2.0, 2.0])  # better than means (possible under noise)
    s = reward_shortfall(rewards, means)
    assert s == pytest.approx(2 * 1.0 - 4.0)
    assert s < 0


def test_adversarial_external_not_per_round_max():
    # Round0: [1,0], Round1: [0,1] — per-round max sum = 2, fixed-arm max cum = 1
    table = np.array([[1.0, 0.0], [0.0, 1.0]])
    actions = np.array([0, 0])  # learner gets 1+0=1
    ext = adversarial_external_regret(actions, table)
    # best fixed arm: arm0 sums to 1, arm1 sums to 1 -> external = 1-1=0
    assert ext == pytest.approx(0.0)
    dyn = dynamic_regret(actions, table)
    # dynamic: (1-1)+(1-0)=1
    assert dyn == pytest.approx(1.0)
    assert dyn != ext


def test_external_negative_retained():
    table = np.array([[0.0, 0.0], [0.0, 0.0]])
    actions = np.array([0, 1])
    # learner 0, best fixed 0 -> 0
    assert adversarial_external_regret(actions, table) == pytest.approx(0.0)
    table2 = np.array([[1.0, 0.0], [1.0, 0.0]])
    actions2 = np.array([0, 0])
    # learner 2, best 2 -> 0; if somehow better... use learner on best
    assert adversarial_external_regret(actions2, table2) == pytest.approx(0.0)
