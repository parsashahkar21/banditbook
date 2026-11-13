"""Hand-worked KL and KL-UCB index tests."""

import math

import pytest

from banditbook.utils.kl import bernoulli_kl, kl_ucb_index


def test_bernoulli_kl_endpoints():
    assert bernoulli_kl(0.0, 0.0) == 0.0
    assert bernoulli_kl(1.0, 1.0) == 0.0
    assert bernoulli_kl(0.0, 1.0) == math.inf
    assert bernoulli_kl(1.0, 0.0) == math.inf
    assert bernoulli_kl(0.5, 0.5) == 0.0
    # d(0, 0.5) = log(1/0.5) = log(2)
    assert bernoulli_kl(0.0, 0.5) == pytest.approx(math.log(2))
    assert bernoulli_kl(1.0, 0.5) == pytest.approx(math.log(2))


def test_kl_ucb_index_hand():
    # If threshold large enough to reach 1
    idx = kl_ucb_index(0.5, n_pulls=10, threshold=10.0)
    assert idx == pytest.approx(1.0)
    # Zero threshold -> empiric mean
    assert kl_ucb_index(0.3, n_pulls=5, threshold=0.0) == pytest.approx(0.3)
    # Unpulled
    assert kl_ucb_index(0.0, n_pulls=0, threshold=1.0) == math.inf
    # Monotone: larger threshold -> larger or equal index
    a = kl_ucb_index(0.2, 20, 0.1)
    b = kl_ucb_index(0.2, 20, 0.5)
    assert b >= a >= 0.2
