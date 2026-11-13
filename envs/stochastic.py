"""Legacy module path. Prefer ``banditbook.envs.stochastic``."""

from banditbook.envs.stochastic import BernoulliBandit, GaussianBandit

__all__ = ["BernoulliBandit", "GaussianBandit"]
