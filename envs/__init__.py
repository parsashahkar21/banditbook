"""Legacy import path — prefer ``banditbook.envs``."""

from banditbook.envs.base import BanditEnvironment
from banditbook.envs.stochastic import BernoulliBandit, GaussianBandit
from banditbook.envs.adversarial import (
    MinimizingAdversary,
    OmniscientAdversary,
    SwitchingAdversary,
)

__all__ = [
    "BanditEnvironment",
    "BernoulliBandit",
    "GaussianBandit",
    "OmniscientAdversary",
    "SwitchingAdversary",
    "MinimizingAdversary",
]
