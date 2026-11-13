"""Legacy module path. Prefer ``banditbook.envs.adversarial``.

The historical ``AdversarialBandit.pull`` built rewards after seeing the action.
Standard experiments now use ``ObliviousAdversarialBandit`` / prepare_round.
Omniscient remains labeled as a counterexample.
"""

from banditbook.envs.adversarial import (
    CountBasedAdversaryBandit,
    MinimizingAdversary,
    ObliviousAdversarialBandit,
    OmniscientAdversary,
    OmniscientAdversaryBandit,
    SwitchingAdversary,
    SwitchingObliviousBandit,
)

__all__ = [
    "OmniscientAdversary",
    "SwitchingAdversary",
    "MinimizingAdversary",
    "ObliviousAdversarialBandit",
    "SwitchingObliviousBandit",
    "OmniscientAdversaryBandit",
    "CountBasedAdversaryBandit",
]
