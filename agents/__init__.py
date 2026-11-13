"""Legacy import path — prefer ``banditbook.agents``."""

from banditbook.agents.etc import ETCAgent
from banditbook.agents.ucb import TunableUCBAgent as UCBAgent

__all__ = ["ETCAgent", "UCBAgent"]
