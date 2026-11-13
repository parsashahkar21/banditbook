"""Legacy module path. Prefer ``banditbook.agents.ucb``.

Note: the historical ``UCBAgent`` is the educational tunable UCB, not
Book Algorithm 6. Use ``AsymptoticallyOptimalUCB`` for the book variant.
"""

from banditbook.agents.ucb import (
    AsymptoticallyOptimalUCB,
    TunableUCBAgent as UCBAgent,
    UCBDeltaAgent,
)

__all__ = ["UCBAgent", "UCBDeltaAgent", "AsymptoticallyOptimalUCB"]
