"""Educational: regret decomposition R_n = sum_i Δ_i E[T_i(n)] (Book Lemma 4.5 intuition).

Pathwise analogue on one run: sum_t Δ_{A_t} = sum_i Δ_i T_i.
Not a proof that empirical plots establish lower bounds.
"""

from __future__ import annotations

import numpy as np

from banditbook.agents.base import Feedback
from banditbook.agents.ucb import AsymptoticallyOptimalUCB
from banditbook.envs.stochastic import BernoulliBandit
from banditbook.utils.rng import make_rng_pair


def pathwise_decomposition(means: np.ndarray, actions: np.ndarray) -> dict:
    means = np.asarray(means, dtype=float)
    actions = np.asarray(actions, dtype=int)
    opt = float(np.max(means))
    gaps = opt - means
    regret = float(np.sum(gaps[actions]))
    counts = np.bincount(actions, minlength=len(means))
    decomp = float(np.dot(gaps, counts))
    return {"regret": regret, "decomposition": decomp, "counts": counts, "gaps": gaps}


if __name__ == "__main__":
    means = np.array([0.2, 0.5, 0.7])
    ar, er, _ = make_rng_pair(0)
    env = BernoulliBandit(means, rng=er)
    agent = AsymptoticallyOptimalUCB(3, rng=ar)
    actions = []
    for _ in range(500):
        a = agent.select_action()
        step = env.step(a)
        agent.update(a, Feedback(reward=step.reward))
        actions.append(a)
    print(pathwise_decomposition(means, np.asarray(actions)))
