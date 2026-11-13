"""Minimal quickstart: one Bernoulli episode with asymptotic UCB."""

from banditbook.agents.base import Feedback
from banditbook.agents.ucb import AsymptoticallyOptimalUCB
from banditbook.envs.stochastic import BernoulliBandit
from banditbook.metrics.regret import pathwise_pseudo_regret
from banditbook.utils.rng import make_rng_pair
import numpy as np

agent_rng, env_rng, _ = make_rng_pair(0)
means = [0.2, 0.5, 0.8]
env = BernoulliBandit(means, rng=env_rng)
agent = AsymptoticallyOptimalUCB(len(means), rng=agent_rng)
actions = []
for _ in range(200):
    a = agent.select_action()
    step = env.step(a)
    agent.update(a, Feedback(reward=step.reward))
    actions.append(a)
regret = pathwise_pseudo_regret(np.asarray(actions), np.asarray(means))
print(f"Final pathwise pseudo-regret: {regret[-1]:.3f}")
