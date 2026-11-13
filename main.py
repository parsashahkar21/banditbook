"""Convenient entry point for a small Gaussian ETC + UCB demo.

Historical note on means:
  The original script used ``means = [0.1, 0.5, 0.8, 0.1, 0.1, 0, 1]``.
  The final ``0, 1`` are two separate arm means (0 and 1), not ``0.1``.
  That vector is preserved below.

For configuration-driven comparisons see:
  python -m banditbook.experiments --config configs/smoke_stochastic.yaml --out results/smoke
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from banditbook.agents.base import Feedback
from banditbook.agents.etc import ETCAgent
from banditbook.agents.ucb import AsymptoticallyOptimalUCB
from banditbook.envs.stochastic import GaussianBandit
from banditbook.metrics.regret import pathwise_pseudo_regret
from banditbook.utils.rng import make_rng_pair


def run_demo(horizon: int = 2000, seed: int = 0, out: str = "results/main_demo.png") -> None:
    # Preserved historical means array (7 arms; last two are 0 and 1).
    means = [0.1, 0.5, 0.8, 0.1, 0.1, 0, 1]
    stds = [1.0] * len(means)

    agent_rng, env_rng, _ = make_rng_pair(seed)
    env = GaussianBandit(means, stds, rng=env_rng)
    # ETC m=50 as in the original script; book Thm 6.1 assumes m <= n/k.
    etc = ETCAgent(n_arms=len(means), m=50, rng=agent_rng)
    ucb = AsymptoticallyOptimalUCB(n_arms=len(means), rng=np.random.default_rng(seed + 1), sigma=1.0)

    def play(agent):
        env.reset(np.random.default_rng(seed))
        agent.reset()
        # Re-bind env rng for fair-ish comparison note: separate streams; not CRN here.
        actions, rewards = [], []
        for _ in range(horizon):
            a = agent.select_action()
            step = env.step(a)
            agent.update(a, Feedback(reward=step.reward))
            actions.append(a)
            rewards.append(step.reward)
        regret = pathwise_pseudo_regret(np.asarray(actions), np.asarray(means))
        return regret

    # Fresh env streams per agent for this demo (documented; not shared CRN).
    agent_rng, env_rng, _ = make_rng_pair(seed)
    env = GaussianBandit(means, stds, rng=env_rng)
    etc = ETCAgent(n_arms=len(means), m=50, rng=agent_rng)
    etc_reg = play(etc)

    agent_rng, env_rng, _ = make_rng_pair(seed + 100)
    env = GaussianBandit(means, stds, rng=env_rng)
    ucb = AsymptoticallyOptimalUCB(n_arms=len(means), rng=agent_rng, sigma=1.0)
    ucb_reg = play(ucb)

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(etc_reg, label="ETC (m=50)", linewidth=2)
    plt.plot(ucb_reg, label="Asymptotically optimal UCB (Alg. 6)", linewidth=2)
    plt.title("Gaussian bandit demo (pathwise pseudo-regret)")
    plt.xlabel("Round")
    plt.ylabel("Cumulative pseudo-regret")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print(f"ETC final pseudo-regret: {etc_reg[-1]:.2f}")
    print(f"UCB final pseudo-regret: {ucb_reg[-1]:.2f}")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="results/main_demo.png")
    args = parser.parse_args()
    run_demo(args.horizon, args.seed, args.out)
