# Milestone 1 audit notes

Source book: [Bandit Algorithms](https://tor-lattimore.com/downloads/book/book.pdf) (Lattimore & Szepesvári).  
**Source verification status:** PDF fetched and checked for Algorithms 1, 3, 6, 7, 8, 9, 10, 11, 12 and Ch. 19 formulas in this session.

## Findings on the original repository

| Area | Original behaviour | Action taken |
|------|-------------------|--------------|
| Means in `main.py` | `[0.1, 0.5, 0.8, 0.1, 0.1, 0, 1]` — trailing `0, 1` are **two arms** | Preserved; documented (not collapsed to `0.1`) |
| Gaussian noise | `stds` passed to `np.random.normal` (standard deviation) | Kept; documented vs book σ-subgaussian convention |
| ETC | Argmin-counts exploration (not book round-robin); no short-horizon policy | Replaced with Algorithm 1 round-robin + documented T < Km policy |
| UCB | Tunable `c * sqrt(log t / T_i)` | Relabelled as educational `TunableUCBAgent`; added Alg. 3 and Alg. 6 |
| Regret | Pathwise pseudo-regret | Kept as primary stochastic metric; added distinct alternatives |
| Adversaries | Reward vector built **after** seeing action; omniscient uses current arm | Standard envs prepare vector first; omniscient kept as labelled counterexample |
| RNG | Global `np.random` | `Generator` + `SeedSequence`; agent/env streams separated |
| Agent API | `select_arm` / `update` / no `reset` | `select_action` / `update(Feedback)` / `reset` |
| Evaluator leakage | Means used only in experiment loop (OK) | Formalised: agents get `Feedback` only |

## Noise-scale convention

For σ-subgaussian rewards, UCB/MOSS bonuses are multiplied by σ (Exercise 7.2). Gaussian `N(μ, σ²)` is σ-subgaussian. Default `sigma=1` matches the book’s 1-subgaussian chapters.
