# Roadmap and completion checkpoint

## Completed (usable release)

- [x] Milestone 1: audit, package structure, RNG, interfaces, adversarial timing
- [x] Milestone 2: distinct metrics + tests
- [x] Milestone 3: finite-armed core agents (uniform, ETC, UCB family, MOSS, KL-UCB, Exp3, Exp3-IX)
- [x] Milestone 4: config-driven runner, smoke configs, artifacts, headless plots
- [x] Milestone 5 (core): theory demos, Exp4, per-context UCB, LinUCB, RLS, G-design, phased elimination (partial)
- [x] Packaging, tests, CI workflow, documentation

## Remaining (Milestone 6 increments)

Verify the corresponding book section before selecting each concrete algorithm.

1. Sparse stochastic linear (Ch. 23) — not a generic Lasso policy unless labelled
2. Adversarial linear Exp3 (Ch. 27) with explicit action-domain assumptions
3. FTRL / mirror descent utilities + bandit instantiations (Ch. 28)
4. Combinatorial full-bandit vs semi-bandit (Ch. 30)
5. Nonstationary methods beyond switching comparator metrics (Ch. 31)
6. Ranking / click-model examples (Ch. 32)
7. Fixed-confidence / fixed-budget BAI (Ch. 33)
8. Bayesian posteriors + Thompson sampling (Ch. 34–36)
9. Small Gittins / Bayesian planning demos with approximation limits (Ch. 35)
10. Partial monitoring (Ch. 37) + tabular MDP examples (Ch. 38) with dedicated feedback APIs

## Non-goals for claiming “complete book code”

- Copying book text or shipping the PDF
- Treating asymptotic reference curves as finite-time lower bounds
- Labelling heuristics as named book algorithms
