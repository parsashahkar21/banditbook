# Chapter-by-chapter coverage matrix (38 chapters)

Statuses: **implemented** | **partial** | **planned** | **theory-only**

This matrix does **not** claim exhaustive book implementation merely because every chapter has a row.

| Ch | Topic | Artifact | Source | Setting | Validation | Limitations | Status |
|----|-------|----------|--------|---------|------------|-------------|--------|
| 1 | Introduction | docs notes | Ch. 1 | — | — | — | theory-only |
| 2 | Foundations of probability | — | Ch. 2 | — | — | — | theory-only |
| 3 | Stochastic processes / Markov | — | Ch. 3 | — | — | — | theory-only |
| 4 | Stochastic bandits / regret | `metrics/regret.py`, docs | Ch. 4 | finite-armed | unit tests | — | partial |
| 5 | Concentration | `docs/notes/concentration.py` | Ch. 5 | educational | runnable demo | not a proof | partial |
| 6 | ETC | `ETCAgent` | Alg. 1 | subgaussian | hand tests | short-horizon policy educational | implemented |
| 7 | UCB(δ) | `UCBDeltaAgent` | Alg. 3 | subgaussian | tests | — | implemented |
| 8 | Asymptotic UCB | `AsymptoticallyOptimalUCB` | Alg. 6 | subgaussian | tests + smoke | — | implemented |
| 9 | MOSS | `MOSSAgent` | Alg. 7 | subgaussian | tests + smoke | needs n | implemented |
| 10 | KL-UCB | `KLUCBAgent`, `utils/kl.py` | Alg. 8 | Bernoulli | KL unit + smoke | bisection tol | implemented |
| 11 | Exp3 | `Exp3Agent` | Alg. 9 | adv. [0,1] | hand + smoke | bounded only | implemented |
| 12 | Exp3-IX | `Exp3IXAgent` | Alg. 10 | adv. [0,1] | hand + smoke | — | implemented |
| 13 | Lower bounds ideas | notes | Ch. 13 | — | demo intuition | not theorems as code | theory-only |
| 14 | Information theory | `bernoulli_kl` util | Ch. 14/10 | Bernoulli KL | unit tests | subset | partial |
| 15 | Minimax lower bounds | notes | Ch. 15 | — | — | — | theory-only |
| 16 | Instance-dependent LB | notes | Ch. 16 | — | — | — | theory-only |
| 17 | High-probability LB | notes | Ch. 17 | — | — | — | theory-only |
| 18 | Contextual / Exp4 | `Exp4Agent`, `PerContextUCBAgent` | Alg. 11, §18.1 | finite experts/contexts | unit tests | — | implemented |
| 19 | Stochastic linear / LinUCB | `LinUCBAgent`, RLS | Ch. 19 | finite actions | unit tests | not continuous A | implemented |
| 20 | LS confidence | `beta_oful`, RLS | Eq. 19.8 / Ch. 20 | linear | unit | simplified use of β | partial |
| 21 | Optimal design | `g_optimal_design` | Ch. 21 | finite A | diagnostic test | Frank–Wolfe approx | implemented |
| 22 | Phased elimination | `PhasedEliminationAgent` | Alg. 12 | finite linear | smoke-level | approx design/rounding | partial |
| 23 | Sparse linear | — | Ch. 23 | — | — | do not substitute Lasso heuristic | planned |
| 24 | Linear minimax LB | — | Ch. 24 | — | — | — | theory-only |
| 25 | Linear asymptotic LB | — | Ch. 25 | — | — | — | theory-only |
| 26 | Convex analysis | — | Ch. 26 | — | — | — | theory-only |
| 27 | Exp3 linear adversarial | — | Ch. 27 | — | — | needs domain assumptions | planned |
| 28 | FTRL / mirror descent | — | Ch. 28 | — | — | — | planned |
| 29 | Adv. vs stoch. linear | — | Ch. 29 | — | — | — | theory-only |
| 30 | Combinatorial | — | Ch. 30 | — | — | full vs semi-bandit | planned |
| 31 | Nonstationary | switching env + dynamic regret | Ch. 31 | partial | smoke switching | dedicated NS algorithms planned | partial |
| 32 | Ranking / click models | — | Ch. 32 | — | — | — | planned |
| 33 | Pure exploration / BAI | exploration metric helpers | Ch. 33 | — | metric unit helpers | FC/FB algorithms planned | partial |
| 34 | Bayesian learning foundations | — | Ch. 34 | — | — | — | theory-only |
| 35 | Bayesian bandits / Gittins | — | Ch. 35 | — | — | approx limits required | planned |
| 36 | Thompson sampling | — | Ch. 36 | — | — | — | planned |
| 37 | Partial monitoring | — | Ch. 37 | — | — | own feedback API | planned |
| 38 | Markov decision processes | — | Ch. 38 | — | — | tabular demos planned | planned |

Checkpoint: Milestones 1–5 core acceptance items are in place; Milestone 6 advanced increments remain mostly **planned**.
