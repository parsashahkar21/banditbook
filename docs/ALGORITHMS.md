# Algorithm-to-source mapping

Reference: https://tor-lattimore.com/downloads/book/book.pdf

| Implementation | Book locus | Setting | Kind | Notes |
|----------------|------------|---------|------|-------|
| `UniformRandomAgent` | — | any finite-armed | baseline | Not a named book algorithm |
| `ETCAgent` | Ch. 6, Alg. 1 | stochastic subgaussian | **exact** | Short-horizon policy documented |
| `UCBDeltaAgent` | Ch. 7, Alg. 3 | 1-subgaussian (+σ) | **exact** (+σ ext.) | δ input; typical δ=1/n² |
| `AsymptoticallyOptimalUCB` | Ch. 8, Alg. 6 | 1-subgaussian (+σ) | **exact** (+σ ext.) | f(t)=1+t log²(t) |
| `TunableUCBAgent` | — | educational | **extension** | **Not** Alg. 6 |
| `MOSSAgent` | Ch. 9, Alg. 7 | 1-subgaussian (+σ) | **exact** (+σ ext.) | Needs horizon |
| `KLUCBAgent` | Ch. 10, Alg. 8 | Bernoulli | **exact** index; **numerical** bisection | tol=1e-8 |
| `Exp3Agent` | Ch. 11, Alg. 9 | adversarial [0,1] | **exact** | Rejects rewards outside [0,1] |
| `Exp3IXAgent` | Ch. 12, Alg. 10 | adversarial [0,1] | **exact** | Loss form; γ=η/2 default |
| `Exp4Agent` | Ch. 18, Alg. 11 | experts / contextual | **exact** (γ=0 default) | Finite expert matrix interface |
| `PerContextUCBAgent` | Ch. 18.1 idea | discrete contexts | **educational** | Independent Alg. 6 per context |
| `LinUCBAgent` | Ch. 19 | stochastic linear, **finite** actions | **specified variant** | Eq. 19.8 β; not continuous A |
| `PhasedEliminationAgent` | Ch. 22, Alg. 12 | stochastic linear, finite A | **approx design** | Frank–Wolfe G-design + rounding |
| `RegularizedLeastSquares` | Ch. 19–20 | linear utilities | **exact** closed form | Cholesky solves |
| `g_optimal_design` | Ch. 21 | finite A | **numerical approx** | Reports optimality gap |

## Intentional deviations

1. **σ-scale** on subgaussian UCB/MOSS: multiply bonus by σ (book Exercise 7.2).
2. **ETC when T < Km**: continue round-robin; no commit (educational edge case; Thm 6.1 assumes m ≤ n/k).
3. **KL-UCB index**: bisection tolerance 1e-8 (numerical approximation of the exact argmax).
4. **G-optimal design**: Frank–Wolfe approximation with reported `optimality_gap`.
5. **TunableUCBAgent**: sensitivity tool only — do not cite as asymptotically optimal UCB.
