# Metrics

## 1. Stochastic pathwise pseudo-regret

\[
\sum_{t=1}^{T} \bigl(\max_i \mu_i - \mu_{A_t}\bigr)
\]

Average across independent runs estimates expected regret under the model.

## 2. Stochastic reward-based shortfall (optional, separately labelled)

\[
T \max_i \mu_i - \sum_{t=1}^{T} X_t
\]

Negative values are **not** clamped (possible under noise).

## 3. Adversarial realised external regret (reward notation)

\[
\max_i \sum_{t=1}^{T} x_{t,i} - \sum_{t=1}^{T} x_{t,A_t}
\]

Comparator is a **single fixed arm** (cumulative reward per arm). Not the per-round max.

## 4. Nonstationary dynamic regret

\[
\sum_{t=1}^{T} \bigl(\max_i x_{t,i} - x_{t,A_t}\bigr)
\]

Uses time-dependent means or realised rewards. Reported separately from external regret.

## 5. Linear / contextual pseudo-regret (per round)

\[
\max_{a \in \mathcal{A}_t} \langle \theta, a \rangle - \langle \theta, a_t \rangle
\]

## 6. Pure exploration

- Recommendation error: \(\mathbf{1}\{\hat{a} \neq a^\star\}\)
- Simple regret: \(\max_i \mu_i - \mu_{\hat{a}}\)
- Sample complexity: first stopping time (tracker utility)

## Adaptive adversaries and expectation/max ordering

For adaptive adversaries, \(\mathbb{E}[\max_i \cdots]\) and \(\max_i \mathbb{E}[\cdots]\) differ.
A theorem bounding expected regret against a **fixed comparator** does **not** automatically bound expected **realised external regret** under adaptive adversaries. Document which quantity you plot.
