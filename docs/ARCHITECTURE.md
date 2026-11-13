# Architecture

```
banditbook/
  agents/          # learners: select_action / update / reset
  envs/            # environments: prepare_round / step → StepResult
  metrics/         # regret & exploration metrics (evaluator-side)
  evaluation/      # episode runner, registries, multi-run experiments
  utils/           # RNG, KL, RLS, optimal design, validation
  plotting/        # headless matplotlib helpers
  config/          # YAML/JSON load + artifact writers
  experiments.py   # CLI entry for config-driven runs
```

## Information flow

1. `env.prepare_round()` may fix the latent reward vector / emit public context.
2. `agent.select_action(context)` chooses an action.
3. `env.step(action)` returns learner reward + `EvaluatorInfo` (hidden).
4. `agent.update(action, Feedback(...))` receives **only** learner-visible fields.

Agents never see true means, θ, unselected rewards, or future outcomes.

## Randomness

`make_rng_pair(seed)` → independent `(agent_rng, env_rng)` via `SeedSequence.spawn`.
Experiment repetitions use `spawn_run_seeds`.

## Reproducibility artifacts

Each run directory stores `config.json`, `environment.json` (versions + git rev if available), `results.npz`, `summary.csv`, and plots.
