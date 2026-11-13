# banditbook

Educational Python implementations of algorithms from
[*Bandit Algorithms*](https://tor-lattimore.com/downloads/book/book.pdf)
by Tor Lattimore and Csaba Szepesvári.

Focused on mathematical correctness, reproducibility, and clear separation between
learner-visible feedback and evaluator-only information. This is **not** an official
companion to the book and does **not** redistribute the PDF or copy book text.

## Features

- Finite-armed stochastic and adversarial bandits (ETC, UCB family, MOSS, KL-UCB, Exp3, Exp3-IX)
- Contextual / linear starters (Exp4, LinUCB, G-optimal design, phased elimination)
- Distinct regret metrics (pseudo-regret, external regret, dynamic regret, …)
- Config-driven experiments with seeded NumPy generators and saved artifacts
- Tests and GitHub Actions CI

## Requirements

- Python 3.10+

## Install

```bash
git clone https://github.com/<your-username>/banditbook.git
cd banditbook
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Quickstart

```bash
python examples/quickstart.py
python main.py --horizon 2000 --out results/main_demo.png
```

Smoke experiments:

```bash
python -m banditbook.experiments --config configs/smoke_stochastic.yaml --out results/smoke_stochastic
python -m banditbook.experiments --config configs/smoke_adversarial.yaml --out results/smoke_adversarial
```

## Tests

```bash
pytest
```

## Project layout

```
banditbook/     # main package (agents, envs, metrics, evaluation, …)
configs/        # YAML experiment configs
docs/           # algorithm map, metrics, coverage, roadmap
examples/       # short runnable scripts
tests/          # unit and reproducibility tests
main.py         # small demo entry point
```

Legacy top-level `agents/` and `envs/` folders re-export the package for older imports.

## Documentation

| Doc | Contents |
|-----|----------|
| [`docs/ALGORITHMS.md`](docs/ALGORITHMS.md) | Algorithm → book chapter mapping |
| [`docs/METRICS.md`](docs/METRICS.md) | Regret definitions |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Package design |
| [`docs/COVERAGE.md`](docs/COVERAGE.md) | Chapter-by-chapter status |
| [`docs/ROADMAP.md`](docs/ROADMAP.md) | Remaining work |
| [`docs/AUDIT.md`](docs/AUDIT.md) | Audit notes vs the original code |

## Citation

If you use this code while studying the book, cite the book itself:

> Tor Lattimore and Csaba Szepesvári. *Bandit Algorithms*. Cambridge University Press, 2020.  
> Free online edition: https://tor-lattimore.com/downloads/book/book.pdf

## Status

Core finite-armed algorithms and the experiment harness are implemented and tested.
Advanced chapters (sparse linear, combinatorial, Thompson sampling, MDPs, …) are
tracked in `docs/COVERAGE.md` and are not claimed as complete.
