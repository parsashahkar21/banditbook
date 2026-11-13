"""Headless regret plotting."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from banditbook.evaluation.runner import ExperimentResult


def plot_mean_regret(
    results: Mapping[str, ExperimentResult],
    path: str | Path,
    *,
    title: str = "Mean regret",
    xlabel: str = "Round",
    ylabel: str | None = None,
) -> Path:
    """Plot mean regret curves with pointwise ±1 stderr bands across runs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, res in results.items():
        mean = res.mean_regret_curve
        se = res.stderr_regret_curve
        t = np.arange(1, len(mean) + 1)
        ax.plot(t, mean, label=name, linewidth=2)
        ax.fill_between(t, mean - se, mean + se, alpha=0.2)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel or f"Mean {next(iter(results.values())).metric_name}")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
