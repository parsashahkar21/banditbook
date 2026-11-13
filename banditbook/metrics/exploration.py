"""Pure-exploration metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def recommendation_error(recommended: int, optimal: int) -> int:
    """1 if recommendation ≠ optimal arm, else 0."""
    return int(int(recommended) != int(optimal))


def simple_regret(recommended: int, means: np.ndarray) -> float:
    """max_i μ_i - μ[recommended]."""
    means = np.asarray(means, dtype=float)
    return float(np.max(means) - means[int(recommended)])


@dataclass
class SampleComplexityTracker:
    """Tracks first time a stopping rule fires and whether recommendation is correct."""

    stopped_at: int | None = None
    recommendation: int | None = None
    correct: bool | None = None

    def record(self, t: int, recommended: int, optimal: int) -> None:
        if self.stopped_at is None:
            self.stopped_at = int(t)
            self.recommendation = int(recommended)
            self.correct = int(recommended) == int(optimal)
