"""Input validation helpers."""

from __future__ import annotations

import numpy as np


def check_positive_int(name: str, value: int, *, allow_zero: bool = False) -> int:
    if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, got {type(value)!r}")
    value = int(value)
    if allow_zero:
        if value < 0:
            raise ValueError(f"{name} must be >= 0, got {value}")
    elif value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")
    return value


def check_probability(name: str, value: float) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")
    return value


def check_arm(arm: int, n_arms: int) -> int:
    if not isinstance(arm, (int, np.integer)) or isinstance(arm, bool):
        raise TypeError(f"arm must be an integer, got {type(arm)!r}")
    arm = int(arm)
    if arm < 0 or arm >= n_arms:
        raise ValueError(f"arm index {arm} out of range for K={n_arms}")
    return arm


def as_1d_float_array(name: str, values, *, min_len: int = 1) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}")
    if arr.size < min_len:
        raise ValueError(f"{name} must have length >= {min_len}, got {arr.size}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr
