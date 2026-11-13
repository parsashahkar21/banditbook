"""NumPy Generator + SeedSequence independent random streams."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def spawn_rngs(seed: int | np.random.SeedSequence | None, n: int) -> list[np.random.Generator]:
    """Spawn ``n`` independent Generators from a seed or SeedSequence."""
    if isinstance(seed, np.random.SeedSequence):
        ss = seed
    else:
        ss = np.random.SeedSequence(seed)
    children = ss.spawn(n)
    return [np.random.default_rng(child) for child in children]


def make_rng_pair(
    seed: int | np.random.SeedSequence | None = None,
) -> tuple[np.random.Generator, np.random.Generator, np.random.SeedSequence]:
    """Return (agent_rng, env_rng, root_seed_sequence).

    Agent and environment randomness are separated so policies and environments
    can be recombined without cross-contamination of streams.
    """
    root = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)
    agent_ss, env_ss = root.spawn(2)
    return np.random.default_rng(agent_ss), np.random.default_rng(env_ss), root


def spawn_run_seeds(master_seed: int | None, n_runs: int) -> list[int]:
    """Deterministic integer seeds for independent experiment repetitions."""
    ss = np.random.SeedSequence(master_seed)
    # Use first 32 bits of each spawned entropy as a stable int seed.
    return [int(child.generate_state(1, dtype=np.uint32)[0]) for child in ss.spawn(n_runs)]
