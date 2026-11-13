"""Configuration loading and run artifact persistence."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from banditbook.evaluation.runner import ExperimentResult


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    text = path.read_text()
    if path.suffix in {".yaml", ".yml"}:
        return yaml.safe_load(text)
    if path.suffix == ".json":
        return json.loads(text)
    raise ValueError(f"Unsupported config format: {path.suffix}")


def get_environment_metadata() -> dict[str, Any]:
    meta = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
    }
    try:
        import matplotlib

        meta["matplotlib"] = matplotlib.__version__
    except Exception:
        pass
    try:
        rev = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        meta["git_revision"] = rev
    except Exception:
        meta["git_revision"] = None
    return meta


def save_run_artifacts(
    out_dir: str | Path,
    config: dict,
    results: dict[str, ExperimentResult],
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, default=_json_default))
    (out_dir / "environment.json").write_text(
        json.dumps(get_environment_metadata(), indent=2)
    )
    summary_rows = []
    npz_payload = {}
    for name, res in results.items():
        npz_payload[f"{name}_regrets"] = res.regrets
        npz_payload[f"{name}_final"] = res.final_regrets
        npz_payload[f"{name}_pulls"] = res.pull_counts
        npz_payload[f"{name}_cum_reward"] = res.cumulative_rewards
        npz_payload[f"{name}_runtime"] = res.runtimes
        summary_rows.append(
            {
                "algorithm": name,
                "metric": res.metric_name,
                "mean_final_regret": float(res.final_regrets.mean()),
                "stderr_final_regret": float(
                    res.final_regrets.std(ddof=1) / np.sqrt(len(res.final_regrets))
                    if len(res.final_regrets) > 1
                    else 0.0
                ),
                "mean_runtime_sec": float(res.runtimes.mean()),
            }
        )
    np.savez_compressed(out_dir / "results.npz", **npz_payload)
    # CSV summary
    import csv

    with (out_dir / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    return out_dir


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    raise TypeError(type(obj))
