"""Configuration-driven experiment entry points."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from banditbook.config.io import load_config, save_run_artifacts
from banditbook.envs.adversarial import make_oblivious_bernoulli_table
from banditbook.evaluation.registry import AGENT_REGISTRY, build_agent
from banditbook.evaluation.runner import run_repetitions
from banditbook.envs import (
    BernoulliBandit,
    GaussianBandit,
    ObliviousAdversarialBandit,
    SwitchingObliviousBandit,
)
from banditbook.plotting import plot_mean_regret
from banditbook.utils.rng import make_rng_pair


def run_from_config(cfg: dict[str, Any], out_dir: str | Path) -> dict:
    horizon = int(cfg["horizon"])
    n_runs = int(cfg["n_runs"])
    seed = int(cfg.get("seed", 0))
    env_cfg = cfg["environment"]
    algs = cfg["algorithms"]
    metric = cfg.get("metric", "auto")

    if env_cfg.get("type") in {"oblivious_bernoulli", "switching"} and cfg.get(
        "share_oblivious_table", True
    ):
        results = _rerun_shared_oblivious(cfg, env_cfg, algs, horizon, n_runs, seed, metric)
    else:
        results = {}
        for alg in algs:
            name = alg["name"]
            label = alg.get("label", name)
            params = dict(alg.get("params", {}))
            if name in {"moss", "exp3", "exp3ix", "exp4", "linucb"} and "horizon" not in params:
                params["horizon"] = horizon
            if name == "ucb_delta" and "delta" not in params:
                params["delta"] = 1.0 / (horizon**2)

            def agent_factory(rng, run_seed, _name=name, _params=params):
                n_arms = _infer_n_arms(env_cfg, horizon, rng)
                p = dict(_params)
                if _name == "etc" and "m" not in p:
                    p["m"] = max(1, horizon // (10 * n_arms))
                return build_agent(_name, n_arms, rng, p)

            def env_factory(rng, run_seed, _env=env_cfg, _h=horizon):
                return _build_env_for_run(_env, _h, rng, run_seed)

            # Stochastic comparisons: same run seeds across algorithms (CRN on env stream
            # only if env_factory uses rng identically; documented in configs).
            results[label] = run_repetitions(
                agent_factory,
                env_factory,
                horizon=horizon,
                n_runs=n_runs,
                seed=seed,
                metric=metric,
            )

    out = save_run_artifacts(out_dir, cfg, results)
    plot_path = Path(out_dir) / "regret.png"
    plot_mean_regret(results, plot_path, title=cfg.get("title", "Regret comparison"))
    return {"out_dir": str(out), "results": results, "plot": str(plot_path)}


def _infer_n_arms(env_cfg: dict, horizon: int, rng: np.random.Generator) -> int:
    t = env_cfg["type"]
    if t == "bernoulli":
        return len(env_cfg["probabilities"])
    if t == "gaussian":
        return len(env_cfg["means"])
    if t == "oblivious_bernoulli":
        return len(env_cfg["probabilities"])
    if t == "switching":
        return int(env_cfg["n_arms"])
    if t == "oblivious_table":
        return int(np.asarray(env_cfg["reward_table"]).shape[1])
    raise ValueError(t)


def _build_env_for_run(env_cfg: dict, horizon: int, rng: np.random.Generator, run_seed: int):
    t = env_cfg["type"]
    if t == "bernoulli":
        return BernoulliBandit(env_cfg["probabilities"], rng=rng)
    if t == "gaussian":
        return GaussianBandit(env_cfg["means"], env_cfg.get("stds"), rng=rng)
    if t == "oblivious_bernoulli":
        # Independent table per algorithm run (overridden by shared path)
        table = make_oblivious_bernoulli_table(
            np.asarray(env_cfg["probabilities"], dtype=float), horizon, rng
        )
        return ObliviousAdversarialBandit(table, rng=rng)
    if t == "switching":
        return SwitchingObliviousBandit(
            env_cfg["n_arms"], horizon, env_cfg.get("switch_interval", 100), rng=rng
        )
    if t == "oblivious_table":
        return ObliviousAdversarialBandit(np.asarray(env_cfg["reward_table"], dtype=float), rng=rng)
    raise ValueError(f"unknown env type {t}")


def _rerun_shared_oblivious(cfg, env_cfg, algs, horizon, n_runs, seed, metric):
    """Re-run so that within each repetition, all algorithms see the same reward table."""
    from banditbook.utils.rng import spawn_run_seeds

    seeds = spawn_run_seeds(seed, n_runs)
    # Collect per-algorithm curves
    from banditbook.evaluation.runner import ExperimentResult, run_episode

    store = {
        a.get("label", a["name"]): {
            "regrets": [],
            "finals": [],
            "pulls": [],
            "rewards": [],
            "times": [],
            "dynamic_finals": [],
        }
        for a in algs
    }

    for run_seed in seeds:
        # One table for this repetition
        _, table_rng, _ = make_rng_pair(run_seed + 99991)
        if env_cfg["type"] == "oblivious_bernoulli":
            table = make_oblivious_bernoulli_table(
                np.asarray(env_cfg["probabilities"], dtype=float), horizon, table_rng
            )
        elif env_cfg["type"] == "switching":
            env_tmp = SwitchingObliviousBandit(
                env_cfg["n_arms"], horizon, env_cfg.get("switch_interval", 100), rng=table_rng
            )
            table = env_tmp.reward_table
        else:
            raise ValueError("shared table only for oblivious_bernoulli/switching")

        for alg in algs:
            name = alg["name"]
            label = alg.get("label", name)
            params = dict(alg.get("params", {}))
            if name in {"moss", "exp3", "exp3ix"} and "horizon" not in params:
                params["horizon"] = horizon
            if name == "ucb_delta" and "delta" not in params:
                params["delta"] = 1.0 / (horizon**2)
            agent_rng, env_rng, _ = make_rng_pair(run_seed)
            n_arms = table.shape[1]
            agent = build_agent(name, n_arms, agent_rng, params)
            env = ObliviousAdversarialBandit(table, rng=env_rng)
            agent.reset(agent_rng)
            env.reset(env_rng)
            log = run_episode(agent, env, horizon)
            from banditbook.evaluation.runner import _cumulative_dynamic, _cumulative_external
            from banditbook.metrics.regret import dynamic_regret as dyn_final

            if metric == "dynamic":
                curve = _cumulative_dynamic(log.actions, table)
                metric_name = "dynamic_regret"
            else:
                curve = _cumulative_external(log.actions, table)
                metric_name = "adversarial_external_regret"
            store[label]["regrets"].append(curve)
            store[label]["finals"].append(curve[-1])
            store[label]["pulls"].append(log.pull_counts)
            store[label]["rewards"].append(log.cumulative_reward)
            store[label]["times"].append(log.runtime_sec)
            store[label]["dynamic_finals"].append(dyn_final(log.actions, table))
            store[label]["metric_name"] = metric_name

    results = {}
    for label, s in store.items():
        results[label] = ExperimentResult(
            config={"horizon": horizon, "n_runs": n_runs, "seed": seed, "shared_table": True},
            seeds=list(seeds),
            regrets=np.asarray(s["regrets"]),
            final_regrets=np.asarray(s["finals"]),
            pull_counts=np.asarray(s["pulls"]),
            cumulative_rewards=np.asarray(s["rewards"]),
            runtimes=np.asarray(s["times"]),
            metric_name=s.get("metric_name", "adversarial_external_regret"),
            extras={"dynamic_final_regret": s["dynamic_finals"]},
        )
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run banditbook experiments")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--out", type=str, default="results/run")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    info = run_from_config(cfg, args.out)
    print(f"Wrote results to {info['out_dir']}")
    print(f"Plot: {info['plot']}")
    for name, res in info["results"].items():
        print(
            f"  {name}: mean final {res.metric_name}="
            f"{res.final_regrets.mean():.3f} ± "
            f"{(res.final_regrets.std(ddof=1)/np.sqrt(len(res.final_regrets)) if len(res.final_regrets)>1 else 0):.3f}"
        )


if __name__ == "__main__":
    main()
