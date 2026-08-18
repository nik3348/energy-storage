"""Scenario robustness evaluation: paired same-seed counterfactuals.

For each policy and each scenario regime, runs the same seeded market weeks
with and without the shock (scenarios consume no randomness, so a scenario
episode and its calm twin differ only inside the shock window) and scores
everything against the rolling-horizon oracle evaluated under the *same*
regime. Reporting capture (policy net / oracle net) per regime separates
"the policy got worse" from "the market opportunity changed".
"""

from collections.abc import Callable

import numpy as np

from energy_storage.baselines import capture_jackknife, collect_episode
from energy_storage.env import EnvConfig
from energy_storage.market import ColdSnap, Drought, FuelShock, HeatWave, PlantOutage
from energy_storage.oracle import RollingHorizonOracle

ORACLE = "_oracle"

ScenarioSampler = Callable[[np.random.Generator, int], list]


def default_scenario_samplers(episode_days: int) -> dict[str, ScenarioSampler | None]:
    """One sampler per scenario type, each fully covering the episode
    (started early enough that ramp-in — and for drought, reservoir
    drawdown — has already happened), plus the calm control."""
    span = episode_days + 8

    return {
        "calm": None,
        "cold-snap": lambda rng, d: [ColdSnap(start_day=d - 4, duration_days=span)],
        "heat-wave": lambda rng, d: [HeatWave(start_day=d - 4, duration_days=span)],
        "fuel-shock": lambda rng, d: [FuelShock(start_day=d - 4, duration_days=span)],
        "drought": lambda rng, d: [Drought(start_day=d - 30, duration_days=span + 30)],
        "plant-outage": lambda rng, d: [PlantOutage(start_day=d - 4, duration_days=span)],
    }


def _with_sampler(config: EnvConfig, sampler: ScenarioSampler | None) -> EnvConfig:
    from dataclasses import replace

    return replace(config, scenario_sampler=sampler)


def evaluate_robustness(
    policies: dict[str, Callable],
    config: EnvConfig,
    episodes: int,
    seed0: int,
    samplers: dict[str, ScenarioSampler | None] | None = None,
    include_oracle: bool = True,
) -> dict[str, dict[str, np.ndarray]]:
    """Per-episode net profit for every (policy, regime) pair on paired seeds.

    Returns results[policy_name][regime_name] -> (episodes,) array of net
    $/episode. When include_oracle, the rolling-horizon oracle is added under
    the ORACLE key as the per-regime normalizer.
    """
    if samplers is None:
        samplers = default_scenario_samplers(config.episode_days)
    policies = dict(policies)
    if include_oracle:
        policies[ORACLE] = RollingHorizonOracle()

    results: dict[str, dict[str, np.ndarray]] = {name: {} for name in policies}
    for regime, sampler in samplers.items():
        regime_config = _with_sampler(config, sampler)
        for name, policy in policies.items():
            nets = np.empty(episodes)
            for ep in range(episodes):
                traj = collect_episode(policy, regime_config, seed=seed0 + ep)
                nets[ep] = traj["profit"].sum() - traj["degradation_cost"].sum()
            results[name][regime] = nets
    return results


def summarize(results: dict[str, dict[str, np.ndarray]]) -> list[dict]:
    """Flatten results into rows: mean±std net per regime, capture as a
    fraction of the same-regime oracle mean with jackknife std (None without
    the oracle)."""
    oracle = results.get(ORACLE)
    rows = []
    for name, by_regime in results.items():
        if name == ORACLE:
            continue
        for regime, nets in by_regime.items():
            capture = capture_std = None
            if oracle is not None and abs(oracle[regime].mean()) > 1e-9:
                capture, capture_std = capture_jackknife(nets, oracle[regime])
            rows.append(
                {
                    "policy": name,
                    "regime": regime,
                    "net_mean": float(nets.mean()),
                    "net_std": float(np.std(nets, ddof=1)) if len(nets) > 1 else 0.0,
                    "net_worst_decile": float(np.quantile(nets, 0.1)),
                    "oracle_net_mean": float(oracle[regime].mean()) if oracle else None,
                    "capture": capture,
                    "capture_std": capture_std,
                }
            )
    return rows


def format_table(rows: list[dict]) -> str:
    header = (
        f"{'policy':<14} {'regime':<13} {'net mean±std':>16} {'worst 10%':>10} "
        f"{'oracle':>10} {'capture±jk std':>15}"
    )
    lines = [header, "-" * len(header)]
    for r in rows:
        capture = (
            f"{100 * r['capture']:6.1f}±{100 * r['capture_std']:4.1f}%"
            if r["capture"] is not None
            else "            n/a"
        )
        oracle = f"{r['oracle_net_mean']:9.2f}$" if r["oracle_net_mean"] is not None else "      n/a"
        lines.append(
            f"{r['policy']:<14} {r['regime']:<13} {r['net_mean']:>7.2f}±{r['net_std']:<6.2f}$ "
            f"{r['net_worst_decile']:>9.2f}$ {oracle:>10} {capture:>15}"
        )
    return "\n".join(lines)
