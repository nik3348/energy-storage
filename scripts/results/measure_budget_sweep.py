#!/usr/bin/env python
"""Score every budget-sweep checkpoint against the reference policies, with
error margins, and write results/budget_sweep.json.

Doubles as the benchmark ladder: run with no checkpoints trained yet and it
still reports the oracle, heuristic, and idle rows, skipping every learned
arm whose checkpoint is missing. Capture is the pooled fraction of the
per-episode hindsight optimum, with a leave-one-episode-out jackknife std
(energy_storage.baselines.capture_jackknife).

Usage:
    uv run --extra train python scripts/results/measure_budget_sweep.py
"""

import argparse
import json
from pathlib import Path

from stable_baselines3 import SAC

from energy_storage.baselines import (
    capture_jackknife,
    evaluate,
    heuristic_policy,
    idle_policy,
    model_policy,
)
from energy_storage.env import EnvConfig
from energy_storage.oracle import RollingHorizonOracle, evaluate_hindsight

BUDGETS = [90, 365, 1095]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--episode-days", type=int, default=14)
    parser.add_argument("--seed0", type=int, default=1_000_000)
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--checkpoint", default="best_model.zip",
                        help="which checkpoint per arm; the figure uses each arm's own best")
    parser.add_argument("--out", type=Path, default=Path("results/budget_sweep.json"))
    args = parser.parse_args()

    config = EnvConfig(episode_days=args.episode_days)
    hindsight = evaluate_hindsight(config, args.episodes, args.seed0)
    print(f"hindsight net ${hindsight['net']:.2f}±{hindsight['net_std']:.2f}/episode = 100%\n")

    # The three budget-swept arms, plus the fixed reference lines the figure draws.
    # Any checkpoint not yet trained is skipped rather than failing outright, so
    # this also works as a no-training smoke test of the reference policies.
    arms: dict[str, Path] = {}
    for budget in BUDGETS:
        for arm in ("online", "replay", "dyna"):
            path = args.models / "dyna" / f"{arm}-d{budget}-s0" / args.checkpoint
            if path.exists():
                arms[f"{arm}-d{budget}"] = path
    unlimited = args.models / "sac-year-v3" / args.checkpoint
    if unlimited.exists():
        arms["unlimited-data"] = unlimited
    else:
        print(f"skipping unlimited-data: no checkpoint at {unlimited}")

    out: dict[str, dict] = {
        "episodes": args.episodes,
        "episode_days": args.episode_days,
        "seed0": args.seed0,
        "checkpoint": args.checkpoint,
        "hindsight_net": hindsight["net"],
        "hindsight_net_std": hindsight["net_std"],
        "hindsight_nets": [float(v) for v in hindsight["nets"]],
        "arms": {},
    }

    header = f"{'arm':<18} {'net $/episode':>18} {'capture (of hindsight)':>24}"
    print(header)
    print("-" * len(header))

    def score(name: str, policy) -> None:
        r = evaluate(policy, config, args.episodes, args.seed0)
        capture, std = capture_jackknife(r["nets"], hindsight["nets"])
        out["arms"][name] = {
            "net": r["net"],
            "net_std": r["net_std"],
            "capture_pct": 100 * capture,
            "capture_std_pct": 100 * std,
            # Per-episode nets are kept so paired between-arm differences can be
            # recomputed offline. Arms share seeds, so the paired difference,
            # not the overlap of two marginal intervals, is the right test.
            "nets": [float(v) for v in r["nets"]],
        }
        print(
            f"{name:<18} {r['net']:>9.2f}±{r['net_std']:<7.2f} "
            f"{100 * capture:>17.1f}±{100 * std:<4.1f}%"
        )

    for name, path in arms.items():
        score(name, model_policy(SAC.load(path)))
    score("rolling-oracle", RollingHorizonOracle())
    score("heuristic", heuristic_policy)
    score("idle", idle_policy)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
