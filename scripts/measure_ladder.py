#!/usr/bin/env python
"""The benchmark ladder with error margins: every reference policy and trained
checkpoint on the same 20 paired held-out episodes, reported as net $/episode
(mean ± std over episodes) and as capture, the pooled fraction of the
per-episode hindsight optimum, with a leave-one-episode-out jackknife std.

Usage:
    uv run --extra train python scripts/measure_ladder.py
"""

import argparse
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--episode-days", type=int, default=14)
    parser.add_argument("--seed0", type=int, default=1_000_000)
    parser.add_argument("--dyna", default="models/dyna/dyna-d365-s0/best_model.zip")
    parser.add_argument("--replay", default="models/dyna/replay-d365-s0/best_model.zip")
    parser.add_argument("--unlimited", default="models/sac-year-v3/best_model.zip")
    args = parser.parse_args()

    config = EnvConfig(episode_days=args.episode_days)
    hindsight = evaluate_hindsight(config, args.episodes, args.seed0)

    # Reference policies need no training, so the ladder still runs (without the
    # learned rows) for anyone who has the source but not the checkpoints.
    policies = {"rolling-oracle": RollingHorizonOracle()}
    for name, path in (
        ("dyna-365", args.dyna),
        ("replay-365", args.replay),
        ("unlimited-data", args.unlimited),
    ):
        if Path(path).exists():
            policies[name] = model_policy(SAC.load(path))
        else:
            print(f"skipping {name}: no checkpoint at {path}")
    policies["heuristic"] = heuristic_policy
    policies["idle"] = idle_policy

    print(
        f"Benchmark ladder, {args.episodes} paired {args.episode_days}-day episodes, "
        f"seed0={args.seed0}\n"
    )
    header = f"{'policy':<16} {'net $/episode':>18} {'capture (of hindsight)':>24}"
    print(header)
    print("-" * len(header))
    print(
        f"{'hindsight':<16} {hindsight['net']:>9.2f}±{hindsight['net_std']:<7.2f} "
        f"{'100.0% (definition)':>24}"
    )
    for name, policy in policies.items():
        r = evaluate(policy, config, args.episodes, args.seed0)
        capture, std = capture_jackknife(r["nets"], hindsight["nets"])
        print(
            f"{name:<16} {r['net']:>9.2f}±{r['net_std']:<7.2f} "
            f"{100 * capture:>17.1f}±{100 * std:<4.1f}%"
        )


if __name__ == "__main__":
    main()
