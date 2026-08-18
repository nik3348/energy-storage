#!/usr/bin/env python
"""Per-scenario zero-shot robustness table for the budget-trained policies.

Loads the frozen D=365 dyna, replay, and unlimited-data (sac-year-v3) checkpoints
and scores each against the rolling-horizon oracle on paired same-seed episodes
under the calm control and the five shock regimes (energy_storage/robustness.py).
None of these policies saw any shock during training, so this is a zero-shot test.

Usage:
    uv run --extra train python scripts/measure_scenario_robustness.py
"""

import argparse

from stable_baselines3 import SAC

from energy_storage.baselines import model_policy
from energy_storage.env import EnvConfig
from energy_storage.robustness import evaluate_robustness, format_table, summarize


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--episode-days", type=int, default=14)
    parser.add_argument("--seed0", type=int, default=1_000_000)
    parser.add_argument("--dyna", default="models/dyna/dyna-d365-s0/best_model.zip")
    parser.add_argument("--replay", default="models/dyna/replay-d365-s0/best_model.zip")
    parser.add_argument("--unlimited", default="models/sac-year-v3/best_model.zip")
    args = parser.parse_args()

    policies = {
        "dyna-365": model_policy(SAC.load(args.dyna)),
        "replay-365": model_policy(SAC.load(args.replay)),
        "unlimited-data": model_policy(SAC.load(args.unlimited)),
    }

    config = EnvConfig(episode_days=args.episode_days)
    results = evaluate_robustness(policies, config, args.episodes, args.seed0)
    rows = summarize(results)
    print(format_table(rows))

    print("\ncapture by policy x regime (%):")
    regimes = list(results[next(iter(policies))].keys())
    header = f"{'policy':<16}" + "".join(f"{r:>13}" for r in regimes)
    print(header)
    for name in policies:
        by_regime = {row["regime"]: row["capture"] for row in rows if row["policy"] == name}
        line = f"{name:<16}" + "".join(
            f"{100 * by_regime[r]:12.1f}%" if by_regime.get(r) is not None else f"{'n/a':>13}"
            for r in regimes
        )
        print(line)


if __name__ == "__main__":
    main()
