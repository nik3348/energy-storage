#!/usr/bin/env python
"""Deployed-policy infeasible-request rate: the bottom block of the safe-
exploration table (scripts/measure_safe_exploration.py has the top block,
the behaviours that could plausibly source real transitions during
training). This measures the fully trained, frozen checkpoints as actually
run at evaluation time, deterministically, on the real market.

Usage:
    uv run --extra train python scripts/measure_deployed_clipping.py
"""

import argparse

from stable_baselines3 import SAC

from energy_storage.baselines import model_policy
from energy_storage.env import BatteryArbitrageEnv, EnvConfig


def measure(policy_fn, config: EnvConfig, episodes: int, seed0: int) -> tuple[int, int, int]:
    """Return (clipped_steps, total_steps, near_bound_steps) over `episodes`
    paired episodes. near_bound counts steps where SoC (before the action)
    sits within 0.1 of a bound, the "parked" behaviour the clipping rate is
    attributed to."""
    env = BatteryArbitrageEnv(config)
    clipped = total = near_bound = 0
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed0 + ep)
        done = False
        while not done:
            soc = env.battery.soc
            near_bound += int(min(abs(soc - config.battery.soc_min), abs(soc - config.battery.soc_max)) < 0.1)
            obs, _, terminated, truncated, info = env.step(policy_fn(env, obs))
            clipped += int(info["constraint_clipped"])
            total += 1
            done = terminated or truncated
    return clipped, total, near_bound


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--episode-days", type=int, default=14)
    parser.add_argument("--seed0", type=int, default=1_000_000)
    parser.add_argument("--budget-days", type=int, default=365)
    parser.add_argument("--dyna", default="models/dyna/dyna-d365-s0/best_model.zip")
    parser.add_argument("--replay", default="models/dyna/replay-d365-s0/best_model.zip")
    parser.add_argument("--unlimited", default="models/sac-year-v3/best_model.zip")
    args = parser.parse_args()

    config = EnvConfig(episode_days=args.episode_days)
    policies = {
        "dyna-365": model_policy(SAC.load(args.dyna)),
        "replay-365": model_policy(SAC.load(args.replay)),
        "unlimited-data": model_policy(SAC.load(args.unlimited)),
    }

    header = f"{'policy':<16} {'infeasible req.':>15} {'rate':>8} {'per 365d budget':>16} {'near-bound':>11}"
    print(f"Deployed-policy infeasible on-asset action requests "
          f"({args.episodes} x {args.episode_days}-day paired episodes, deterministic)\n")
    print(header)
    print("-" * len(header))
    budget_steps = args.budget_days * 24
    for name, policy in policies.items():
        clipped, total, near_bound = measure(policy, config, args.episodes, args.seed0)
        rate = clipped / total if total else 0.0
        near_rate = near_bound / total if total else 0.0
        print(f"{name:<16} {clipped:>8}/{total:<6} {rate:>7.1%} {rate * budget_steps:>15.0f} {near_rate:>10.1%}")


if __name__ == "__main__":
    main()
