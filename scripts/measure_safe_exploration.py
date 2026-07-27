#!/usr/bin/env python
"""Safe-exploration measurement: infeasible on-asset action requests per policy.

The environment clips every action, so no *hard* constraint violation is ever
executed. What is measurable, and what matters for safe exploration, is how
often a policy *requests* an infeasible action, one that has to be throttled by
an SoC bound (charging past soc_max or discharging below soc_min). On a real
asset every such request is an action the operator's safety layer must refuse.

This script counts the constraint-bound request rate for the behaviours that
would drive the physical battery in each experiment arm:

  random     uniform actions, the exploration a model-free agent does on-asset,
             especially during its warmup
  planner    the rolling-horizon DP controller that supplies the Dyna arm's
             real-asset transitions (feasible by construction: expected 0)
  heuristic  the fixed charge-overnight / discharge-evening rule
  idle       never acts (trivially 0)

The point of the Dyna + planner-demonstration design is that the asset only ever
executes the planner, so its on-asset infeasible-request rate is zero; a
model-free agent trained on the asset pays the random rate over its whole budget.

Usage:
    uv run python scripts/measure_safe_exploration.py
"""

import argparse

import numpy as np

from energy_storage.baselines import heuristic_policy, idle_policy
from energy_storage.env import BatteryArbitrageEnv, EnvConfig
from energy_storage.oracle import RollingHorizonOracle


def random_policy_factory(seed: int):
    rng = np.random.default_rng(seed)

    def policy(env, obs):
        return np.array([rng.uniform(-1.0, 1.0)], dtype=np.float32)

    return policy


def measure(policy_fn, config: EnvConfig, episodes: int, seed0: int) -> tuple[int, int]:
    """Return (clipped_steps, total_steps) over `episodes` paired episodes."""
    env = BatteryArbitrageEnv(config)
    clipped = total = 0
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed0 + ep)
        done = False
        while not done:
            obs, _, terminated, truncated, info = env.step(policy_fn(env, obs))
            clipped += int(info["constraint_clipped"])
            total += 1
            done = terminated or truncated
    return clipped, total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--episode-days", type=int, default=14)
    parser.add_argument("--seed0", type=int, default=1_000_000)
    parser.add_argument("--budget-days", type=int, default=365,
                        help="report absolute infeasible requests over this on-asset budget")
    args = parser.parse_args()

    config = EnvConfig(episode_days=args.episode_days)
    policies = {
        "random": random_policy_factory(0),
        "planner": RollingHorizonOracle(),
        "heuristic": heuristic_policy,
        "idle": idle_policy,
    }

    header = f"{'policy':<10} {'infeasible req.':>15} {'rate':>8} {'per 365d budget':>16}"
    print(f"Safe exploration: infeasible on-asset action requests "
          f"({args.episodes} x {args.episode_days}-day paired episodes)\n")
    print(header)
    print("-" * len(header))
    budget_steps = args.budget_days * 24
    for name, policy in policies.items():
        clipped, total = measure(policy, config, args.episodes, args.seed0)
        rate = clipped / total if total else 0.0
        print(f"{name:<10} {clipped:>8}/{total:<6} {rate:>7.1%} {rate * budget_steps:>15.0f}")


if __name__ == "__main__":
    main()
