#!/usr/bin/env python
"""Table VII: infeasible on-asset action requests, both blocks.

The environment clips every action, so no *hard* constraint violation is ever
executed. What is measurable, and what matters for safe exploration, is how
often a policy *requests* an infeasible action, one that has to be throttled by
an SoC bound (charging past soc_max or discharging below soc_min). On a real
asset every such request is an action the operator's safety layer must refuse.

Top block: training-time behaviours, the candidates for what could plausibly
source real transitions during training.

  random     uniform actions, the exploration a model-free agent does on-asset,
             especially during its warmup
  planner    the rolling-horizon DP controller that supplies the Dyna arm's
             real-asset transitions (feasible by construction: expected 0)
  heuristic  the fixed charge-overnight / discharge-evening rule
  idle       never acts (trivially 0)

The point of the Dyna + planner-demonstration design is that the asset only
ever executes the planner, so its on-asset infeasible-request rate is zero; a
model-free agent trained on the asset pays the random rate over its whole
budget.

Bottom block: the fully trained, frozen checkpoints as actually run at
evaluation time, deterministically, on the real market. Also reports the
near-bound rate, the share of steps where SoC (before the action) already sits
within 0.1 of a bound, which is what the clipping rate is attributed to.

Usage:
    uv run --extra train python scripts/results/measure_safe_exploration.py
"""

import argparse

import numpy as np
from stable_baselines3 import SAC

from energy_storage.baselines import heuristic_policy, idle_policy, model_policy
from energy_storage.env import BatteryArbitrageEnv, EnvConfig
from energy_storage.oracle import RollingHorizonOracle


def random_policy_factory(seed: int):
    rng = np.random.default_rng(seed)

    def policy(env, obs):
        return np.array([rng.uniform(-1.0, 1.0)], dtype=np.float32)

    return policy


def measure_training(policy_fn, config: EnvConfig, episodes: int, seed0: int) -> np.ndarray:
    """Per-episode infeasible-request rates over `episodes` paired episodes."""
    env = BatteryArbitrageEnv(config)
    rates = np.empty(episodes)
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed0 + ep)
        clipped = total = 0
        done = False
        while not done:
            obs, _, terminated, truncated, info = env.step(policy_fn(env, obs))
            clipped += int(info["constraint_clipped"])
            total += 1
            done = terminated or truncated
        rates[ep] = clipped / total
    return rates


def measure_deployed(
    policy_fn, config: EnvConfig, episodes: int, seed0: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per-episode (clipped_rate, near_bound_rate) arrays over `episodes`
    paired episodes."""
    env = BatteryArbitrageEnv(config)
    clip_rates = np.empty(episodes)
    near_rates = np.empty(episodes)
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed0 + ep)
        clipped = total = near_bound = 0
        done = False
        while not done:
            soc = env.battery.soc
            near_bound += int(
                min(abs(soc - config.battery.soc_min), abs(soc - config.battery.soc_max)) < 0.1
            )
            obs, _, terminated, truncated, info = env.step(policy_fn(env, obs))
            clipped += int(info["constraint_clipped"])
            total += 1
            done = terminated or truncated
        clip_rates[ep] = clipped / total
        near_rates[ep] = near_bound / total
    return clip_rates, near_rates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--episode-days", type=int, default=14)
    parser.add_argument("--seed0", type=int, default=1_000_000)
    parser.add_argument(
        "--budget-days", type=int, default=365,
        help="report absolute infeasible requests over this on-asset budget",
    )
    parser.add_argument("--dyna", default="models/dyna/dyna-d365-s0/best_model.zip")
    parser.add_argument("--replay", default="models/dyna/replay-d365-s0/best_model.zip")
    parser.add_argument("--unlimited", default="models/sac-year-v3/best_model.zip")
    args = parser.parse_args()

    config = EnvConfig(episode_days=args.episode_days)
    budget_steps = args.budget_days * 24

    print(f"Table VII, top block: training-time behaviours "
          f"({args.episodes} x {args.episode_days}-day paired episodes)\n")
    header = f"{'policy':<10} {'rate mean±std':>16} {'per 365d budget':>16}"
    print(header)
    print("-" * len(header))
    training_policies = {
        "random": random_policy_factory(0),
        "planner": RollingHorizonOracle(),
        "heuristic": heuristic_policy,
        "idle": idle_policy,
    }
    for name, policy in training_policies.items():
        rates = measure_training(policy, config, args.episodes, args.seed0)
        mean, std = rates.mean(), rates.std(ddof=1)
        print(f"{name:<10} {100 * mean:>9.1f}±{100 * std:<4.1f}% {mean * budget_steps:>15.0f}")

    print(f"\nTable VII, bottom block: deployed policies "
          f"({args.episodes} x {args.episode_days}-day paired episodes, deterministic)\n")
    header = f"{'policy':<16} {'rate mean±std':>16} {'per 365d budget':>16} {'near-bound mean±std':>20}"
    print(header)
    print("-" * len(header))
    deployed_policies = {
        "dyna-365": model_policy(SAC.load(args.dyna)),
        "replay-365": model_policy(SAC.load(args.replay)),
        "unlimited-data": model_policy(SAC.load(args.unlimited)),
    }
    for name, policy in deployed_policies.items():
        clip_rates, near_rates = measure_deployed(policy, config, args.episodes, args.seed0)
        print(
            f"{name:<16} {100 * clip_rates.mean():>9.1f}±{100 * clip_rates.std(ddof=1):<4.1f}% "
            f"{clip_rates.mean() * budget_steps:>15.0f} "
            f"{100 * near_rates.mean():>13.1f}±{100 * near_rates.std(ddof=1):<4.1f}%"
        )


if __name__ == "__main__":
    main()
