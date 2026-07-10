"""Baseline policies and rollout helpers shared by training and diagnostics."""

from collections import defaultdict

import numpy as np

from energy_storage.env import BatteryArbitrageEnv, EnvConfig


def idle_policy(env, obs):
    return np.array([0.0], dtype=np.float32)


def heuristic_policy(env, obs):
    """Charge in the overnight trough, discharge into the evening peak."""
    hour = env._hour
    if 1 <= hour <= 5:
        return np.array([1.0], dtype=np.float32)
    if 18 <= hour <= 21:
        return np.array([-1.0], dtype=np.float32)
    return np.array([0.0], dtype=np.float32)


TRACKED_KEYS = (
    "price",
    "profit",
    "degradation_cost",
    "soc",
    "soh",
    "grid_energy_kwh",
    "cycle_loss",
    "calendar_loss",
    "hour",
)


def collect_episode(policy_fn, config: EnvConfig, seed: int) -> dict[str, np.ndarray]:
    """Roll out one episode, returning per-hour trajectories."""
    env = BatteryArbitrageEnv(config)
    obs, _ = env.reset(seed=seed)
    record = defaultdict(list)
    done = False
    while not done:
        obs, reward, terminated, truncated, info = env.step(policy_fn(env, obs))
        for key in TRACKED_KEYS:
            record[key].append(info[key])
        record["reward"].append(reward)
        done = terminated or truncated
    return {key: np.asarray(values) for key, values in record.items()}


def evaluate(policy_fn, config: EnvConfig, episodes: int, seed0: int) -> dict[str, float]:
    """Mean per-episode economics over several seeded episodes."""
    profits, degradations, rewards = [], [], []
    for ep in range(episodes):
        traj = collect_episode(policy_fn, config, seed=seed0 + ep)
        profits.append(traj["profit"].sum())
        degradations.append(traj["degradation_cost"].sum())
        rewards.append(traj["reward"].sum())
    return {
        "profit": float(np.mean(profits)),
        "degradation": float(np.mean(degradations)),
        "net": float(np.mean(profits) - np.mean(degradations)),
        "reward": float(np.mean(rewards)),
    }
