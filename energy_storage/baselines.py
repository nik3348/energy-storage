"""Baseline policies and rollout helpers shared by training and diagnostics."""

from collections import defaultdict

import numpy as np

from energy_storage.env import BatteryArbitrageEnv, EnvConfig


def idle_policy(env, obs):
    return np.array([0.0], dtype=np.float32)


def heuristic_policy(env, obs):
    """Charge in the overnight trough, discharge into the evening peak."""
    if 1 <= env.hour <= 5:
        return np.array([1.0], dtype=np.float32)
    if 18 <= env.hour <= 21:
        return np.array([-1.0], dtype=np.float32)
    return np.array([0.0], dtype=np.float32)


def model_policy(model):
    """Wrap an SB3-style model (has .predict) as a policy_fn(env, obs)."""

    def policy(env, obs):
        action, _ = model.predict(obs, deterministic=True)
        return action

    return policy


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
    "day",
)


def capture_jackknife(nets: np.ndarray, reference_nets: np.ndarray) -> tuple[float, float]:
    """Pooled capture (mean policy net / mean reference net) with a
    leave-one-episode-out jackknife std.

    Capture is reported as a ratio of means rather than a mean of per-episode
    ratios because a single near-zero reference episode makes per-episode
    ratios explode. The matched error margin for such a pooled ratio is the
    jackknife: recompute it with each paired episode left out, and scale the
    spread of those estimates by sqrt((n-1)^2/n).
    """
    nets, reference_nets = np.asarray(nets), np.asarray(reference_nets)
    capture = float(nets.mean() / reference_nets.mean())
    n = len(nets)
    if n < 2:
        return capture, 0.0
    idx = np.arange(n)
    theta = np.array(
        [nets[idx != i].mean() / reference_nets[idx != i].mean() for i in range(n)]
    )
    std = float(np.sqrt((n - 1) / n * np.sum((theta - theta.mean()) ** 2)))
    return capture, std


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
    """Mean per-episode economics over several seeded episodes. `net_std` is
    the per-episode std (ddof=1) and `nets` the per-episode array, for callers
    reporting error margins over the paired episodes."""
    profits, degradations, rewards = [], [], []
    for ep in range(episodes):
        traj = collect_episode(policy_fn, config, seed=seed0 + ep)
        profits.append(traj["profit"].sum())
        degradations.append(traj["degradation_cost"].sum())
        rewards.append(traj["reward"].sum())
    nets = np.asarray(profits) - np.asarray(degradations)
    return {
        "profit": float(np.mean(profits)),
        "degradation": float(np.mean(degradations)),
        "net": float(np.mean(nets)),
        "net_std": float(np.std(nets, ddof=1)) if len(nets) > 1 else 0.0,
        "reward": float(np.mean(rewards)),
        "nets": nets,
    }
