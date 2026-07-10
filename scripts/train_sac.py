#!/usr/bin/env python
"""Train a SAC agent on the battery arbitrage environment.

Usage:
    uv run --extra train python scripts/train_sac.py
    uv run --extra train python scripts/train_sac.py --timesteps 500000 --n-envs 8

Trains, saves the best and final models to --models-dir, then evaluates the
agent against idle and rule-based-heuristic baselines on held-out seeds.
"""

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env

from energy_storage import BatteryArbitrageEnv, EnvConfig


def make_env_fn(episode_days: int):
    return lambda: BatteryArbitrageEnv(EnvConfig(episode_days=episode_days))


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


def evaluate(policy_fn, episode_days: int, episodes: int, seed0: int) -> dict:
    env = BatteryArbitrageEnv(EnvConfig(episode_days=episode_days))
    profits, degradations, rewards = [], [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed0 + ep)
        profit = degradation = total_reward = 0.0
        done = False
        while not done:
            obs, reward, terminated, truncated, info = env.step(policy_fn(env, obs))
            profit += info["profit"]
            degradation += info["degradation_cost"]
            total_reward += reward
            done = terminated or truncated
        profits.append(profit)
        degradations.append(degradation)
        rewards.append(total_reward)
    return {
        "profit": float(np.mean(profits)),
        "degradation": float(np.mean(degradations)),
        "net": float(np.mean(profits) - np.mean(degradations)),
        "reward": float(np.mean(rewards)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--episode-days", type=int, default=14)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    args = parser.parse_args()

    args.models_dir.mkdir(parents=True, exist_ok=True)

    train_env = make_vec_env(make_env_fn(args.episode_days), n_envs=args.n_envs, seed=args.seed)
    eval_env = make_vec_env(make_env_fn(args.episode_days), n_envs=1, seed=args.seed + 10_000)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(args.models_dir),
        n_eval_episodes=5,
        eval_freq=max(20_000 // args.n_envs, 1),
        verbose=1,
    )

    model = SAC(
        "MlpPolicy",
        train_env,
        seed=args.seed,
        verbose=1,
        learning_starts=2_000,
    )
    model.learn(total_timesteps=args.timesteps, callback=eval_callback, progress_bar=False)
    final_path = args.models_dir / "sac_final"
    model.save(final_path)
    print(f"\nSaved final model to {final_path}.zip")

    def sac_policy(env, obs):
        action, _ = model.predict(obs, deterministic=True)
        return action

    # Held-out evaluation seeds, disjoint from training and EvalCallback seeds.
    print(f"\nEvaluation over {args.eval_episodes} held-out episodes "
          f"({args.episode_days} days each):")
    header = f"{'policy':<12} {'profit':>10} {'degradation':>12} {'net':>10} {'reward':>10}"
    print(header)
    print("-" * len(header))
    for name, policy in [("idle", idle_policy), ("heuristic", heuristic_policy), ("sac", sac_policy)]:
        stats = evaluate(policy, args.episode_days, args.eval_episodes, seed0=1_000_000)
        print(
            f"{name:<12} {stats['profit']:>9.2f}$ {stats['degradation']:>11.2f}$ "
            f"{stats['net']:>9.2f}$ {stats['reward']:>10.3f}"
        )


if __name__ == "__main__":
    main()
