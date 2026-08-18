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

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env

from energy_storage import BatteryArbitrageEnv, EnvConfig
from energy_storage.baselines import evaluate, heuristic_policy, idle_policy, model_policy


def make_env_fn(config: EnvConfig):
    return lambda: BatteryArbitrageEnv(config)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--episode-days", type=int, default=14)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=20_000,
        help="training steps between EvalCallback evals; raise for long episodes",
    )
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    args.models_dir.mkdir(parents=True, exist_ok=True)

    run = None
    if args.wandb:
        import wandb

        run = wandb.init(
            project="energy-storage",
            job_type="train",
            config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            sync_tensorboard=True,
        )

    env_config = EnvConfig(episode_days=args.episode_days)
    train_env = make_vec_env(make_env_fn(env_config), n_envs=args.n_envs, seed=args.seed)
    eval_env = make_vec_env(make_env_fn(env_config), n_envs=1, seed=args.seed + 10_000)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(args.models_dir),
        n_eval_episodes=5,
        eval_freq=max(args.eval_freq // args.n_envs, 1),
        verbose=1,
    )

    model = SAC(
        "MlpPolicy",
        train_env,
        seed=args.seed,
        verbose=1,
        learning_starts=2_000,
        tensorboard_log=f"logs/{run.id}" if run else None,
    )
    callbacks = [eval_callback]
    if run:
        from wandb.integration.sb3 import WandbCallback

        callbacks.append(WandbCallback(verbose=1))
    model.learn(total_timesteps=args.timesteps, callback=callbacks, progress_bar=False)
    final_path = args.models_dir / "sac_final"
    model.save(final_path)
    print(f"\nSaved final model to {final_path}.zip")

    sac_policy = model_policy(model)

    # Held-out evaluation seeds, disjoint from training and EvalCallback seeds.
    print(f"\nEvaluation over {args.eval_episodes} held-out episodes "
          f"({args.episode_days} days each):")
    header = f"{'policy':<12} {'profit':>10} {'degradation':>12} {'net':>10} {'reward':>10}"
    print(header)
    print("-" * len(header))
    for name, policy in [("idle", idle_policy), ("heuristic", heuristic_policy), ("sac", sac_policy)]:
        stats = evaluate(policy, env_config, args.eval_episodes, seed0=1_000_000)
        print(
            f"{name:<12} {stats['profit']:>9.2f}$ {stats['degradation']:>11.2f}$ "
            f"{stats['net']:>9.2f}$ {stats['reward']:>10.3f}"
        )
        if run:
            import wandb

            wandb.log({f"final_eval/{name}_{k}": v for k, v in stats.items()})
    if run:
        run.finish()


if __name__ == "__main__":
    main()
