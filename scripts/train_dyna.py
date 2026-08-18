#!/usr/bin/env python
"""Run one arm of the Dyna sample-efficiency experiment.

Every arm sees exactly --budget-days D of real market history:
  online  standard SAC on the real simulator, stopped after 24*D env steps
  replay  SAC on the D stored days replayed verbatim (unlimited epochs)
  dyna    SAC on synthetic days from the fitted market model, with
          rolling-horizon planner transitions on the stored days seeded
          into the replay buffer

The history file is shared across arms (collected on first use); the dyna
arm refuses to run without a model saved by validate_market_model.py, which
is what enforces the validation gate. During training, checkpoint selection
(EvalCallback) uses only the arm's own data source — the real simulator is
touched exclusively by the final held-out evaluation.

Usage:
    uv run --extra train python scripts/train_dyna.py --arm dyna --budget-days 365
"""

import argparse
from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env

from energy_storage import BatteryArbitrageEnv, EnvConfig
from energy_storage.adaptation import planner_transitions, seed_replay_buffer
from energy_storage.baselines import (
    evaluate,
    heuristic_policy,
    idle_policy,
    model_policy,
)
from energy_storage.env import default_market_config
from energy_storage.market_history import MarketHistory, ReplayedMarket
from energy_storage.market_model import LearnedMarket, MarketModelEnsemble
from energy_storage.oracle import evaluate_hindsight


def make_env_fn(config: EnvConfig):
    return lambda: BatteryArbitrageEnv(config)


def load_or_collect_history(path: Path, days: int, seed: int) -> MarketHistory:
    if path.exists():
        history = MarketHistory.load(path)
        assert len(history) == days, (
            f"{path} holds {len(history)} days, expected {days}"
        )
        return history
    history = MarketHistory.collect(default_market_config(), seed=seed, days=days)
    path.parent.mkdir(parents=True, exist_ok=True)
    history.save(path)
    print(f"collected and wrote {path}")
    return history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=["online", "replay", "dyna"], required=True)
    parser.add_argument(
        "--budget-days",
        type=int,
        default=365,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--episode-days",
        type=int,
        default=14,
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=500_000,
        help="gradient budget for replay/dyna; the online arm is capped at 24*D regardless",
    )
    parser.add_argument(
        "--history",
        type=Path,
        default=None,
        help="default data/history-d{D}-s{seed}.npz",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="gated ensemble from validate_market_model.py; default models/market-model-d{D}-s{seed}.pt",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=None,
        help="default: timesteps/25",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help="default models/dyna/{arm}-d{D}-s{seed}",
    )
    parser.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    d = args.budget_days
    history_path = args.history or Path("data") / f"history-d{d}-s{args.seed}.npz"
    models_dir = (
        args.models_dir or Path("models") / "dyna" / f"{args.arm}-d{d}-s{args.seed}"
    )
    models_dir.mkdir(parents=True, exist_ok=True)

    real_config = EnvConfig(episode_days=args.episode_days)

    if args.arm == "online":
        timesteps = 24 * d
        train_config = real_config
    elif args.arm == "replay":
        timesteps = args.timesteps
        history = load_or_collect_history(history_path, d, args.seed)
        train_config = EnvConfig(
            episode_days=args.episode_days,
            market_factory=lambda mc, seed: ReplayedMarket(history, mc),
        )
    else:  # dyna
        timesteps = args.timesteps
        history = load_or_collect_history(history_path, d, args.seed)
        model_path = args.model or Path("models") / f"market-model-d{d}-s{args.seed}.pt"
        if not model_path.exists():
            raise SystemExit(
                f"{model_path} not found — run validate_market_model.py for this "
                "budget/seed first; the dyna arm only trains on a gated model."
            )
        ensemble = MarketModelEnsemble.load(model_path)
        train_config = EnvConfig(
            episode_days=args.episode_days,
            market_factory=lambda mc, seed: LearnedMarket(ensemble, mc, seed=seed),
        )

    run = None
    if args.wandb:
        import wandb

        run = wandb.init(
            project="energy-storage",
            job_type="train-dyna",
            config={
                k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
            }
            | {"timesteps_effective": timesteps},
            sync_tensorboard=True,
        )

    train_env = make_vec_env(
        make_env_fn(train_config), n_envs=args.n_envs, seed=args.seed
    )
    eval_env = make_vec_env(
        make_env_fn(train_config), n_envs=1, seed=args.seed + 10_000
    )
    eval_freq = args.eval_freq or max(timesteps // 25, 500)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(models_dir),
        n_eval_episodes=5,
        eval_freq=max(eval_freq // args.n_envs, 1),
        verbose=1,
    )

    # Warmup scales with the step budget (same rule for every arm) so the
    # small online budgets aren't consumed entirely by random-action warmup.
    learning_starts = min(2_000, max(timesteps // 10, 100))
    model = SAC(
        "MlpPolicy",
        train_env,
        seed=args.seed,
        verbose=1,
        learning_starts=learning_starts,
        tensorboard_log=f"logs/{run.id}" if run else None,
    )

    if args.arm == "dyna":
        replay_config = EnvConfig(
            episode_days=args.episode_days,
            market_factory=lambda mc, seed: ReplayedMarket(history, mc),
        )
        print("seeding replay buffer with rolling-horizon planner transitions...")
        transitions = planner_transitions(replay_config, d, seed0=args.seed + 500_000)
        n = seed_replay_buffer(model, transitions, args.n_envs)
        print(f"seeded {n} planner transitions on the {d} stored days")

    callbacks = [eval_callback]
    if run:
        from wandb.integration.sb3 import WandbCallback

        callbacks.append(WandbCallback(verbose=1))
    model.learn(total_timesteps=timesteps, callback=callbacks, progress_bar=False)
    final_path = models_dir / "sac_final"
    model.save(final_path)
    print(f"\nSaved final model to {final_path}.zip")

    # Held-out evaluation on the REAL simulator — identical across arms.
    print(
        f"\nReal-market evaluation, {args.eval_episodes} paired episodes "
        f"({args.episode_days} days each):"
    )
    hindsight = evaluate_hindsight(
        real_config, episodes=args.eval_episodes, seed0=1_000_000
    )
    policies = [
        ("idle", idle_policy),
        ("heuristic", heuristic_policy),
        ("final", model_policy(model)),
    ]
    best_path = models_dir / "best_model.zip"
    if best_path.exists():
        policies.append(("best", model_policy(SAC.load(best_path))))
    header = f"{'policy':<12} {'net':>10} {'% hindsight':>12}"
    print(f"hindsight net: ${hindsight['net']:.2f}")
    print(header)
    print("-" * len(header))
    for name, policy in policies:
        stats = evaluate(policy, real_config, args.eval_episodes, seed0=1_000_000)
        pct = (
            100.0 * stats["net"] / hindsight["net"]
            if hindsight["net"]
            else float("nan")
        )
        print(f"{name:<12} {stats['net']:>9.2f}$ {pct:>11.1f}%")
        if run:
            import wandb

            wandb.log(
                {f"real_eval/{name}_{k}": v for k, v in stats.items()}
                | {f"real_eval/{name}_pct_hindsight": pct}
            )
    if run:
        import wandb

        wandb.log({"real_eval/hindsight_net": hindsight["net"]})
        run.finish()


if __name__ == "__main__":
    main()
