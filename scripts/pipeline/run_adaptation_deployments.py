#!/usr/bin/env python
"""Run one arm of the adaptation experiment.

A deployment = one continuous real-market run: --pre-days calm, then a
persistent shift at day T = pre-days that lasts to the end of the run. Arms
sharing a --seed0 see identical markets (prices are exogenous), so runs are
paired; the oracle arm is the instant-adaptation normalizer that
results/plot_adaptation.py divides every capture curve by — run it for the same
seeds as every other arm.

Writes one CSV per seed to {out}/{shift}/{arm}-s{seed}.csv with per-day
profit/degradation/net plus whatever the arm logged nightly (NLL,
disagreement, adapted flag). wandb is off by default here — the CSVs are
the source of truth and runs are many and small; pass --wandb to log too.

Usage:
    uv run --extra train python scripts/pipeline/run_adaptation_deployments.py --shift fuel-step --arm oracle --seeds 10
    uv run --extra train python scripts/pipeline/run_adaptation_deployments.py --shift fuel-step --arm adaptive-dyna --seeds 10
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from energy_storage.adaptation import (
    AdaptiveDynaArm,
    AdaptiveDynaConfig,
    OnlineFinetuneArm,
    PolicyArm,
    run_stream,
)
from energy_storage.baselines import heuristic_policy, model_policy
from energy_storage.env import EnvConfig
from energy_storage.market.scenarios import ColdSnap, FuelShock, PlantOutage, Scenario
from energy_storage.market_history import MarketHistory
from energy_storage.oracle import RollingHorizonOracle

SHIFTS = ["fuel-step", "capacity-loss", "cold-regime"]
ARMS = ["oracle", "heuristic", "frozen-dyna", "frozen-ideal", "online-ft", "adaptive-dyna"]


def make_shift(kind: str, start_day: int, duration_days: int) -> Scenario:
    """Persistent shifts: the scenario window covers the rest of the run."""
    if kind == "fuel-step":
        return FuelShock(start_day=start_day, duration_days=duration_days, magnitude=2.0)
    if kind == "capacity-loss":
        # coal-1 is the cheap 100 MW baseload with a min-run block: losing it
        # permanently reshapes the merit order, not just the price level.
        return PlantOutage(
            start_day=start_day, generator_name="coal-1", duration_days=duration_days
        )
    if kind == "cold-regime":
        return ColdSnap(start_day=start_day, duration_days=duration_days)
    raise ValueError(f"unknown shift {kind!r}")


def build_arm(args: argparse.Namespace, config: EnvConfig, seed: int):
    if args.arm == "oracle":
        return PolicyArm(RollingHorizonOracle())
    if args.arm == "heuristic":
        return PolicyArm(heuristic_policy)
    if args.arm in ("frozen-dyna", "frozen-ideal"):
        from stable_baselines3 import SAC

        path = args.dyna_model if args.arm == "frozen-dyna" else args.ideal_model
        return PolicyArm(model_policy(SAC.load(path, device=args.device)))
    if args.arm == "online-ft":
        return OnlineFinetuneArm(
            args.ideal_model,
            config,
            gradient_steps_per_night=args.grad_steps,
            seed=seed,
            device=args.device,
        )
    # adaptive-dyna
    return AdaptiveDynaArm(
        args.dyna_model,
        args.market_model,
        MarketHistory.load(args.history),
        config,
        AdaptiveDynaConfig(
            window_days=args.window_days,
            ft_epochs=args.ft_epochs,
            sac_steps=args.sac_steps,
            nll_trigger=args.nll_trigger,
        ),
        seed=seed,
        device=args.device,
    )


def write_csv(path: Path, result, extra_keys: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["day", "profit", "degradation", "net", *extra_keys])
        for d in range(len(result.profit)):
            row = [d, result.profit[d], result.degradation[d], result.net[d]]
            for key in extra_keys:
                value = result.extras[key][d]
                row.append("" if np.isnan(value) else value)
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--shift", choices=SHIFTS, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seeds", type=int, default=10, help="number of paired deployments")
    parser.add_argument("--seed0", type=int, default=7_000_000)
    parser.add_argument("--pre-days", type=int, default=30)
    parser.add_argument("--post-days", type=int, default=90)
    parser.add_argument(
        "--dyna-model", type=Path, default=Path("models/dyna/dyna-d365-s0/best_model.zip")
    )
    parser.add_argument(
        "--ideal-model", type=Path, default=Path("models/sac-year-v3/best_model.zip")
    )
    parser.add_argument(
        "--market-model", type=Path, default=Path("models/market-model-d365-s0.pt")
    )
    parser.add_argument("--history", type=Path, default=Path("data/history-d365-s0.npz"))
    parser.add_argument("--out", type=Path, default=Path("results/adaptation"))
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/adaptation"),
        help="per-(shift,arm,seed) nightly checkpoint for resuming a killed/crashed "
        "run without redoing already-completed nights; deleted on completion",
    )
    parser.add_argument("--window-days", type=int, default=90)
    parser.add_argument("--ft-epochs", type=int, default=25)
    parser.add_argument(
        "--sac-steps", type=int, default=10_000, help="imagined SAC steps per adapting night"
    )
    parser.add_argument(
        "--nll-trigger",
        type=float,
        default=None,
        help="adaptive-dyna: adapt only above this per-day NLL; default adapts nightly",
    )
    parser.add_argument(
        "--grad-steps", type=int, default=240, help="online-ft: gradient steps per night"
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="torch device for arms that load a model; cpu by default because "
        "these are small MLPs and 'auto'/cuda only adds shared-GPU contention "
        "risk when many seeds run concurrently",
    )
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    horizon = args.pre_days + args.post_days
    config = EnvConfig(
        episode_days=horizon,
        scenario_sampler=lambda rng, start: [
            make_shift(args.shift, start + args.pre_days, args.post_days)
        ],
    )
    out_dir = args.out / args.shift
    out_dir.mkdir(parents=True, exist_ok=True)

    run = None
    if args.wandb:
        import wandb

        run = wandb.init(
            project="energy-storage",
            job_type="adaptation",
            config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        )

    for i in range(args.seeds):
        seed = args.seed0 + i
        arm = build_arm(args, config, seed)
        checkpoint_dir = args.checkpoint_dir / args.shift / f"{args.arm}-s{seed}"
        result = run_stream(arm, config, seed, checkpoint_dir=checkpoint_dir)
        extra_keys = sorted(result.extras)
        path = out_dir / f"{args.arm}-s{seed}.csv"
        write_csv(path, result, extra_keys)
        pre = result.net[: args.pre_days]
        post = result.net[args.pre_days :]
        print(
            f"{args.shift} {args.arm} seed {seed}: "
            f"pre-shift net {pre.sum():+8.2f}$ ({pre.mean():+.3f}$/day), "
            f"post-shift net {post.sum():+8.2f}$ ({post.mean():+.3f}$/day) -> {path}"
        )
        if run:
            import wandb

            wandb.log(
                {
                    "seed": seed,
                    "pre_net_per_day": float(pre.mean()),
                    "post_net_per_day": float(post.mean()),
                }
            )
    if run:
        run.finish()


if __name__ == "__main__":
    main()
