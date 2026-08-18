#!/usr/bin/env python
"""Offline replay of the adaptive-dyna nightly ensemble refit, re-gated after
every night against the same calibration statistic (mean daily spread) and
the same held-out real year used to certify the original D=365 model
(Section "The Validation Gate and the Data Threshold").

Prices are exogenous, so this reuses the exact fuel-step deployment stream at
--seed (matching run_adaptation.py's default seed0, i.e. the first paired
deployment) with a cheap idle policy standing in for whatever the real arm
would do: the nightly refit depends only on observed prices, not on the
policy's actions. This skips the SAC fine-tune and the planner-demo refresh
(irrelevant to the ensemble's calibration), which is why this reproduces the
Appendix drift table far more cheaply than an actual adaptive-dyna deployment.

Usage:
    uv run --extra train python scripts/measure_ensemble_drift.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from validate_market_model import gates, stats, synthetic_year  # noqa: E402

from energy_storage.baselines import idle_policy
from energy_storage.env import BatteryArbitrageEnv, EnvConfig, default_market_config
from energy_storage.market.scenarios import FuelShock
from energy_storage.market_history import MarketHistory, build_dataset
from energy_storage.market_model import MarketModelEnsemble

CHECKPOINT_NIGHTS = [0, 1, 2, 3, 5, 10, 20, 30, 40, 60, 90, 119]


def refit_step(ensemble, history, seam, night, window_days, anchor_days, ft_epochs, ft_lr, cap):
    """Exactly AdaptiveDynaArm.end_of_day's ensemble-refit block (adaptation.py),
    with the SAC fine-tune and planner-demo refresh omitted (they don't affect
    the ensemble's calibration)."""
    x_all, y_all = build_dataset(history, cap)
    rows = np.arange(len(x_all))
    rows = rows[rows != seam - 1]
    recent = rows[-window_days:]
    pool = rows[rows < seam - 1]
    rng = np.random.default_rng(1000 + night)
    n_anchor = min(anchor_days, len(pool))
    anchors = rng.choice(pool, size=n_anchor, replace=False) if n_anchor else np.empty(0, dtype=int)
    sel = np.unique(np.concatenate([recent, anchors]))
    ensemble.fine_tune(x_all[sel], y_all[sel], epochs=ft_epochs, lr=ft_lr, seed=night)


def gate_ensemble(ensemble, market_config, cap, real_spread, lo, hi, seed_base, n_years) -> float:
    years = [
        synthetic_year(ensemble, market_config, seed=seed_base + i) for i in range(n_years)
    ]
    synth = MarketHistory(
        prices=np.concatenate([y.prices for y in years]),
        day_of_year=np.concatenate([y.day_of_year for y in years]),
        is_weekend=np.concatenate([y.is_weekend for y in years]),
        is_holiday=np.concatenate([y.is_holiday for y in years]),
    )
    return stats(synth, cap)["mean_daily_spread"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7_000_000)
    parser.add_argument("--pre-days", type=int, default=30)
    parser.add_argument("--post-days", type=int, default=90)
    parser.add_argument("--history", type=Path, default=Path("data/history-d365-s0.npz"))
    parser.add_argument("--ensemble", type=Path, default=Path("models/market-model-d365-s0.pt"))
    parser.add_argument("--window-days", type=int, default=90)
    parser.add_argument("--anchor-days", type=int, default=30)
    parser.add_argument("--ft-epochs", type=int, default=25)
    parser.add_argument("--ft-lr", type=float, default=3e-4)
    parser.add_argument(
        "--real-seed", type=int, default=1000,
        help="held-out real year seed; 1000 matches the D=365 gate's own "
        "reference (validate_market_model.py default), so this diagnostic "
        "is scored against the exact same band as Table I / Section 6.2",
    )
    parser.add_argument(
        "--save-snapshots", type=Path, default=None,
        help="directory to save the ensemble at each checkpoint night, for "
        "later re-gating with a larger synthetic-year sample",
    )
    parser.add_argument(
        "--synthetic-years", type=int, default=20,
        help="synthetic years generated per checkpoint gate check; 5 (the "
        "validate_market_model.py default) is visibly noisy near the band "
        "edge for this diagnostic, so this script defaults higher",
    )
    args = parser.parse_args()

    market_config = default_market_config()
    cap = market_config.price_cap

    real = MarketHistory.collect(market_config, seed=args.real_seed, days=365)
    real_spread = stats(real, cap)["mean_daily_spread"]
    lo, hi = 0.7 * real_spread, 1.3 * real_spread
    print(
        f"held-out real year (seed {args.real_seed}): mean daily spread "
        f"{real_spread:.2f}, band [{lo:.1f}, {hi:.1f}]\n"
    )

    history = MarketHistory.load(args.history)
    seam = len(history)
    ensemble = MarketModelEnsemble.load(args.ensemble)

    horizon = args.pre_days + args.post_days
    config = EnvConfig(
        episode_days=horizon,
        scenario_sampler=lambda rng, start: [
            FuelShock(start_day=start + args.pre_days, duration_days=args.post_days, magnitude=2.0)
        ],
    )
    env = BatteryArbitrageEnv(config)
    obs, _ = env.reset(seed=args.seed)

    rows = []
    for night in range(horizon):
        if night in CHECKPOINT_NIGHTS:
            spread = gate_ensemble(ensemble, market_config, cap, real_spread, lo, hi, 20_000 + 1000 * night, args.synthetic_years)
            in_band = lo < spread < hi
            rows.append((night, spread, in_band))
            print(f"night {night:>3}  spread {spread:7.2f}  {'IN BAND' if in_band else 'OUT'}")
            if args.save_snapshots is not None:
                args.save_snapshots.mkdir(parents=True, exist_ok=True)
                ensemble.save(args.save_snapshots / f"night{night}.pt")

        # Drive one real day (24 hourly steps) through the env; idle is fine
        # since the policy's actions don't affect the exogenous prices.
        meta = env.today
        prices = []
        for _ in range(24):
            obs, _, terminated, truncated, info = env.step(idle_policy(env, obs))
            prices.append(info["price"])
            if terminated or truncated:
                break
        history.append_day(
            int(meta.day_of_year), bool(meta.is_weekend), bool(meta.is_holiday), np.asarray(prices)
        )

        refit_step(
            ensemble, history, seam, night + 1,
            args.window_days, args.anchor_days, args.ft_epochs, args.ft_lr, cap,
        )

    print("\nnight,spread,in_band")
    for night, spread, in_band in rows:
        print(f"{night},{spread:.1f},{in_band}")


if __name__ == "__main__":
    main()
