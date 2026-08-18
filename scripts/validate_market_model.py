#!/usr/bin/env python
"""Fit the market-day model on a budget of real history and gate it on the
market's calibration statistics BEFORE any RL trains on it.

A model that misses the spread/spike statistics silently guts arbitrage
value in imagination, so the Dyna result is only interpretable if the
synthetic years pass (roughly) the same gates as tests/test_market.py.

Usage:
    uv run --extra train python scripts/validate_market_model.py --budget-days 365
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from energy_storage.env import default_market_config
from energy_storage.market_history import MarketHistory, build_dataset
from energy_storage.market_model import LearnedMarket, MarketModelEnsemble
from energy_storage.viz import AQUA, BLUE, MUTED, SECONDARY, SURFACE, style_axis

BURN_IN_DAYS = 3


def synthetic_year(ensemble, config, seed: int) -> MarketHistory:
    market = LearnedMarket(ensemble, config, seed=seed)
    market.day = -BURN_IN_DAYS
    for _ in range(BURN_IN_DAYS):
        market.simulate_day()
    return MarketHistory.from_days([market.simulate_day() for _ in range(365)])


def stats(history: MarketHistory, cap: float) -> dict[str, float]:
    daily = history.prices  # (days, 24)
    prices = daily.ravel()
    doy = history.day_of_year
    winter = daily[(doy < 45) | (doy >= 320)].ravel()
    summer = daily[(135 <= doy) & (doy < 250)].ravel()
    return {
        "median": float(np.median(prices)),
        "mean": float(prices.mean()),
        "winter_median_gap": float(np.median(winter) - np.median(summer)),
        "winter_mean_gap": float(winter.mean() - summer.mean()),
        "evening_minus_night": float(daily[:, 18:21].mean() - daily[:, 2:5].mean()),
        "mean_daily_spread": float((daily.max(axis=1) - daily.min(axis=1)).mean()),
        "cap_share": float((prices >= cap - 0.5).mean()),
        "negative_share": float((prices < 0).mean()),
    }


def gates(synthetic: dict[str, float], real: dict[str, float]) -> dict[str, bool]:
    """The test_market.py assertions, applied to the synthetic years, plus a
    spread-fidelity check against the held-out real year."""
    return {
        "median": 30.0 < synthetic["median"] < 90.0,
        "winter_median_gap": synthetic["winter_median_gap"] > 6.0,
        "winter_mean_gap": synthetic["winter_mean_gap"] > 15.0,
        "evening_minus_night": synthetic["evening_minus_night"] > 5.0,
        "mean_daily_spread": (
            0.7 * real["mean_daily_spread"]
            < synthetic["mean_daily_spread"]
            < 1.3 * real["mean_daily_spread"]
        ),
        "cap_share": 0.0 < synthetic["cap_share"] < 0.015,
        "negative_share": 0.0 < synthetic["negative_share"] < 0.15,
    }


def figure(real: MarketHistory, synth: MarketHistory, cap: float):
    fig, (ax_profile, ax_hist) = plt.subplots(1, 2, figsize=(11, 4), facecolor=SURFACE)

    hours = np.arange(24)
    ax_profile.plot(hours, real.prices.mean(axis=0), color=BLUE, linewidth=2, label="real")
    ax_profile.plot(hours, synth.prices.mean(axis=0), color=AQUA, linewidth=2, label="synthetic")
    style_axis(ax_profile, "Mean price by hour of day (\\$/MWh)")
    ax_profile.set_xticks(np.arange(0, 24, 3))
    ax_profile.set_xlabel("hour of day", color=MUTED, fontsize=9)
    ax_profile.legend(frameon=False, fontsize=9, labelcolor=SECONDARY)

    bins = np.linspace(-50, 350, 80)
    ax_hist.hist(real.prices.ravel(), bins=bins, color=BLUE, alpha=0.6, density=True, label="real")
    ax_hist.hist(
        synth.prices.ravel(), bins=bins, color=AQUA, alpha=0.6, density=True, label="synthetic"
    )
    style_axis(ax_hist, "Hourly price distribution (body; cap hours off-scale)")
    ax_hist.set_xlabel("\\$/MWh", color=MUTED, fontsize=9)
    ax_hist.legend(frameon=False, fontsize=9, labelcolor=SECONDARY)

    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget-days", type=int, default=365)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--history", type=Path, default=None, help="npz from collect_history.py; default: simulate fresh")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--years", type=int, default=3, help="synthetic years to generate")
    parser.add_argument("--save-model", type=Path, default=None, help="default models/market-model-d{D}-s{seed}.pt")
    parser.add_argument("--out", type=Path, default=Path("diagnostics"))
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    config = default_market_config()
    cap = config.price_cap

    if args.history is not None:
        history = MarketHistory.load(args.history)
    else:
        history = MarketHistory.collect(config, seed=args.seed, days=args.budget_days)
    print(f"fitting K={args.k} ensemble on {len(history)} observed days...")
    x, y = build_dataset(history, cap)
    ensemble = MarketModelEnsemble(k=args.k, hidden=args.hidden, seed=args.seed)
    losses = ensemble.fit(x, y, epochs=args.epochs, seed=args.seed)
    print("final member losses:", [round(l, 3) for l in losses])

    # Held-out real year (different seed than the training budget) as the
    # reference the synthetic years are compared against.
    real = MarketHistory.collect(config, seed=args.seed + 1000, days=365)
    synth_years = [
        synthetic_year(ensemble, config, seed=10_000 + i) for i in range(args.years)
    ]
    synth = MarketHistory(
        prices=np.concatenate([s.prices for s in synth_years]),
        day_of_year=np.concatenate([s.day_of_year for s in synth_years]),
        is_weekend=np.concatenate([s.is_weekend for s in synth_years]),
        is_holiday=np.concatenate([s.is_holiday for s in synth_years]),
    )

    real_stats, synth_stats = stats(real, cap), stats(synth, cap)
    verdicts = gates(synth_stats, real_stats)
    print(f"\n{'metric':<22}{'real':>10}{'synthetic':>12}   gate")
    for key in real_stats:
        verdict = {True: "PASS", False: "FAIL"}.get(verdicts.get(key), "")
        print(f"{key:<22}{real_stats[key]:>10.3f}{synth_stats[key]:>12.3f}   {verdict}")
    passed = all(verdicts.values())
    print("\nGATE:", "PASS — model is fit to train on" if passed else "FAIL — do not train on this model")

    # The saved file is the gate certificate: train_dyna.py refuses to run
    # the dyna arm without it, so only a passing model gets written.
    if passed:
        save_model = args.save_model or Path("models") / f"market-model-d{args.budget_days}-s{args.seed}.pt"
        save_model.parent.mkdir(parents=True, exist_ok=True)
        ensemble.save(save_model)
        print(f"wrote {save_model}")
    else:
        print("model NOT saved (gate failed)")

    fig = figure(real, synth, cap)
    args.out.mkdir(parents=True, exist_ok=True)
    fig_path = args.out / f"market_model_gate_d{args.budget_days}.png"
    fig.savefig(fig_path, dpi=150, facecolor=SURFACE)
    print(f"wrote {fig_path}")

    if args.wandb:
        import wandb

        config_log = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        run = wandb.init(project="energy-storage", job_type="market-model-gate", config=config_log)
        wandb.log(
            {f"real/{k}": v for k, v in real_stats.items()}
            | {f"synthetic/{k}": v for k, v in synth_stats.items()}
            | {f"gate/{k}": int(v) for k, v in verdicts.items()}
            | {"gate/all": int(passed), "figure": wandb.Image(fig)}
        )
        run.finish()
        print(f"wandb run: {run.url}")

    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
