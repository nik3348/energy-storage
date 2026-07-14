#!/usr/bin/env python
"""Predictive-accuracy report for a fitted market-day model.

The validation gate (validate_market_model.py) checks distributional
realism of free-running synthetic years; this script measures conditional
accuracy on a held-out real year: given the real previous day and the
calendar, how well does the model predict the actual next day?

Three legs:
  point       RMSE/MAE of the model's mean day vs the real day, benchmarked
              against persistence (yesterday's prices) and climatology
              (hour-of-day x month mean from the training budget)
  probabilistic  held-out per-hour NLL (exact, chain rule over hours,
              mixture over ensemble members) and 80%-interval coverage
  structure   peak/trough-hour hit rate (within +/-1h) and daily-spread
              correlation — the arbitrage-relevant shape of the day

Usage:
    uv run --extra train python scripts/evaluate_market_model.py --budget-days 365
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from energy_storage.env import default_market_config
from energy_storage.market_history import MarketHistory, build_dataset, denorm_price
from energy_storage.market_model import MarketModelEnsemble

N_SAMPLES = 30


def mixture_nll_per_hour(ensemble: MarketModelEnsemble, x: np.ndarray, y: np.ndarray) -> float:
    """Exact held-out NLL per hour: chain rule within each day (teacher-forced
    per-hour Gaussians), uniform mixture over ensemble members across the day."""
    x_t = torch.from_numpy(x.astype(np.float32))
    y_t = torch.from_numpy(y.astype(np.float32))
    day_logps = []
    with torch.no_grad():
        for model in ensemble.members:
            mu, log_std = model(x_t, y_t)
            hour_logp = -0.5 * ((y_t - mu) * torch.exp(-log_std)) ** 2 - log_std - 0.5 * np.log(2 * np.pi)
            day_logps.append(hour_logp.sum(dim=1))  # (days,)
    mixture = torch.logsumexp(torch.stack(day_logps), dim=0) - np.log(len(ensemble.members))
    return float(-mixture.mean() / y.shape[1])


def model_predictions(
    ensemble: MarketModelEnsemble, x: np.ndarray, cap: float, floor: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Free-running samples per conditioning row. Returns (mean, q10, q90)
    in $/MWh, each (days, 24)."""
    rng = np.random.default_rng(seed)
    means, q10s, q90s = [], [], []
    for row in x:
        draws = np.stack(
            [
                np.clip(
                    denorm_price(
                        ensemble.sample_day(row, rng, member=s % len(ensemble.members)), cap
                    ),
                    floor,
                    cap,
                )
                for s in range(N_SAMPLES)
            ]
        )
        means.append(draws.mean(axis=0))
        q10s.append(np.quantile(draws, 0.1, axis=0))
        q90s.append(np.quantile(draws, 0.9, axis=0))
    return np.stack(means), np.stack(q10s), np.stack(q90s)


def climatology(train: MarketHistory) -> dict[tuple[int, int], np.ndarray]:
    """Hour-of-day profile per month of the training budget."""
    months = train.day_of_year // 31
    return {
        m: train.prices[months == m].mean(axis=0)
        for m in np.unique(months)
    }


def climatology_prediction(clim: dict, day_of_year: np.ndarray) -> np.ndarray:
    months = day_of_year // 31
    keys = np.array(sorted(clim))
    rows = []
    for m in months:
        nearest = keys[np.abs(keys - m).argmin()]  # unseen month -> nearest seen
        rows.append(clim[nearest])
    return np.stack(rows)


def point_errors(pred: np.ndarray, actual: np.ndarray) -> tuple[float, float]:
    return float(np.sqrt(((pred - actual) ** 2).mean())), float(np.abs(pred - actual).mean())


def structure_metrics(pred: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    peak_hit = np.abs(pred.argmax(axis=1) - actual.argmax(axis=1)) <= 1
    trough_hit = np.abs(pred.argmin(axis=1) - actual.argmin(axis=1)) <= 1
    spread_pred = pred.max(axis=1) - pred.min(axis=1)
    spread_act = actual.max(axis=1) - actual.min(axis=1)
    return {
        "peak_hour_hit": float(peak_hit.mean()),
        "trough_hour_hit": float(trough_hit.mean()),
        "spread_corr": float(np.corrcoef(spread_pred, spread_act)[0, 1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget-days", type=int, default=365)
    parser.add_argument("--seed", type=int, default=0, help="training budget seed (for the model path and climatology)")
    parser.add_argument("--model", type=Path, default=None, help="default models/market-model-d{D}-s{seed}.pt")
    parser.add_argument("--eval-seed", type=int, default=None, help="held-out year seed; default seed+2000")
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    config = default_market_config()
    cap, floor = config.price_cap, config.price_floor
    model_path = args.model or Path("models") / f"market-model-d{args.budget_days}-s{args.seed}.pt"
    ensemble = MarketModelEnsemble.load(model_path)

    # Training budget (deterministic re-simulation) for climatology only.
    train = MarketHistory.collect(config, seed=args.seed, days=args.budget_days)
    eval_seed = args.eval_seed if args.eval_seed is not None else args.seed + 2000
    held_out = MarketHistory.collect(config, seed=eval_seed, days=366)

    x, y = build_dataset(held_out, cap)  # conditioning on REAL previous days
    actual = held_out.prices[1:]

    print(f"model: {model_path}  held-out year seed: {eval_seed}")
    nll = mixture_nll_per_hour(ensemble, x, y)

    mean_pred, q10, q90 = model_predictions(ensemble, x, cap, floor, seed=eval_seed)
    persistence = held_out.prices[:-1]
    clim_pred = climatology_prediction(climatology(train), held_out.day_of_year[1:])

    print(f"\nheld-out NLL per hour (log-normalized space): {nll:.3f}  (lower is better)")
    print(f"\n{'predictor':<14}{'RMSE $':>10}{'MAE $':>10}")
    rows = {}
    for name, pred in [("model mean", mean_pred), ("persistence", persistence), ("climatology", clim_pred)]:
        rmse, mae = point_errors(pred, actual)
        rows[name] = (rmse, mae)
        print(f"{name:<14}{rmse:>10.2f}{mae:>10.2f}")

    coverage = float(((actual >= q10) & (actual <= q90)).mean())
    print(f"\n80% interval coverage: {coverage:.3f}  (0.80 is calibrated; higher = over-dispersed)")

    structure = structure_metrics(mean_pred, actual)
    print(f"peak-hour hit rate (+/-1h):   {structure['peak_hour_hit']:.3f}")
    print(f"trough-hour hit rate (+/-1h): {structure['trough_hour_hit']:.3f}")
    print(f"daily-spread correlation:     {structure['spread_corr']:.3f}")

    if args.wandb:
        import wandb

        config_log = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        run = wandb.init(project="energy-storage", job_type="market-model-accuracy", config=config_log)
        wandb.log(
            {"nll_per_hour": nll, "coverage_80": coverage}
            | {f"rmse/{k.replace(' ', '_')}": v[0] for k, v in rows.items()}
            | {f"mae/{k.replace(' ', '_')}": v[1] for k, v in rows.items()}
            | structure
        )
        run.finish()
        print(f"wandb run: {run.url}")


if __name__ == "__main__":
    main()
