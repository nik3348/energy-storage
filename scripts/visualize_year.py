#!/usr/bin/env python
"""Visualize a full simulated market year of day-ahead prices.

Standalone — market only, no env or trained model. Electricity prices are
violently skewed (scarcity hours clear at the cap, an order of magnitude
above the median), so both panels clip their color/y scale at a high
percentile and mark the capped hours instead of letting a few spikes
flatten the rest of the year.

Usage:
    uv run --extra train python scripts/visualize_year.py
    uv run --extra train python scripts/visualize_year.py --seed 7 --no-wandb
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from energy_storage.market import Market
from energy_storage.viz import (
    BASELINE,
    BLUE,
    INK,
    MUTED,
    SECONDARY,
    SURFACE,
    style_axis,
)

SEQ_DARK = "#104281"
HEATMAP_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
                "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]


def simulate_year(seed: int, days: int):
    market = Market(seed=seed)
    results = [market.simulate_day() for _ in range(days)]
    daily = np.stack([d.prices for d in results])  # (days, 24)
    return daily, market.config.price_cap


def month_ticks(ax, days):
    ax.set_xlabel("day of year", color=MUTED, fontsize=9)
    ax.set_xlim(0, days)
    ax.set_xticks(np.arange(0, days + 1, 30))


def prices_axis(ax, daily, cap):
    """Daily mean line + min-max band, y clipped at p99.5, capped hours marked."""
    days = np.arange(len(daily))
    prices = daily.ravel()
    # Clip on the sub-cap distribution: cap hours can outnumber the top
    # percentile of all prices, which would put the clip at the cap itself.
    subcap = prices[prices < cap - 0.5]
    clip = float(np.percentile(subcap, 99.5))
    cap_hours = int((prices >= cap - 0.5).sum())

    ax.fill_between(
        days, daily.min(axis=1), np.minimum(daily.max(axis=1), clip),
        color=BLUE, alpha=0.18, linewidth=0,
    )
    ax.plot(days, daily.mean(axis=1), color=BLUE, linewidth=2)
    cap_days = days[(daily >= cap - 0.5).any(axis=1)]
    if len(cap_days):
        ax.plot(
            cap_days, np.full(len(cap_days), clip * 1.02), linestyle="none",
            marker="v", markersize=5, color=SEQ_DARK, clip_on=False,
        )
    ax.axhline(0, color=BASELINE, linewidth=1)
    ax.set_ylim(min(-10.0, float(prices.min()) - 10.0), clip * 1.04)
    style_axis(ax, "Day-ahead price over the year (\\$/MWh)")
    ax.set_title(ax.get_title(loc="left"), loc="left", fontsize=10, color=INK, pad=22)
    ax.text(
        0, 1.015,
        f"line = daily mean · band = daily min-max (clipped at sub-cap p99.5 = \\${clip:.0f}) · "
        f"triangles = days with hours at the \\${cap:.0f} cap ({cap_hours} h total)",
        transform=ax.transAxes, fontsize=8, color=SECONDARY, va="bottom",
    )


def heatmap_axis(fig, ax, daily):
    """Day x hour price heatmap, color clipped at p98."""
    vmax = float(np.quantile(daily, 0.98))
    cmap = LinearSegmentedColormap.from_list("price_seq", HEATMAP_RAMP)
    im = ax.imshow(
        daily.T, aspect="auto", origin="lower", cmap=cmap, vmin=0.0, vmax=vmax,
        extent=(0, len(daily), 0, 24), interpolation="nearest",
    )
    style_axis(ax, "Price by day and hour (\\$/MWh) — spikes saturate dark")
    ax.grid(False)
    ax.set_yticks([0, 6, 12, 18, 24])
    ax.set_ylabel("hour of day", color=MUTED, fontsize=9)
    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.ax.tick_params(colors=MUTED, labelsize=8)
    cbar.outline.set_visible(False)
    cbar.set_label(f"\\$/MWh (color clipped at p98 = \\${vmax:.0f})", color=MUTED, fontsize=8)


def figure(daily, cap):
    fig, (ax_price, ax_heat) = plt.subplots(
        2, 1, figsize=(11, 7.5), facecolor=SURFACE,
        gridspec_kw={"height_ratios": [1.2, 1]},
    )
    prices_axis(ax_price, daily, cap)
    month_ticks(ax_price, len(daily))
    heatmap_axis(fig, ax_heat, daily)
    month_ticks(ax_heat, len(daily))
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--out", type=Path, default=Path("diagnostics"))
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    daily, cap = simulate_year(args.seed, args.days)
    prices = daily.ravel()
    print(
        f"median {np.median(prices):.1f}  mean {prices.mean():.1f}  "
        f"cap hours {(prices >= cap - 0.5).sum()}  negative hours {(prices < 0).sum()}"
    )

    fig = figure(daily, cap)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "prices_year.png"
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    print(f"wrote {path}")

    if args.wandb:
        import wandb

        config_log = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        run = wandb.init(project="energy-storage", job_type="visualize-year", config=config_log)
        wandb.log({"prices_year": wandb.Image(fig)})
        run.finish()
        print(f"wandb run: {run.url}")


if __name__ == "__main__":
    main()
