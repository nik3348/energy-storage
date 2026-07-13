#!/usr/bin/env python
"""Visualize one or more simulated market days: price, supply stack, battery.

Standalone — no trained model required. Runs the env with a chosen policy and
produces a single figure with three stacked panels sharing the hour axis:

  1. Day-ahead price ($/MWh)
  2. Supply stack: per-hour generation by tech, stacked in the actual clearing
     merit order (cheapest bid at the bottom, dearest dispatched on top). A
     dashed demand line is overlaid so scarcity hours (demand > stack top)
     are visible.
  3. Battery SoC and SoH (0-1)

Usage:
    uv run --extra train python scripts/visualize_day.py --days 2 --policy heuristic
    uv run --extra train python scripts/visualize_day.py --days 3 --policy model --model models/best_model.zip
"""

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from energy_storage import BatteryArbitrageEnv, EnvConfig
from energy_storage.baselines import heuristic_policy, idle_policy, model_policy
from energy_storage.market.day_ahead import DayResult
from energy_storage.viz import (
    AQUA,
    BASELINE,
    BLUE,
    GRID,
    INK,
    MUTED,
    SECONDARY,
    SURFACE,
    TECH_COLORS,
    day_ticks,
    style_axis,
)


def collect_days_and_battery(env, policy_fn, n_days, seed=0):
    """Roll the env forward, collecting one DayResult per simulated day plus
    per-hour battery SoC/SoH. Returns (days, soc, soh, hours) where days is a
    list of DayResult in chronological order and soc/soh are (n_days*24,)
    arrays aligned to those days' hours."""
    obs, _ = env.reset(seed=seed)
    seen_days: dict[int, DayResult] = {}
    soc_by_day: dict[int, list[float]] = defaultdict(list)
    soh_by_day: dict[int, list[float]] = defaultdict(list)
    done = False
    steps = 0
    max_steps = n_days * 24 + 24  # small headroom for buffer rollover
    while not done and steps < max_steps:
        day_id = env.today.day
        seen_days.setdefault(day_id, env.today)
        obs, _, terminated, truncated, info = env.step(policy_fn(env, obs))
        soc_by_day[day_id].append(info["soc"])
        soh_by_day[day_id].append(info["soh"])
        done = terminated or truncated
        steps += 1

    # Order by first-seen insertion (chronological, since the env advances
    # day_id monotonically except for the reset-time buffer day).
    ordered_ids = list(seen_days.keys())
    days = [seen_days[i] for i in ordered_ids[:n_days]]
    soc = np.concatenate([soc_by_day[i] for i in ordered_ids[:n_days]])
    soh = np.concatenate([soh_by_day[i] for i in ordered_ids[:n_days]])
    return days, soc, soh


def stack_axis(ax, days):
    """Draw the per-hour supply stack on ax, sorted each hour by that hour's
    actual marginal bid (cheapest at the bottom)."""
    n_hours = len(days) * 24
    techs_present = set()
    for d in days:
        techs_present.update(d.generation_mw.keys())
    # Fixed legend order = conventional merit order (reads bottom-to-top).
    legend_order = [t for t in TECH_COLORS if t in techs_present]
    extras = sorted(t for t in techs_present if t not in TECH_COLORS)
    legend_order += extras

    bottoms = np.zeros(n_hours)
    # For each hour, build (tech, gen, bid) triples, sort by bid, then place.
    # We iterate tech in legend_order but stack in bid-sorted per-hour order,
    # which means each tech's color appears at varying heights across hours
    # — exactly the "re-sorted each hour" behavior we want.
    h_global = 0
    for d in days:
        gen = d.generation_mw
        bids = d.marginal_bid_by_tech
        for h in range(24):
            triples = []
            for t in legend_order:
                arr = gen.get(t)
                if arr is None:
                    continue
                mw = float(arr[h])
                if mw <= 1e-9:
                    continue
                bid = float(bids.get(t, np.full(24, np.nan))[h])
                triples.append((t, mw, bid if not np.isnan(bid) else np.inf))
            # Cheapest bid at the bottom.
            triples.sort(key=lambda x: x[2])
            base = bottoms[h_global]
            for t, mw, _ in triples:
                ax.bar(h_global, mw, bottom=base, width=0.9,
                       color=TECH_COLORS.get(t, MUTED), edgecolor="none")
                base += mw
            h_global += 1

    # Demand overlay (dashed) so scarcity hours are readable.
    demand = np.concatenate([d.demand_mw for d in days])
    ax.plot(np.arange(n_hours), demand, color=INK, linewidth=1.2,
            linestyle="--", label="demand")

    ax.set_xlim(-0.5, n_hours - 0.5)
    style_axis(ax, "Generation by tech, stacked in clearing merit order (MW)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=TECH_COLORS.get(t, MUTED)) for t in legend_order]
    handles.append(plt.Line2D([0], [0], color=INK, linewidth=1.2, linestyle="--"))
    labels = legend_order + ["demand"]
    ax.legend(handles, labels, loc="upper left", ncols=len(labels),
              frameon=False, fontsize=8, labelcolor=SECONDARY,
              columnspacing=1.2, handlelength=1.4)


def figure(days, soc, soh):
    n_hours = len(days) * 24
    t = np.arange(n_hours)
    prices = np.concatenate([d.prices for d in days])
    fig, (ax_price, ax_stack, ax_bat) = plt.subplots(
        3, 1, figsize=(11, 9), sharex=True,
        gridspec_kw={"height_ratios": [1, 2, 1]}, facecolor=SURFACE,
    )

    ax_price.plot(t, prices, color=BLUE, linewidth=2)
    style_axis(ax_price, "Day-ahead price ($/MWh)")

    stack_axis(ax_stack, days)

    ax_bat.plot(t, soc, color=AQUA, linewidth=2, label="SoC")
    ax_bat.plot(t, soh, color=MUTED, linewidth=1.5, label="SoH")
    ax_bat.set_ylim(0, 1)
    style_axis(ax_bat, "Battery state of charge / state of health")
    ax_bat.legend(loc="lower left", ncols=2, frameon=False, fontsize=9,
                  labelcolor=SECONDARY, columnspacing=1.5, handlelength=2)
    day_ticks(ax_bat, n_hours)
    ax_bat.set_xlabel("hour", color=MUTED, fontsize=9)

    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy", choices=("idle", "heuristic", "model"), default="heuristic")
    parser.add_argument("--model", type=Path, default=Path("models/best_model.zip"),
                        help="path to a trained SB3 model (used with --policy model)")
    parser.add_argument("--out", type=Path, default=Path("diagnostics"))
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if args.policy == "model":
        from stable_baselines3 import SAC
        policy_fn = model_policy(SAC.load(args.model))
    elif args.policy == "heuristic":
        policy_fn = heuristic_policy
    else:
        policy_fn = idle_policy

    config = EnvConfig(episode_days=args.days)
    env = BatteryArbitrageEnv(config)
    days, soc, soh = collect_days_and_battery(env, policy_fn, args.days, seed=args.seed)
    print(f"day-of-year starts at {days[0].day_of_year} ({len(days)} days)")

    fig = figure(days, soc, soh)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "supply_stack.png"
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    print(f"wrote {path}")

    if args.wandb:
        import wandb
        config_log = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        run = wandb.init(project="energy-storage", job_type="visualize-day", config=config_log)
        wandb.log({"supply_stack": wandb.Image(fig)})
        run.finish()
        print(f"wandb run: {run.url}")


if __name__ == "__main__":
    main()
