#!/usr/bin/env python
"""Visual diagnostics for a trained agent: what does it actually do?

Usage:
    uv run --extra train python scripts/diagnose.py --model models/best_model.zip

Produces three figures (saved to --out, logged to wandb unless --no-wandb):
  behavior.png   one week of day-ahead price, battery power, and SoC
  economics.png  cumulative net profit: agent vs heuristic vs idle, same market
  schedule.png   mean charge/discharge energy by hour of day
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import SAC

from energy_storage import EnvConfig
from energy_storage.baselines import (
    collect_episode,
    evaluate,
    heuristic_policy,
    idle_policy,
    model_policy,
)

# Reference dataviz palette (light mode), fixed slot order.
BLUE, AQUA, YELLOW = "#2a78d6", "#1baf7a", "#eda100"
RED = "#e34948"  # diverging counterpart to blue
INK, SECONDARY, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"


def style_axis(ax, title):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_title(title, loc="left", fontsize=10, color=INK)


def day_ticks(ax, hours, first_day=0):
    days = np.arange(0, hours + 1, 24)
    ax.set_xticks(days)
    ax.set_xticklabels([f"day {first_day + d // 24 + 1}" for d in days[:-1]] + [""])
    for d in days[1:-1]:
        ax.axvline(d, color=GRID, linewidth=0.8)


def most_active_window_start(energy, hours):
    """Day-aligned start of the window with the most battery activity."""
    if len(energy) <= hours:
        return 0
    starts = range(0, len(energy) - hours + 1, 24)
    return max(starts, key=lambda s: np.abs(energy[s : s + hours]).sum())


def behavior_figure(traj, days):
    hours = days * 24
    start = most_active_window_start(traj["grid_energy_kwh"], hours)
    window = slice(start, start + hours)
    t = np.arange(hours)
    fig, (ax_price, ax_power, ax_soc) = plt.subplots(
        3, 1, figsize=(10, 7), sharex=True, facecolor=SURFACE
    )

    ax_price.plot(t, traj["price"][window], color=BLUE, linewidth=2)
    style_axis(ax_price, "Day-ahead price ($/MWh)")

    energy = traj["grid_energy_kwh"][window]
    colors = np.where(energy >= 0, BLUE, RED)
    ax_power.bar(t, energy, width=0.8, color=colors)
    ax_power.axhline(0, color=BASELINE, linewidth=1)
    style_axis(ax_power, "Battery grid energy (kWh/h) — charge in blue, discharge in red")

    ax_soc.plot(t, traj["soc"][window], color=AQUA, linewidth=2)
    ax_soc.set_ylim(0, 1)
    style_axis(ax_soc, "State of charge")
    day_ticks(ax_soc, hours, first_day=start // 24)

    fig.tight_layout()
    return fig


def economics_figure(trajectories):
    fig, ax = plt.subplots(figsize=(10, 4.5), facecolor=SURFACE)
    slots = {"sac": BLUE, "heuristic": AQUA, "idle": YELLOW}
    for name, color in slots.items():
        traj = trajectories[name]
        net = np.cumsum(traj["profit"] - traj["degradation_cost"])
        t = np.arange(len(net))
        ax.plot(t, net, color=color, linewidth=2, label=name)
        ax.annotate(
            f"{name}  ${net[-1]:.0f}",
            xy=(t[-1], net[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            fontsize=9,
            color=SECONDARY,
            va="center",
        )
    ax.axhline(0, color=BASELINE, linewidth=1)
    style_axis(ax, "Cumulative net profit ($), same market — agent vs baselines")
    day_ticks(ax, len(net))
    ax.legend(loc="lower left", ncols=3, frameon=False, fontsize=9, labelcolor=SECONDARY)
    ax.margins(x=0.08)
    fig.tight_layout()
    return fig


def schedule_figure(trajectories_by_episode):
    by_hour = np.zeros(24)
    counts = np.zeros(24)
    for traj in trajectories_by_episode:
        for hour, energy in zip(traj["hour"], traj["grid_energy_kwh"]):
            by_hour[hour] += energy
            counts[hour] += 1
    mean_energy = by_hour / np.maximum(counts, 1)

    fig, ax = plt.subplots(figsize=(10, 4), facecolor=SURFACE)
    colors = np.where(mean_energy >= 0, BLUE, RED)
    ax.bar(np.arange(24), mean_energy, width=0.7, color=colors)
    ax.axhline(0, color=BASELINE, linewidth=1)
    style_axis(ax, "Mean grid energy by hour of day (kWh/h) — charge in blue, discharge in red")
    ax.set_xticks(np.arange(0, 24, 3))
    ax.set_xlabel("hour of day", color=MUTED, fontsize=9)
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/best_model.zip"))
    parser.add_argument("--episode-days", type=int, default=14)
    parser.add_argument("--plot-days", type=int, default=7)
    parser.add_argument("--episodes", type=int, default=10, help="episodes for the schedule plot")
    parser.add_argument("--seed", type=int, default=2_000_000)
    parser.add_argument("--out", type=Path, default=Path("diagnostics"))
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    model = SAC.load(args.model)
    sac_policy = model_policy(model)

    config = EnvConfig(episode_days=args.episode_days)

    print("Rolling out episodes...")
    trajectories = {
        name: collect_episode(policy, config, seed=args.seed)
        for name, policy in [("sac", sac_policy), ("heuristic", heuristic_policy), ("idle", idle_policy)]
    }
    sac_episodes = [trajectories["sac"]] + [
        collect_episode(sac_policy, config, seed=args.seed + ep) for ep in range(1, args.episodes)
    ]

    figures = {
        "behavior": behavior_figure(trajectories["sac"], days=args.plot_days),
        "economics": economics_figure(trajectories),
        "schedule": schedule_figure(sac_episodes),
    }

    args.out.mkdir(parents=True, exist_ok=True)
    for name, fig in figures.items():
        path = args.out / f"{name}.png"
        fig.savefig(path, dpi=150, facecolor=SURFACE)
        print(f"wrote {path}")

    # Summary economics averaged over all diagnostic episodes, not just the
    # single plotted seed.
    summary = {}
    for name, policy in [("sac", sac_policy), ("heuristic", heuristic_policy), ("idle", idle_policy)]:
        stats = evaluate(policy, config, episodes=args.episodes, seed0=args.seed)
        summary.update({f"{name}/{k}": v for k, v in stats.items()})
    summary["sac/soh_loss"] = float(trajectories["sac"]["soh"][0] - trajectories["sac"]["soh"][-1])
    print("summary:", {k: round(v, 3) for k, v in summary.items()})

    if args.wandb:
        import wandb

        config_log = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        run = wandb.init(project="energy-storage", job_type="diagnostics", config=config_log)
        wandb.log({name: wandb.Image(fig) for name, fig in figures.items()} | summary)
        run.finish()
        print(f"wandb run: {run.url}")


if __name__ == "__main__":
    main()
