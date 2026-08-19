#!/usr/bin/env python
"""The full benchmark ladder: idle, heuristic, replay-365, dyna-365,
unlimited-data, and the rolling-horizon oracle, all as a fraction of the
hindsight optimum (Section 5.1, "Oracles and Baselines").

idle/heuristic/rolling-oracle come from results/budget_sweep.json (a single
coherent measurement run, 20 paired 14-day episodes, price_cap=$1000/MWh).
dyna-365/replay-365 use the same published D=365, 5-training-seed numbers as
plot_budget_sweep.py (results/fixed_history_sweep_final.json), so the two
figures agree. unlimited-data uses the published Table tab:robustness-scenarios
calm-column value, matching plot_budget_sweep.py.

Usage:
    uv run --extra train python scripts/results/plot_benchmark_ladder.py
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from energy_storage.viz import AQUA, BASELINE, BLUE, INK, MUTED, SECONDARY, SURFACE, YELLOW, style_axis

# Published references, kept consistent with plot_budget_sweep.py.
UNLIMITED_MEAN, UNLIMITED_STD = 72.6, 13.1
DYNA_MEAN, DYNA_STD = 70.7, 2.6
REPLAY_MEAN, REPLAY_STD = 54.2, 5.4


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ladder", type=Path, default=Path("results/budget_sweep.json"))
    parser.add_argument("--out", type=Path, default=Path("diagnostics"))
    args = parser.parse_args()
    args.out.mkdir(exist_ok=True)

    data = json.loads(args.ladder.read_text())
    idle = data["arms"]["idle"]
    heuristic = data["arms"]["heuristic"]
    oracle = data["arms"]["rolling-oracle"]

    rows = [
        ("idle", idle["capture_pct"], idle["capture_std_pct"], MUTED, "solid"),
        ("heuristic", heuristic["capture_pct"], heuristic["capture_std_pct"], MUTED, "solid"),
        ("replay-365", REPLAY_MEAN, REPLAY_STD, YELLOW, "solid"),
        ("dyna-365", DYNA_MEAN, DYNA_STD, AQUA, "solid"),
        ("unlimited-data", UNLIMITED_MEAN, UNLIMITED_STD, BLUE, "solid"),
        ("rolling-horizon\noracle", oracle["capture_pct"], oracle["capture_std_pct"], SURFACE, "hatch"),
    ]

    fig, ax = plt.subplots(figsize=(10, 6.6), facecolor=SURFACE)
    x = range(len(rows))
    for xi, (label, mean, std, color, style) in zip(x, rows):
        if style == "hatch":
            ax.bar(xi, mean, yerr=std, width=0.6, facecolor=SURFACE, edgecolor=INK,
                   hatch="////", linewidth=1.3, capsize=4,
                   error_kw={"ecolor": INK, "elinewidth": 1.2, "capthick": 1.2}, zorder=2)
        else:
            ax.bar(xi, mean, yerr=std, width=0.6, color=color, capsize=4,
                   error_kw={"ecolor": INK, "elinewidth": 1.2, "capthick": 1.2}, zorder=2)
        label_y = mean + std + 4 if mean >= 0 else mean - std - 4
        va = "bottom" if mean >= 0 else "top"
        ax.text(xi, label_y, f"{mean:.1f}%", ha="center", va=va, fontsize=9.5, color=INK)

    ax.axhline(100, color=INK, linewidth=1.2, linestyle="--", zorder=1)
    ax.text(0.5, 103.5, "hindsight optimum = 100%", ha="center", va="bottom",
            fontsize=8.5, color=SECONDARY)

    ax.axhline(0, color=BASELINE, linewidth=1)
    ax.set_xticks(list(x))
    ax.set_xticklabels([r[0] for r in rows], fontsize=9.5, color=SECONDARY)
    ax.tick_params(axis="x", pad=32)
    ax.set_ylabel("Capture: % of hindsight optimum", fontsize=9, color=SECONDARY)
    ax.set_ylim(-85, 118)
    style_axis(ax, "Full benchmark ladder, price_cap = $1000/MWh, 20 paired 14-day episodes")

    fig.suptitle(
        "RQ1 in context: where dyna-365 sits on the full benchmark ladder",
        fontsize=11, color=INK, x=0.02, ha="left", y=0.99,
    )
    fig.text(
        0.02, 0.90,
        "Idle still loses money to unavoidable calendar ageing; the heuristic over-cycles and stays net-negative. Dyna-365\n"
        "recovers almost all the unlimited-data ceiling on two orders of magnitude less real data. The rolling-horizon oracle\n"
        "(24h window, replanned hourly) reaches 99.9% of hindsight, so the remaining gap above dyna is a learning\n"
        "shortfall, not a missing-information one.",
        fontsize=8.5, color=SECONDARY, ha="left", va="top",
    )

    fig.tight_layout(rect=(0, 0.02, 1, 0.72))
    path = args.out / "benchmark_ladder.png"
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
