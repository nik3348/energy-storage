#!/usr/bin/env python
"""Figure for RQ2: does Dyna + planner demonstrations reduce infeasible
on-asset action requests versus a model-free agent, at data collection time
and after deployment?

Plots the two blocks of Table VII (tab:safe-exploration) side by side: the
training-time on-asset behaviours (where the planner's zero rate is the whole
point of the design) and the deployed, frozen checkpoints (where that
advantage does not survive). Numbers are the published table values, not
re-measured here; re-run scripts/results/measure_safe_exploration.py to
regenerate them.

Usage:
    uv run --extra train python scripts/results/plot_safe_exploration.py
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from energy_storage.viz import AQUA, BASELINE, BLUE, INK, MUTED, RED, SECONDARY, SURFACE, YELLOW, style_axis

TRAINING = [
    ("Random\n(model-free warmup)", 35.8, 3.3, RED),
    ("Heuristic\n(fixed rule)", 29.4, 0.1, MUTED),
    ("Planner\n(Dyna's data source)", 0.0, 0.0, AQUA),
    ("Idle", 0.0, 0.0, SECONDARY),
]

DEPLOYED = [
    ("dyna-365", 92.8, 6.7, AQUA),
    ("replay-365", 95.2, 5.8, YELLOW),
    ("unlimited-data", 95.1, 4.9, BLUE),
]


def bar_panel(ax, rows, title):
    labels = [r[0] for r in rows]
    means = [r[1] for r in rows]
    stds = [r[2] for r in rows]
    colors = [r[3] for r in rows]
    x = range(len(rows))
    ax.bar(x, means, yerr=stds, color=colors, width=0.6, capsize=4,
           error_kw={"ecolor": INK, "elinewidth": 1.2, "capthick": 1.2})
    for xi, m, s in zip(x, means, stds):
        ax.text(xi, m + s + 2.5, f"{m:.1f}%", ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=9, color=SECONDARY)
    ax.set_ylim(0, 122)
    style_axis(ax, title)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("diagnostics"))
    args = parser.parse_args()
    args.out.mkdir(exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5.6), facecolor=SURFACE)
    bar_panel(ax1, TRAINING, "During training: who sources real on-asset transitions")
    ax1.set_ylabel("Infeasible-request rate (%)", fontsize=9, color=SECONDARY)
    bar_panel(ax2, DEPLOYED, "After deployment: the frozen, trained checkpoints")

    bracket_y = 111
    ax2.plot([0, 0, 2, 2], [bracket_y - 2, bracket_y, bracket_y, bracket_y - 2],
              color=BASELINE, lw=1)
    ax2.text(1, bracket_y + 1.5, "statistically indistinguishable",
              ha="center", va="bottom", fontsize=8.5, color=SECONDARY)

    fig.suptitle(
        "RQ2: does Dyna + planner demonstrations request fewer infeasible actions than model-free?",
        fontsize=11, color=INK, x=0.02, ha="left", y=0.99,
    )
    fig.text(
        0.02, 0.90,
        "Yes during data collection (planner: 0% vs. random: 35.8%). No once deployed: dyna-365, replay-365 and\n"
        "unlimited-data converge on the same ~93-95% rate, since a converged policy parks near an SoC bound and every further push reads as infeasible.",
        fontsize=8.5, color=SECONDARY, ha="left", va="top",
    )

    fig.tight_layout(rect=(0, 0, 1, 0.78))
    path = args.out / "safe_exploration.png"
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
