#!/usr/bin/env python
"""Figure for RQ1: under a fixed budget of observed days, can training
inside a learned market substitute for real data?

Reads the per-seed dyna/replay captures from
results/fixed_history_sweep_final.json (five training seeds, all pinned to
the same D=365 real-history draw and gated ensemble, Table VI in the paper)
and plots them against the unlimited-data model-free ceiling reported in
Table tab:robustness-scenarios' calm column. Dyna and replay error bars are
std across the five training seeds; the unlimited-data error bar is a single
policy's leave-one-episode-out jackknife std, a different kind of spread,
called out in the figure text rather than hidden.

Usage:
    uv run --extra train python scripts/results/plot_budget_sweep.py
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from energy_storage.viz import AQUA, BASELINE, BLUE, INK, SECONDARY, SURFACE, YELLOW, style_axis

# Published unlimited-data reference (Table tab:robustness-scenarios, calm column):
# an unbounded-budget SAC policy, the model-free ceiling.
UNLIMITED_MEAN, UNLIMITED_STD = 72.6, 13.1
PAIRED_MEAN, PAIRED_SEM = 4.62, 0.61


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", type=Path, default=Path("results/fixed_history_sweep_final.json"))
    parser.add_argument("--out", type=Path, default=Path("diagnostics"))
    args = parser.parse_args()
    args.out.mkdir(exist_ok=True)

    data = json.loads(args.sweep.read_text())
    dyna_pts = [v["capture"] for v in data["runs"]["dyna"].values()]
    replay_pts = [v["capture"] for v in data["runs"]["replay"].values()]
    dyna_mean, dyna_std = data["summary"]["dyna"]["capture_mean"], data["summary"]["dyna"]["capture_std"]
    replay_mean, replay_std = data["summary"]["replay"]["capture_mean"], data["summary"]["replay"]["capture_std"]

    fig, ax = plt.subplots(figsize=(8.5, 6.6), facecolor=SURFACE)

    bars = [
        ("dyna-365", dyna_mean, dyna_std, AQUA, dyna_pts),
        ("replay-365", replay_mean, replay_std, YELLOW, replay_pts),
        ("unlimited-data", UNLIMITED_MEAN, UNLIMITED_STD, BLUE, None),
    ]
    x = range(len(bars))
    means = [b[1] for b in bars]
    stds = [b[2] for b in bars]
    colors = [b[3] for b in bars]
    ax.bar(x, means, yerr=stds, color=colors, width=0.55, capsize=4,
           error_kw={"ecolor": INK, "elinewidth": 1.2, "capthick": 1.2}, zorder=2)

    rng = np.random.default_rng(0)
    for xi, (_, _, _, _, pts) in zip(x, bars):
        if pts is None:
            continue
        jitter = rng.uniform(-0.12, 0.12, size=len(pts))
        ax.scatter([xi] * len(pts) + jitter, pts, s=26, color=SURFACE,
                   edgecolors=INK, linewidths=1.0, zorder=3)

    for xi, m, s in zip(x, means, stds):
        ax.text(xi, m + s + 3, f"{m:.1f}%", ha="center", va="bottom", fontsize=10, color=INK)

    bracket_y = 96
    ax.plot([0, 0, 1, 1], [bracket_y - 1.5, bracket_y, bracket_y, bracket_y - 1.5], color=BASELINE, lw=1)
    ax.text(0.5, bracket_y + 1.2, f"paired: +${PAIRED_MEAN:.2f}±{PAIRED_SEM:.2f}/episode\n(positive at every seed)",
            ha="center", va="bottom", fontsize=8.5, color=SECONDARY)

    ax.set_xticks(list(x))
    ax.set_xticklabels([b[0] for b in bars], fontsize=10, color=SECONDARY)
    for xi, budget_label in zip(x, ["365 real days", "365 real days", "≫365 real days\n(uncapped)"]):
        ax.text(xi, -9, budget_label, ha="center", va="top", fontsize=8, color=SECONDARY)

    ax.set_ylabel("Capture: % of hindsight optimum", fontsize=9, color=SECONDARY)
    ax.set_ylim(0, 112)
    style_axis(ax, "D = 365 real days, the one budget of {90, 365, 1095} that passes the calibration gate")

    fig.suptitle(
        "RQ1: under a fixed budget of observed days, can training inside a\n"
        "learned market substitute for real data?",
        fontsize=11, color=INK, x=0.02, ha="left", y=0.985,
    )
    fig.text(
        0.02, 0.80,
        "Yes. Dyna recovers the unlimited-data ceiling within its own seed-to-seed spread on two orders of magnitude less real\n"
        "data. A replay control on the identical year, with unlimited gradient steps, finishes far behind. Dots are the 5\n"
        "individual training seeds; dyna/replay error bars are std across seeds, unlimited-data's is a single policy's episode jackknife.",
        fontsize=8.5, color=SECONDARY, ha="left", va="top",
    )

    fig.tight_layout(rect=(0, 0.03, 1, 0.68))
    path = args.out / "budget_sweep_rq1.png"
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
