#!/usr/bin/env python
"""Figure for Experiment 4 (RQ4): can an in-imagination adaptation loop
recover from persistent regime shifts?

Values are Table tab:adaptation-results (docs/main.tex), pre/post capture
(pooled 7-day net profit over the same-window oracle, jackknife std),
10 paired 120-day deployments per shift. Hardcoded from the published table,
consistent with plot_scenario_robustness.py's approach (no backing results/
JSON aggregates all five arms at the paper's exact metric definition).

Colors match energy_storage/viz.py's ARM_COLORS convention, already used by
plot_adaptation.py for the per-shift recovery-curve figures in the appendix,
so this summary figure reads as the same visual language.

Usage:
    uv run --extra train python scripts/results/plot_adaptation_rq4.py
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from energy_storage.viz import AQUA, BASELINE, BLUE, INK, RED, SECONDARY, SURFACE, style_axis

SHIFTS = ["fuel-step", "capacity-loss", "cold-regime"]

# {arm: {"pre": (means, stds), "post": (means, stds)}}
ARMS = {
    "adaptive-dyna": {
        "pre": ([66.5, 63.4, 71.2], [10.3, 14.3, 17.4]),
        "post": ([37.1, 77.7, 74.0], [7.2, 4.6, 3.1]),
        "color": AQUA,
    },
    "frozen-dyna": {
        "pre": ([88.5, 90.2, 90.3], [8.0, 7.2, 7.2]),
        "post": ([53.4, 88.7, 82.9], [8.1, 3.5, 4.2]),
        "color": RED,
    },
    "frozen-ideal": {
        "pre": ([91.8, 90.5, 90.6], [5.0, 5.1, 5.1]),
        "post": ([41.6, 91.1, 87.6], [8.7, 2.1, 2.0]),
        "color": BLUE,
    },
}

POST_GAP = [16.3, 11.0, 8.9]  # frozen-dyna minus adaptive-dyna, post-shift (paper text)


def panel(ax, window, title):
    n_arms = len(ARMS)
    width = 0.24
    x = np.arange(len(SHIFTS))
    offsets = np.linspace(-(n_arms - 1) / 2, (n_arms - 1) / 2, n_arms) * width

    for offset, (arm, spec) in zip(offsets, ARMS.items()):
        means, stds = spec[window]
        ax.bar(x + offset, means, width=width * 0.92, yerr=stds, color=spec["color"],
               capsize=3, label=arm,
               error_kw={"ecolor": INK, "elinewidth": 1.0, "capthick": 1.0}, zorder=2)

    ax.set_xticks(list(x))
    ax.set_xticklabels(SHIFTS, fontsize=10, color=SECONDARY)
    ax.set_ylim(0, 112)
    style_axis(ax, title)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("diagnostics"))
    args = parser.parse_args()
    args.out.mkdir(exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6.6), facecolor=SURFACE)
    panel(ax1, "pre", "Pre-shift (days 1-29): adaptive-dyna already trails")
    ax1.set_ylabel("Capture: % of same-window oracle", fontsize=9, color=SECONDARY)
    panel(ax2, "post", "Post-shift (day 30 on): adaptive-dyna never wins")

    x = np.arange(len(SHIFTS))
    n_arms = len(ARMS)
    width = 0.24
    offsets = np.linspace(-(n_arms - 1) / 2, (n_arms - 1) / 2, n_arms) * width
    adaptive_off, frozen_off = offsets[0], offsets[1]
    for xi, gap in zip(x, POST_GAP):
        top = max(ARMS["adaptive-dyna"]["post"][0][xi] + ARMS["adaptive-dyna"]["post"][1][xi],
                  ARMS["frozen-dyna"]["post"][0][xi] + ARMS["frozen-dyna"]["post"][1][xi]) + 6
        ax2.plot([xi + adaptive_off, xi + adaptive_off, xi + frozen_off, xi + frozen_off],
                 [top - 1.5, top, top, top - 1.5], color=BASELINE, lw=1)
        ax2.text(xi + (adaptive_off + frozen_off) / 2, top + 1.2, f"-{gap:.1f}pp",
                  ha="center", va="bottom", fontsize=8.5, color=SECONDARY)

    legend = ax2.legend(loc="lower center", bbox_to_anchor=(-0.08, -0.30), ncol=3, frameon=False, fontsize=9.5)
    for text in legend.get_texts():
        text.set_color(SECONDARY)

    fig.suptitle(
        "Experiment 4 (RQ4): can an in-imagination adaptation loop recover from persistent regime shifts?",
        fontsize=11, color=INK, x=0.02, ha="left", y=0.99,
    )
    fig.text(
        0.02, 0.90,
        "No. Adaptive-dyna's post-shift capture trails frozen-dyna (simply never updating) by 16.3, 11.0 and 8.9 points on\n"
        "fuel-step, capacity-loss and cold-regime, and it is already behind before the shift even begins: nightly refitting on a\n"
        "rolling window quietly drifts the market model out of its own calibration band on stationary data, days before any\n"
        "shift arrives. frozen-ideal (the unlimited-data policy, also frozen) is shown as a ceiling reference.",
        fontsize=8.5, color=SECONDARY, ha="left", va="top",
    )

    fig.tight_layout(rect=(0, 0.07, 1, 0.72))
    path = args.out / "adaptation_rq4.png"
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
