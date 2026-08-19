#!/usr/bin/env python
"""Figure for RQ3: does the gated world model degrade gracefully under
transient shocks it never trained on?

Values are Table tab:robustness-scenarios (docs/main.tex), zero-shot capture
as % of the same-regime rolling-horizon oracle, pooled ratio ± leave-one-
episode-out jackknife std, 20 paired 14-day episodes. No policy saw any
shock during training. There is no backing results/ JSON for this table
(scripts/results/measure_scenario_robustness.py prints rather than saves),
so the numbers are hardcoded from the published table.

Usage:
    uv run --extra train python scripts/results/plot_scenario_robustness.py
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from energy_storage.viz import AQUA, BASELINE, BLUE, INK, SECONDARY, SURFACE, YELLOW, style_axis

SCENARIOS = ["Calm", "Cold-snap", "Heat-wave", "Fuel-shock", "Drought", "Plant-outage"]

# {arm: (means, stds)} aligned to SCENARIOS, dyna placed in the middle
ARMS = {
    "replay-365": ([38.9, 43.7, 33.1, 19.1, 43.0, 47.1], [32.4, 6.9, 18.2, 9.7, 43.4, 14.8], YELLOW),
    "dyna-365": ([71.1, 90.0, 83.1, 58.1, 72.9, 87.3], [14.8, 2.6, 8.2, 6.4, 18.2, 3.5], AQUA),
    "unlimited-data": ([72.6, 90.8, 83.0, 56.4, 72.1, 89.4], [13.1, 1.9, 7.5, 6.4, 18.3, 2.3], BLUE),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("diagnostics"))
    args = parser.parse_args()
    args.out.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 6.6), facecolor=SURFACE)

    n_arms = len(ARMS)
    width = 0.25
    x = np.arange(len(SCENARIOS))
    offsets = np.linspace(-(n_arms - 1) / 2, (n_arms - 1) / 2, n_arms) * width

    for offset, (arm, (means, stds, color)) in zip(offsets, ARMS.items()):
        bars = ax.bar(x + offset, means, width=width * 0.92, yerr=stds, color=color,
                       capsize=3, label=arm,
                       error_kw={"ecolor": INK, "elinewidth": 1.0, "capthick": 1.0}, zorder=2)
        if arm == "dyna-365":
            # Label only the two headline points (calm baseline, hardest shock);
            # labeling all six crowds the 100%-oracle line on the tall bars.
            for scenario, xi, m, s in zip(SCENARIOS, x + offset, means, stds):
                if scenario in ("Calm", "Fuel-shock"):
                    ax.text(xi, m + s + 3, f"{m:.1f}", ha="center", va="bottom", fontsize=9, color=INK)

    ax.axhline(100, color=INK, linewidth=1, linestyle="--", zorder=1)
    ax.text(len(SCENARIOS) - 0.5, 102, "same-regime oracle = 100%", ha="right", va="bottom",
            fontsize=8.5, color=SECONDARY)

    ax.set_xticks(list(x))
    ax.set_xticklabels(SCENARIOS, fontsize=10, color=SECONDARY)
    ax.set_ylabel("Zero-shot capture: % of same-regime rolling-horizon oracle", fontsize=9, color=SECONDARY)
    ax.set_ylim(0, 120)
    style_axis(ax, "No policy saw any shock during training: calm control plus five transient shocks, 20 paired 14-day episodes")

    legend = ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.30), ncol=3, frameon=False, fontsize=9.5)
    for text in legend.get_texts():
        text.set_color(SECONDARY)

    fig.suptitle(
        "Experiment 3 (RQ3): does the gated world model degrade gracefully under transient shocks it never trained on?",
        fontsize=11, color=INK, x=0.02, ha="left", y=0.99,
    )
    fig.text(
        0.02, 0.90,
        "Yes, and more calm-regime data doesn't buy more of it. Dyna-365 tracks the unlimited-data ceiling shock for shock\n"
        "on two orders of magnitude less real data; no shock separates the two arms by more than their jackknife margins.\n"
        "Replay is the weakest arm throughout. Fuel-shock is hardest for every policy; cold-snap and plant-outage actually\n"
        "lift capture above calm, since both mostly raise a price level the 24h window already conveys.",
        fontsize=8.5, color=SECONDARY, ha="left", va="top",
    )

    fig.tight_layout(rect=(0, 0.06, 1, 0.72))
    path = args.out / "scenario_robustness_rq3.png"
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
