#!/usr/bin/env python
"""Figure for Experiment 5 (RQ5): does a physics prior need fewer real
transitions than a plain MLP to predict the battery's own dynamics?

Values are Table tab:pinn-accuracy (docs/main.tex): held-out pooled relative
MAE, uniform coverage, mean +/- std over ten fit seeds, against the default
(linear) battery physics. Hardcoded from the published table, same approach
as the other RQ figures (no backing results/ JSON at this exact metric).

mlp -> pinn-hard isolates the output clamp (feasibility only, no loss term);
pinn-hard -> pinn-soft adds the physics residual loss. Lower is better.

Usage:
    uv run --extra train python scripts/results/plot_pinn_rq5.py
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from energy_storage.viz import AQUA, BASELINE, BLUE, INK, SECONDARY, SURFACE, YELLOW, style_axis

N_BUDGETS = [500, 200, 100, 50, 25, 15, 10]

# {arm: (means, stds, color, bold_at)} aligned to N_BUDGETS; bold_at is the
# set of N where this arm is the published table's best point estimate.
ARMS = {
    "mlp": ([0.018, 0.034, 0.043, 0.076, 0.147, 0.358, 0.454],
            [0.003, 0.003, 0.003, 0.006, 0.009, 0.014, 0.029], BLUE, {100, 50, 25, 15, 10}),
    "pinn-hard (clamp only)": ([0.020, 0.041, 0.058, 0.128, 0.151, 0.498, 0.648],
                                [0.005, 0.005, 0.006, 0.010, 0.014, 0.043, 0.033], YELLOW, set()),
    "pinn-soft (clamp + residual)": ([0.014, 0.032, 0.065, 0.170, 0.187, 0.521, 0.504],
                                       [0.002, 0.003, 0.006, 0.020, 0.016, 0.011, 0.013], AQUA, {500, 200}),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("diagnostics"))
    args = parser.parse_args()
    args.out.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(9.5, 6.6), facecolor=SURFACE)

    ax.axvspan(9, 200, color=BASELINE, alpha=0.35, zorder=0)
    ax.text(42, 0.0115, "MLP wins or ties at every N here", fontsize=8.5, color=SECONDARY,
            ha="center", va="bottom")

    for arm, (means, stds, color, bold_at) in ARMS.items():
        x = np.array(N_BUDGETS)
        means = np.array(means)
        stds = np.array(stds)
        ax.errorbar(x, means, yerr=stds, marker="o", markersize=5, color=color,
                    linewidth=1.6, capsize=3, label=arm, zorder=3)
        for xi, m, n in zip(x, means, N_BUDGETS):
            if n in bold_at:
                ax.scatter([xi], [m], s=90, facecolors="none", edgecolors=INK, linewidths=1.3, zorder=4)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(0.01, 0.9)
    ax.set_xticks(N_BUDGETS)
    ax.set_xticklabels([str(n) for n in N_BUDGETS])
    ax.invert_xaxis()
    ax.minorticks_off()
    ax.set_xlabel("N: real transitions the fit sees (fewer → harder)", fontsize=9, color=SECONDARY)
    ax.set_ylabel("Held-out relative MAE (log scale, lower is better)", fontsize=9, color=SECONDARY)
    style_axis(ax, "Default (linear) battery physics, ten fit seeds per budget; open circles mark each budget's best point estimate")

    legend = ax.legend(loc="upper left", frameon=False, fontsize=9.5)
    for text in legend.get_texts():
        text.set_color(SECONDARY)

    fig.suptitle(
        "Experiment 5 (RQ5): does a physics prior need fewer real transitions than a plain MLP?",
        fontsize=11, color=INK, x=0.02, ha="left", y=0.99,
    )
    fig.text(
        0.02, 0.90,
        "No, and least of all in the small-N regime a PINN is meant to own. The soft-residual PINN only wins at N=500, and\n"
        "even its edge at N=200 sits inside one standard deviation. Below that the plain MLP wins outright and by widening\n"
        "margins: the battery's step map is smooth enough that an unconstrained network interpolates it from a handful of points.",
        fontsize=8.5, color=SECONDARY, ha="left", va="top",
    )

    fig.tight_layout(rect=(0, 0.02, 1, 0.72))
    path = args.out / "pinn_rq5.png"
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
