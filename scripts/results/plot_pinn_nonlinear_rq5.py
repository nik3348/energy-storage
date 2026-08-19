#!/usr/bin/env python
"""Figure for the Experiment 5 (RQ5) follow-up: was the map simply too easy?

Reruns the accuracy sweep against a deliberately nonlinear degradation law
(quadratic cycle-stress exponent, end-of-life knee) instead of the default
near-linear physics. Values are Table tab:pinn-nonlinear (docs/main.tex):
held-out pooled relative MAE, mean +/- std over ten fit seeds. Only mlp and
pinn-hard are compared here (not pinn-soft) -- pinn-soft's residual loss is
hard-coded to the *original* linear degradation formula, so testing it
unchanged against this new nonlinear ground truth would silently be a
wrong-physics test, which Table tab:pinn-wrong runs deliberately instead.

Note the normaliser is this ground truth's own target spread, so absolute
levels are not comparable to plot_pinn_rq5.py's linear-physics figure --
only the mlp-vs-pinn-hard ordering is.

Usage:
    uv run --extra train python scripts/results/plot_pinn_nonlinear_rq5.py
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from energy_storage.viz import BLUE, INK, SECONDARY, SURFACE, YELLOW, style_axis

N_BUDGETS = [500, 200, 100, 50, 25, 15, 10]

# {arm: (means, stds, color, bold_at)}
ARMS = {
    "mlp": ([0.024, 0.047, 0.060, 0.131, 0.178, 0.272, 0.386],
            [0.002, 0.005, 0.002, 0.011, 0.009, 0.017, 0.013], BLUE, {200, 100, 50, 15, 10}),
    "pinn-hard (clamp only)": ([0.022, 0.048, 0.066, 0.136, 0.170, 0.425, 0.552],
                                [0.002, 0.003, 0.005, 0.009, 0.011, 0.021, 0.031], YELLOW, {500, 25}),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("diagnostics"))
    args = parser.parse_args()
    args.out.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(9.5, 6.6), facecolor=SURFACE)

    for arm, (means, stds, color, bold_at) in ARMS.items():
        x = np.array(N_BUDGETS)
        means = np.array(means)
        stds = np.array(stds)
        ax.errorbar(x, means, yerr=stds, marker="o", markersize=5, color=color,
                    linewidth=1.6, capsize=3, label=arm, zorder=3)
        for xi, m, n in zip(x, means, N_BUDGETS):
            if n in bold_at:
                ax.scatter([xi], [m], s=90, facecolors="none", edgecolors=INK, linewidths=1.3, zorder=4)

    ax.annotate("pinn-hard's only edge\n(new vs. linear physics)", xy=(500, 0.022), xytext=(260, 0.032),
                fontsize=8.5, color=SECONDARY, ha="left",
                arrowprops={"arrowstyle": "-", "color": SECONDARY, "lw": 0.9})
    ax.annotate("tied within 1 std", xy=(35, 0.16), xytext=(38, 0.24),
                fontsize=8.5, color=SECONDARY, ha="left",
                arrowprops={"arrowstyle": "-", "color": SECONDARY, "lw": 0.9})
    ax.annotate("mlp's margin widens again,\njust as under linear physics", xy=(14, 0.40), xytext=(18, 0.27),
                fontsize=8.5, color=SECONDARY, ha="left",
                arrowprops={"arrowstyle": "-", "color": SECONDARY, "lw": 0.9})

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(0.014, 0.75)
    ax.set_xticks(N_BUDGETS)
    ax.set_xticklabels([str(n) for n in N_BUDGETS])
    ax.invert_xaxis()
    ax.minorticks_off()
    ax.set_xlabel("N: real transitions the fit sees (fewer → harder)", fontsize=9, color=SECONDARY)
    ax.set_ylabel("Held-out relative MAE (log scale, lower is better)", fontsize=9, color=SECONDARY)
    style_axis(ax, "Nonlinear ground truth (quadratic cycle stress, end-of-life knee), ten fit seeds per budget")

    legend = ax.legend(loc="upper left", frameon=False, fontsize=9.5)
    for text in legend.get_texts():
        text.set_color(SECONDARY)

    fig.suptitle(
        "Experiment 5 follow-up (RQ5): was the map simply too easy for a physics prior to matter?",
        fontsize=11, color=INK, x=0.02, ha="left", y=0.99,
    )
    fig.text(
        0.02, 0.90,
        "No. Making the degradation law genuinely nonlinear does not rescue the physics-informed side. mlp still wins at every\n"
        "budget in the low-data regime that motivates a physics prior; pinn-hard's only edge is at N=500 (a new result, not seen\n"
        "under the original linear physics), and the two sit inside each other's seed spread at N=200, 50 and 25.",
        fontsize=8.5, color=SECONDARY, ha="left", va="top",
    )

    fig.tight_layout(rect=(0, 0.02, 1, 0.72))
    path = args.out / "pinn_nonlinear_rq5.png"
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
