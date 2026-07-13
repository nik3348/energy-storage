"""Shared plotting helpers for the diagnostic/visualization scripts.

Matplotlib is a train-extra dependency, so this module is only imported by
scripts that already require the train extra — never by the core package.
"""

import matplotlib.pyplot as plt
import numpy as np

# Reference dataviz palette (light mode), fixed slot order.
BLUE, AQUA, YELLOW = "#2a78d6", "#1baf7a", "#eda100"
RED = "#e34948"  # diverging counterpart to blue
INK, SECONDARY, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"

# Tech -> fill color for the supply-stack plot. Ordered cheapest-first by
# the conventional merit order so the legend reads bottom-to-top.
TECH_COLORS = {
    "solar": YELLOW,
    "wind": AQUA,
    "geothermal": "#6a8f4e",
    "hydro": BLUE,
    "coal": "#4a4a48",
    "gas": RED,
}


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
