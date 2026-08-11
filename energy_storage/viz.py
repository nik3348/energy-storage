"""Shared plotting helpers for the figure scripts.

Matplotlib is a train-extra dependency, so this module is only imported by
scripts that already require the train extra — never by the core package.
"""

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
