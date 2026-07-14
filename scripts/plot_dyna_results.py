#!/usr/bin/env python
"""The Dyna experiment headline figure: % of hindsight optimum vs real-data
budget D, one line per arm, against the fixed reference ladder.

Numbers are the in-arm best checkpoints from the July 2026 sweep (train_dyna.py,
seed 0, 20 paired eval episodes at seed0=1,000,000, hindsight net $28.03;
full tables in wandb job_type=train-dyna). Arm C exists only at D=365 —
the market-model validation gate refused certification at 90 and 1095 days.

Usage:
    uv run --extra train python scripts/plot_dyna_results.py
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from energy_storage.viz import AQUA, BASELINE, BLUE, MUTED, RED, SECONDARY, SURFACE, YELLOW, style_axis

BUDGETS = [90, 365, 1095]

# % of hindsight optimum, in-arm best checkpoint (final in comments).
ONLINE = {90: -18.0, 365: -16.3, 1095: 9.0}  # final: -17.4 / -37.0 / +1.3
REPLAY = {90: 11.4, 365: 38.8, 1095: 42.1}  # final: -11.5 / 54.1 / 48.6
DYNA = {365: 71.0}  # final: 71.5

CEILING = 72.5  # sac-year-v3 best (1M steps, ~30 years of fresh days)
ORACLE = 99.9  # rolling-horizon oracle
HEURISTIC = -10.9
IDLE = -47.6


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("diagnostics"))
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(8.5, 5.5), facecolor=SURFACE)

    for pct, label, color, style in [
        (ORACLE, "rolling-horizon oracle (99.9%)", MUTED, ":"),
        (CEILING, "model-free SAC, unlimited data (72.5%)", SECONDARY, "--"),
        (HEURISTIC, "heuristic (-10.9%)", MUTED, ":"),
        (IDLE, "idle (-47.6%)", MUTED, ":"),
    ]:
        ax.axhline(pct, color=color, linestyle=style, linewidth=1.2)
        ax.text(1250, pct + 1.5, label, fontsize=8, color=SECONDARY, ha="right")
    ax.axhline(0, color=BASELINE, linewidth=1)

    ax.plot(BUDGETS, [ONLINE[d] for d in BUDGETS], marker="o", color=YELLOW, linewidth=2, label="A online (24·D real steps)")
    ax.plot(BUDGETS, [REPLAY[d] for d in BUDGETS], marker="s", color=BLUE, linewidth=2, label="B replay (stored days, 500k steps)")
    ax.plot([365], [DYNA[365]], marker="*", markersize=16, color=RED, linestyle="none", label="C dyna (learned model, 500k steps)")
    for d in (90, 1095):
        ax.plot([d], [-60], marker="x", color=RED, markersize=8, clip_on=False)
    ax.text(90, -57, "gate refused", fontsize=7, color=RED, ha="center")
    ax.text(1095, -57, "gate refused", fontsize=7, color=RED, ha="center")

    ax.set_xscale("log")
    ax.set_xticks(BUDGETS)
    ax.set_xticklabels([str(d) for d in BUDGETS])
    ax.set_xlabel("real market history budget D (days, log scale)", color=MUTED, fontsize=9)
    ax.set_ylabel("% of hindsight optimum", color=MUTED, fontsize=9)
    ax.set_ylim(-60, 108)
    style_axis(ax, "Sample efficiency: policy quality vs real-data budget (20 paired 14-day episodes)")
    ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=SECONDARY)

    fig.tight_layout()
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "dyna_sample_efficiency.png"
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
