#!/usr/bin/env python
"""The Dyna experiment headline figure: % of hindsight optimum vs real-data
budget D, one line per arm, against the fixed reference ladder.

Numbers come from results/budget_sweep.json (written by
scripts/measure_budget_sweep.py), which re-scores each stored checkpoint on 20
paired held-out episodes at seed0=1,000,000. Error bars are the
leave-one-episode-out jackknife std on pooled capture, the same estimator the
results tables use, so the figure and the tables cannot drift apart. Arm C
exists only at D=365 (the market-model validation gate refused certification at
90 and 1095 days).

Usage:
    uv run --extra train python scripts/measure_budget_sweep.py   # refresh data
    uv run --extra train python scripts/plot_dyna_results.py
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from energy_storage.viz import AQUA, BASELINE, BLUE, MUTED, RED, SECONDARY, SURFACE, YELLOW, style_axis

BUDGETS = [90, 365, 1095]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/budget_sweep.json"))
    parser.add_argument("--out", type=Path, default=Path("diagnostics"))
    args = parser.parse_args()

    if not args.results.exists():
        raise SystemExit(
            f"{args.results} not found - run scripts/measure_budget_sweep.py first"
        )
    data = json.loads(args.results.read_text())
    arms = data["arms"]

    def pct(name: str) -> tuple[float, float]:
        a = arms[name]
        return a["capture_pct"], a["capture_std_pct"]

    fig, ax = plt.subplots(figsize=(8.5, 5.5), facecolor=SURFACE)

    # Reference ladder. The model-free ceiling carries a shaded margin because
    # whether dyna reaches it is the figure's central comparison; the remaining
    # references are context and carry their margin in the label only.
    ceiling, ceiling_std = pct("unlimited-data")
    ax.axhspan(
        ceiling - ceiling_std, ceiling + ceiling_std,
        color=SECONDARY, alpha=0.12, linewidth=0,
    )
    for value, std, label, color, style in [
        (*pct("rolling-oracle"), "rolling-horizon oracle", MUTED, ":"),
        (ceiling, ceiling_std, "model-free SAC, unlimited data", SECONDARY, "--"),
        (*pct("heuristic"), "heuristic", MUTED, ":"),
        (*pct("idle"), "idle", MUTED, ":"),
    ]:
        ax.axhline(value, color=color, linestyle=style, linewidth=1.2)
        ax.text(
            1250, value + 1.5, f"{label} ({value:.1f}$\\pm${std:.1f}%)",
            fontsize=8, color=SECONDARY, ha="right",
        )
    ax.axhline(0, color=BASELINE, linewidth=1)

    for arm, marker, color, label in [
        ("online", "o", YELLOW, "A online (24·D real steps)"),
        ("replay", "s", BLUE, "B replay (stored days, 500k steps)"),
    ]:
        budgets = [d for d in BUDGETS if f"{arm}-d{d}" in arms]
        values = [pct(f"{arm}-d{d}")[0] for d in budgets]
        errors = [pct(f"{arm}-d{d}")[1] for d in budgets]
        ax.errorbar(
            budgets, values, yerr=errors, marker=marker, color=color, linewidth=2,
            capsize=4, elinewidth=1.2, label=label,
        )

    dyna, dyna_std = pct("dyna-d365")
    ax.errorbar(
        [365], [dyna], yerr=[dyna_std], marker="*", markersize=16, color=RED,
        linestyle="none", capsize=4, elinewidth=1.2,
        label="C dyna (learned model, 500k steps)",
    )
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
    style_axis(
        ax,
        f"Sample efficiency: policy quality vs real-data budget "
        f"({data['episodes']} paired {data['episode_days']}-day episodes, ±jackknife std)",
    )
    ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=SECONDARY)

    fig.tight_layout()
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "dyna_sample_efficiency.png"
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
