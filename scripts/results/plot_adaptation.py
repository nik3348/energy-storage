#!/usr/bin/env python
"""Recovery curves and summary table for the adaptation experiment.

Reads the per-day CSVs written by pipeline/run_adaptation_deployments.py for one shift and plots
capture(t) = 7-day-windowed policy net / rolling-horizon-oracle net on the
same seed (the oracle re-plans on observed prices, so it adapts instantly —
capture is "fraction of the instantly-adapted optimum retained"). Prints
recovery lag (days after T until capture regains 90% of the arm's own
pre-shift level) and cumulative post-shift regret vs the oracle.

Usage:
    uv run --extra train python scripts/results/plot_adaptation.py --shift fuel-step
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from energy_storage.viz import AQUA, BASELINE, BLUE, MUTED, RED, SECONDARY, SURFACE, YELLOW, style_axis

WINDOW = 7
ARM_COLORS = {
    "adaptive-dyna": AQUA,
    "frozen-dyna": RED,
    "frozen-ideal": BLUE,
    "online-ft": YELLOW,
    "heuristic": MUTED,
}


def load_runs(directory: Path) -> dict[str, dict[int, dict[str, np.ndarray]]]:
    """{arm: {seed: named column arrays}} from {arm}-s{seed}.csv files."""
    runs: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    for path in sorted(directory.glob("*-s*.csv")):
        arm, _, seed = path.stem.rpartition("-s")
        data = np.genfromtxt(path, delimiter=",", names=True)
        columns = {name: np.atleast_1d(data[name]) for name in data.dtype.names}
        runs.setdefault(arm, {})[int(seed)] = columns
    return runs


def rolling_sum(values: np.ndarray, window: int) -> np.ndarray:
    return np.convolve(values, np.ones(window), mode="valid")


MIN_ORACLE_WINDOW = 5.0  # $; calm-week oracle nets are often <$1 — ratios there are noise


def capture_curve(arm_net: np.ndarray, oracle_net: np.ndarray) -> np.ndarray:
    """Windowed capture; day i of the result is the window ending on day
    i + WINDOW - 1 of the stream."""
    denominator = rolling_sum(oracle_net, WINDOW)
    curve = rolling_sum(arm_net, WINDOW) / np.where(
        np.abs(denominator) < MIN_ORACLE_WINDOW, np.nan, denominator
    )
    return curve


def pooled_capture_curve(arm_nets: list[np.ndarray], oracle_nets: list[np.ndarray]) -> np.ndarray:
    """Ratio of sums across seeds (not mean of per-seed ratios): a stable
    estimator when individual oracle windows are near zero."""
    n = min(map(len, arm_nets + oracle_nets))
    numerator = rolling_sum(np.sum([a[:n] for a in arm_nets], axis=0), WINDOW)
    denominator = rolling_sum(np.sum([o[:n] for o in oracle_nets], axis=0), WINDOW)
    return numerator / np.where(np.abs(denominator) < MIN_ORACLE_WINDOW, np.nan, denominator)


def recovery_lag(capture: np.ndarray, pre_days: int) -> float:
    """Days after T until capture regains 90% of its own pre-shift mean.
    Windowed day i ends on stream day i + WINDOW - 1."""
    pre = capture[: pre_days - WINDOW + 1]  # windows fully inside the calm period
    if len(pre) == 0 or np.all(np.isnan(pre)):
        return np.nan
    target = 0.9 * np.nanmean(pre)
    # First window fully inside the post-shift period:
    start = pre_days
    for i in range(start, len(capture)):
        if capture[i] >= target:
            return i - start + 1.0  # windowed day i ends on stream day i+WINDOW-1
    return np.inf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shift", required=True)
    parser.add_argument("--results", type=Path, default=Path("results/adaptation"))
    parser.add_argument("--pre-days", type=int, default=30)
    parser.add_argument("--out", type=Path, default=Path("diagnostics"))
    args = parser.parse_args()

    runs = load_runs(args.results / args.shift)
    if "oracle" not in runs:
        raise SystemExit(f"no oracle runs in {args.results / args.shift} — run the oracle arm first")
    oracle = runs.pop("oracle")

    fig, (ax, ax_nll) = plt.subplots(
        2, 1, figsize=(9.5, 7), facecolor=SURFACE, sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    t_shift = args.pre_days

    print(f"{args.shift}: capture vs rolling-horizon oracle, {WINDOW}-day windows")
    header = f"{'arm':<16} {'seeds':>5} {'pre capture':>12} {'post capture':>13} {'recovery lag':>13} {'regret $':>9}"
    print(header)
    print("-" * len(header))

    for arm, seeds in sorted(runs.items()):
        curves, regrets, arm_nets, oracle_nets = [], [], [], []
        for seed, columns in seeds.items():
            if seed not in oracle:
                print(f"  (skipping {arm} seed {seed}: no paired oracle run)")
                continue
            oracle_net = oracle[seed]["net"]
            n = min(len(columns["net"]), len(oracle_net))
            curves.append(capture_curve(columns["net"][:n], oracle_net[:n]))
            regrets.append(float(np.sum(oracle_net[t_shift:n] - columns["net"][t_shift:n])))
            arm_nets.append(columns["net"][:n])
            oracle_nets.append(oracle_net[:n])
        if not curves:
            continue
        # Pooled (ratio-of-sums) curve is the headline: per-seed ratios blow up
        # whenever a seed's own calm-week oracle net is near zero.
        pooled = pooled_capture_curve(arm_nets, oracle_nets)
        n = min(min(map(len, curves)), len(pooled))
        stack = np.stack([c[:n] for c in curves])
        days = np.arange(n) + WINDOW - 1  # window end day
        color = ARM_COLORS.get(arm, SECONDARY)
        ax.plot(days, pooled[:n], color=color, linewidth=2, label=arm)
        if len(stack) > 1:
            sem = np.nanstd(stack, axis=0) / np.sqrt(len(stack))
            ax.fill_between(
                days, pooled[:n] - sem, pooled[:n] + sem, color=color, alpha=0.15, linewidth=0
            )

        lag = recovery_lag(pooled, args.pre_days)
        lag_txt = f"{lag:.0f}d" if np.isfinite(lag) else ("n/a" if np.isnan(lag) else "never")
        pre_cap = np.nanmean(pooled[: args.pre_days - WINDOW + 1])
        post_cap = np.nanmean(pooled[args.pre_days :])
        print(
            f"{arm:<16} {len(curves):>5} {pre_cap:>11.1%} {post_cap:>12.1%} "
            f"{lag_txt:>13} {np.mean(regrets):>8.2f}"
        )

    ax.axhline(1.0, color=BASELINE, linewidth=1)
    ax.axvline(t_shift, color=MUTED, linestyle="--", linewidth=1)
    ax.text(t_shift + 1, ax.get_ylim()[1] * 0.97, "shift begins (T)", fontsize=8, color=SECONDARY, va="top")
    ax.set_ylabel(f"capture ({WINDOW}-day net / oracle net)", color=MUTED, fontsize=9)
    style_axis(ax, f"Adaptation to a persistent {args.shift} shift (mean over paired seeds, ±SEM)")
    ax.legend(loc="lower left", frameon=False, fontsize=9, labelcolor=SECONDARY)

    # Detection trace: the adaptive arm's per-day model NLL, before/after the
    # nightly fine-tune.
    for arm, seeds in runs.items():
        traces = [c["nll"] for c in seeds.values() if "nll" in c]
        if not traces:
            continue
        n = min(map(len, traces))
        mean_nll = np.nanmean(np.stack([t[:n] for t in traces]), axis=0)
        ax_nll.plot(np.arange(n), mean_nll, color=ARM_COLORS.get(arm, SECONDARY), linewidth=1.5, label=f"{arm} NLL")
        posts = [c["nll_post"] for c in seeds.values() if "nll_post" in c]
        if posts:
            n2 = min(map(len, posts))
            mean_post = np.nanmean(np.stack([t[:n2] for t in posts]), axis=0)
            ax_nll.plot(np.arange(n2), mean_post, color=ARM_COLORS.get(arm, SECONDARY), linewidth=1.5, linestyle=":", label=f"{arm} NLL after fine-tune")
    ax_nll.axvline(t_shift, color=MUTED, linestyle="--", linewidth=1)
    ax_nll.set_xlabel("deployment day", color=MUTED, fontsize=9)
    ax_nll.set_ylabel("model NLL / day", color=MUTED, fontsize=9)
    style_axis(ax_nll, "Detection: observed-day NLL under the ensemble")
    if ax_nll.get_legend_handles_labels()[0]:
        ax_nll.legend(loc="upper right", frameon=False, fontsize=8, labelcolor=SECONDARY)

    fig.tight_layout()
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"adaptation_{args.shift}.png"
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
