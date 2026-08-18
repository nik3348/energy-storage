#!/usr/bin/env python
"""Error margins for the adaptation experiment's summary tables.

plot_adaptation.py reports pooled (ratio-of-sums) capture: the stable headline
estimator, but with no spread attached. Per-seed capture ratios are useless for
that purpose pre-shift (calm-week oracle nets sit near zero, so individual
ratios explode — the reason pooling exists). The statistically matched error
bar for a pooled ratio is the leave-one-seed-out jackknife: recompute the
pooled statistic with each deployment left out and scale the spread of those
estimates. This script reproduces exactly the estimators the write-up quotes
(nanmean of the pooled 7-day-windowed capture curve over a day range) and
attaches jackknife std across the paired deployments; regret is a plain
per-seed mean ± std.

Windows quoted: pre-shift (windows fully inside days 0..T-1), post-shift
(windows ending on day T..end), settled calm (windows inside days 20..29),
settled post (windows inside days 37..119).

Usage:
    uv run python scripts/adaptation_stats.py                 # all shifts
    uv run python scripts/adaptation_stats.py --shift fuel-step
"""

import argparse
from pathlib import Path

import numpy as np

ARMS = ["adaptive-dyna", "frozen-dyna", "frozen-ideal", "online-ft", "heuristic"]
WINDOW = 7
MIN_ORACLE_WINDOW = 5.0  # $; same floor as plot_adaptation.py


def load_runs(directory: Path) -> dict[str, dict[int, dict[str, np.ndarray]]]:
    runs: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    for path in sorted(directory.glob("*-s*.csv")):
        arm, _, seed = path.stem.rpartition("-s")
        data = np.genfromtxt(path, delimiter=",", names=True)
        columns = {name: np.atleast_1d(data[name]) for name in data.dtype.names}
        runs.setdefault(arm, {})[int(seed)] = columns
    return runs


def rolling_sum(values: np.ndarray, window: int) -> np.ndarray:
    return np.convolve(values, np.ones(window), mode="valid")


def pooled_capture_curve(arm_nets: list[np.ndarray], oracle_nets: list[np.ndarray]) -> np.ndarray:
    n = min(map(len, arm_nets + oracle_nets))
    numerator = rolling_sum(np.sum([a[:n] for a in arm_nets], axis=0), WINDOW)
    denominator = rolling_sum(np.sum([o[:n] for o in oracle_nets], axis=0), WINDOW)
    return numerator / np.where(np.abs(denominator) < MIN_ORACLE_WINDOW, np.nan, denominator)


def window_means(curve: np.ndarray, pre_days: int) -> dict[str, float]:
    """The four quoted statistics from one pooled capture curve. Windowed day i
    ends on stream day i + WINDOW - 1."""
    return {
        "pre": float(np.nanmean(curve[: pre_days - WINDOW + 1])),
        "post": float(np.nanmean(curve[pre_days:])),
        "calm": float(np.nanmean(curve[20 - WINDOW + 1 + 6 : 30 - WINDOW + 1])),  # ends 26..29
        "settled": float(np.nanmean(curve[43 - WINDOW + 1 : 120 - WINDOW + 1])),  # ends 43..119
    }


def jackknife(arm_nets: list[np.ndarray], oracle_nets: list[np.ndarray], pre_days: int):
    """Pooled statistics with leave-one-seed-out jackknife std."""
    full = window_means(pooled_capture_curve(arm_nets, oracle_nets), pre_days)
    n = len(arm_nets)
    leave_out = [
        window_means(
            pooled_capture_curve(
                [a for j, a in enumerate(arm_nets) if j != i],
                [o for j, o in enumerate(oracle_nets) if j != i],
            ),
            pre_days,
        )
        for i in range(n)
    ]
    stds = {}
    for key in full:
        theta = np.array([lo[key] for lo in leave_out])
        stds[key] = float(np.sqrt((n - 1) / n * np.nansum((theta - np.nanmean(theta)) ** 2)))
    return full, stds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/adaptation"))
    parser.add_argument("--shift", default=None, help="one shift; default: every subdirectory")
    parser.add_argument("--pre-days", type=int, default=30)
    args = parser.parse_args()

    shifts = [args.shift] if args.shift else sorted(p.name for p in args.results.iterdir() if p.is_dir())
    for shift in shifts:
        runs = load_runs(args.results / shift)
        oracle = runs.pop("oracle")
        print(f"\n{shift}: pooled capture ± jackknife std over paired deployments")
        header = (
            f"{'arm':<16} {'seeds':>5} {'pre':>14} {'post':>14} "
            f"{'calm 20-29':>14} {'settled 37-119':>15} {'regret $':>17}"
        )
        print(header)
        print("-" * len(header))
        for arm in ARMS:
            if arm not in runs:
                continue
            arm_nets, oracle_nets, regrets = [], [], []
            for seed, cols in sorted(runs[arm].items()):
                if seed not in oracle:
                    continue
                net, onet = cols["net"], oracle[seed]["net"]
                n = min(len(net), len(onet))
                arm_nets.append(net[:n])
                oracle_nets.append(onet[:n])
                regrets.append(float(np.sum(onet[args.pre_days:n] - net[args.pre_days:n])))
            full, stds = jackknife(arm_nets, oracle_nets, args.pre_days)
            cells = "".join(
                f" {100 * full[k]:6.1f}±{100 * stds[k]:4.1f}%" for k in ("pre", "post", "calm", "settled")
            )
            reg = f"{np.mean(regrets):8.2f}±{np.std(regrets, ddof=1):6.2f}"
            print(f"{arm:<16} {len(arm_nets):>5}{cells} {reg:>17}")

        # Paired difference frozen-dyna minus adaptive-dyna (same seeds), the
        # headline "adapting never beats freezing" claim with its own margin.
        nets = {}
        for arm in ("frozen-dyna", "adaptive-dyna"):
            nets[arm] = {
                seed: cols["net"] for seed, cols in sorted(runs[arm].items()) if seed in oracle
            }
        seeds = sorted(set(nets["frozen-dyna"]) & set(nets["adaptive-dyna"]))
        f_nets = [nets["frozen-dyna"][s] for s in seeds]
        a_nets = [nets["adaptive-dyna"][s] for s in seeds]
        o_nets = [oracle[s]["net"] for s in seeds]
        n_runs = len(seeds)

        def diff_stat(idx: list[int]) -> dict[str, float]:
            f = window_means(
                pooled_capture_curve([f_nets[i] for i in idx], [o_nets[i] for i in idx]), args.pre_days
            )
            a = window_means(
                pooled_capture_curve([a_nets[i] for i in idx], [o_nets[i] for i in idx]), args.pre_days
            )
            return {k: f[k] - a[k] for k in f}

        full = diff_stat(list(range(n_runs)))
        leave_out = [diff_stat([j for j in range(n_runs) if j != i]) for i in range(n_runs)]
        line = []
        for key in ("pre", "post"):
            theta = np.array([lo[key] for lo in leave_out])
            std = float(np.sqrt((n_runs - 1) / n_runs * np.nansum((theta - np.nanmean(theta)) ** 2)))
            line.append(f"{key} {100 * full[key]:+.1f}±{100 * std:.1f}pp")
        print(f"{'':<16} paired frozen-dyna minus adaptive-dyna: " + ", ".join(line))


if __name__ == "__main__":
    main()
