#!/usr/bin/env python
"""Collect D days of real market history — the Dyna experiment's data budget.

Every experiment arm sees exactly this history and nothing else, so collect
once per (seed, budget) and point the arms at the file.

Usage:
    uv run python scripts/collect_history.py --days 365 --seed 0
"""

import argparse
from pathlib import Path

import numpy as np

from energy_storage.env import default_market_config
from energy_storage.market_history import MarketHistory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None, help="default data/history-d{days}-s{seed}.npz")
    args = parser.parse_args()

    out = args.out or Path("data") / f"history-d{args.days}-s{args.seed}.npz"
    history = MarketHistory.collect(default_market_config(), seed=args.seed, days=args.days)
    out.parent.mkdir(parents=True, exist_ok=True)
    history.save(out)
    prices = history.prices.ravel()
    print(
        f"wrote {out}: {len(history)} days, median ${float(np.median(prices)):.1f}, "
        f"mean ${prices.mean():.1f}"
    )


if __name__ == "__main__":
    main()
