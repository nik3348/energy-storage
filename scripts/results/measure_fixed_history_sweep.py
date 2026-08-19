#!/usr/bin/env python
"""Score the fixed-history training-seed sweep: does the dyna-over-replay gap
survive re-running the training, with the real-data draw held fixed?

Every checkpoint here was trained on seed 0's 365 stored days and seed 0's
gated ensemble (each arm re-run under pipeline/train_dyna_arms.py with --history and
--model pinned to seed 0 but --seed varied), so the seed indexes the
training run only. Each arm is scored on the same 20 paired held-out episodes
as everywhere else, which makes two error margins meaningful and distinct:

- capture jackknife, the leave-one-episode-out spread of the pooled ratio, i.e.
  market-week variance within one training run;
- across-seed std, the spread of those captures over training runs, i.e. the
  quantity this sweep exists to measure.

The headline is the paired per-episode difference (dyna - replay at the same
training seed, same 20 episodes), whose standard error cancels the market-week
variance that dominates both marginal numbers.

Usage:
    uv run --extra train python scripts/results/measure_fixed_history_sweep.py
"""

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC

from energy_storage.baselines import capture_jackknife, evaluate, model_policy
from energy_storage.env import EnvConfig
from energy_storage.oracle import evaluate_hindsight

ARMS = ["dyna", "replay"]


def checkpoint_dir(models: Path, arm: str, seed: int, budget: int) -> Path:
    """Seed 0 was trained before the sweep, directly on the pinned artifacts,
    so it lives under the original name and counts as the seed-0 point."""
    if seed == 0:
        return models / "dyna" / f"{arm}-d{budget}-s{seed}"
    return models / "dyna" / f"{arm}-d{budget}-fixedhist-s{seed}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--budget-days", type=int, default=365)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--episode-days", type=int, default=14)
    parser.add_argument("--seed0", type=int, default=1_000_000)
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--checkpoint", default="best_model.zip")
    parser.add_argument(
        "--out", type=Path, default=Path("results/fixed_history_sweep.json")
    )
    args = parser.parse_args()

    config = EnvConfig(episode_days=args.episode_days)
    hindsight = evaluate_hindsight(config, args.episodes, args.seed0)
    hind_nets = np.asarray(hindsight["nets"])
    print(
        f"hindsight net ${hindsight['net']:.2f}±{hindsight['net_std']:.2f}"
        f"/episode = 100%\n"
    )

    runs: dict[str, dict[int, dict]] = {arm: {} for arm in ARMS}
    for seed in args.seeds:
        for arm in ARMS:
            path = checkpoint_dir(args.models, arm, seed, args.budget_days)
            ckpt = path / args.checkpoint
            if not ckpt.exists():
                print(f"skip {arm} s{seed}: {ckpt} not found")
                continue
            stats = evaluate(
                model_policy(SAC.load(ckpt)), config, args.episodes, args.seed0
            )
            capture, jack = capture_jackknife(stats["nets"], hind_nets)
            runs[arm][seed] = {
                "checkpoint": str(ckpt),
                "net": stats["net"],
                "net_std": stats["net_std"],
                "capture": 100.0 * capture,
                "capture_jackknife": 100.0 * jack,
                "nets": np.asarray(stats["nets"]).tolist(),
            }
            print(
                f"{arm:<7} s{seed}  ${stats['net']:>6.2f}/ep  "
                f"{100 * capture:>5.1f}±{100 * jack:.1f}% of hindsight"
            )

    # Marginal spread over training runs, per arm.
    summary: dict[str, dict] = {}
    for arm in ARMS:
        caps = np.array([r["capture"] for r in runs[arm].values()])
        nets = np.array([r["net"] for r in runs[arm].values()])
        if not len(caps):
            continue
        summary[arm] = {
            "n_seeds": len(caps),
            "capture_mean": float(caps.mean()),
            "capture_std": float(caps.std(ddof=1)) if len(caps) > 1 else 0.0,
            "capture_min": float(caps.min()),
            "capture_max": float(caps.max()),
            "net_mean": float(nets.mean()),
            "net_std_across_seeds": float(nets.std(ddof=1)) if len(nets) > 1 else 0.0,
        }

    # The headline: paired per-episode dyna - replay at each training seed.
    paired: dict[str, dict] = {}
    for seed in args.seeds:
        if seed not in runs["dyna"] or seed not in runs["replay"]:
            continue
        diff = np.asarray(runs["dyna"][seed]["nets"]) - np.asarray(
            runs["replay"][seed]["nets"]
        )
        paired[str(seed)] = {
            "mean": float(diff.mean()),
            "se": float(diff.std(ddof=1) / np.sqrt(len(diff))),
            "capture_gap": runs["dyna"][seed]["capture"]
            - runs["replay"][seed]["capture"],
        }

    out = {
        "config": {
            "budget_days": args.budget_days,
            "episodes": args.episodes,
            "episode_days": args.episode_days,
            "seed0": args.seed0,
            "seeds": args.seeds,
            "checkpoint": args.checkpoint,
            "history": f"data/history-d{args.budget_days}-s0.npz",
            "market_model": f"models/market-model-d{args.budget_days}-s0.pt",
        },
        "hindsight": {"net": hindsight["net"], "net_std": hindsight["net_std"]},
        "runs": runs,
        "summary": summary,
        "paired_dyna_minus_replay": paired,
    }

    print("\narm      n  capture over training seeds        net $/ep")
    print("-" * 62)
    for arm, s in summary.items():
        print(
            f"{arm:<7} {s['n_seeds']:>2}  {s['capture_mean']:>5.1f}±{s['capture_std']:.1f}% "
            f"(range {s['capture_min']:.1f}-{s['capture_max']:.1f})  "
            f"${s['net_mean']:.2f}±{s['net_std_across_seeds']:.2f}"
        )

    if paired:
        diffs = np.array([p["mean"] for p in paired.values()])
        print("\npaired dyna - replay, same training seed and same 20 episodes:")
        for seed, p in paired.items():
            print(
                f"  s{seed}  {p['mean']:>+6.2f} ± {p['se']:.2f} $/ep "
                f"({p['capture_gap']:+.1f} capture points)"
            )
        if len(diffs) > 1:
            across = float(diffs.std(ddof=1) / np.sqrt(len(diffs)))
            out["paired_dyna_minus_replay"]["across_seeds"] = {
                "mean": float(diffs.mean()),
                "sem": across,
                "n": len(diffs),
                "n_positive": int((diffs > 0).sum()),
            }
            print(
                f"  across {len(diffs)} training seeds: "
                f"{diffs.mean():+.2f} ± {across:.2f} $/ep (SEM), "
                f"{(diffs > 0).sum()}/{len(diffs)} positive"
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
