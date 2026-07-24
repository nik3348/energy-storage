#!/usr/bin/env python
"""Stage 1 of the PINN battery experiment (docs/pinn-battery-design.md).

Fit the MLP and PINN battery surrogates on a shrinking budget of real
transitions and compare them on a held-out full-envelope test set — both on
predictive accuracy and on *physical-violation rate* (the fraction of test
points where a surrogate predicts an impossible transition: rising SoH or
wrong-sign grid energy). Two coverage regimes:

  uniform  training transitions drawn uniformly over the whole envelope
           (the easy case — dense coverage of a smooth 3-D map)
  narrow   training transitions from a gentle mid-SoC operating band, tested on
           the full envelope (the realistic case — the agent later queries the
           model in states the log never covered)

The gate for Stage 2: does a physics prior buy a *predictive* sample-efficiency
crossover? (Spoiler in the design doc's risk 2: the dynamics may be too simple.)

Usage:
    uv run --extra train python scripts/evaluate_battery_model.py
"""

import argparse
from pathlib import Path

import numpy as np

from energy_storage.battery import Battery, BatteryConfig
from energy_storage import battery_model as bm


def sample_narrow(n: int, cfg: BatteryConfig, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Transitions from a gentle mid-SoC, low-power, fresh-battery operating
    band — a stand-in for a cautious real controller's log."""
    rng = np.random.default_rng(seed)
    soc = rng.uniform(0.35, 0.65, n)
    soh = rng.uniform(0.95, 1.0, n)
    pf = rng.uniform(-0.4, 0.4, n)
    battery = Battery(cfg)
    x = np.empty((n, 3), np.float32)
    y = np.empty((n, 3), np.float32)
    for i in range(n):
        battery.reset(soc=float(soc[i]), soh=float(soh[i]))
        r = battery.step(bm._power_frac_to_kw(float(pf[i]), cfg), 1.0)
        x[i] = (soc[i], soh[i], pf[i])
        y[i] = (r.soc - soc[i], r.soh - soh[i], r.grid_energy_kwh)
    return x, y


def evaluate(pred: np.ndarray, y: np.ndarray, pf: np.ndarray, scale: np.ndarray) -> dict:
    err = np.abs(pred - y) / scale
    return {
        "pooled": float(err.mean()),
        "dsoc": float(err[:, 0].mean()),
        "dsoh": float(err[:, 1].mean()),
        "grid": float(err[:, 2].mean()),
        "free_deg": float(np.mean(pred[:, 1] > 1e-12)),
        "wrong_grid": float(np.mean(np.sign(pred[:, 2]) != np.sign(pf))),
    }


def main() -> None:
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--budgets", type=int, nargs="+", default=[500, 200, 100, 50, 25, 15, 10]
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--test-size", type=int, default=8000)
    parser.add_argument("--out", type=Path, default=Path("data/battery-model-stage1.npz"))
    args = parser.parse_args()

    cfg = BatteryConfig()
    xt, yt = bm.sample_transitions(args.test_size, cfg, seed=99)  # full-envelope test
    pf_t = xt[:, 2]
    scale = np.maximum(yt.std(0), 1e-8)

    samplers = {
        "uniform": lambda n, s: bm.sample_transitions(n, cfg, seed=s),
        "narrow": lambda n, s: sample_narrow(n, cfg, seed=s),
    }
    # Three arms: the MLP baseline; the hard-admissibility PINN (no residual);
    # and the tuned soft-constraint PINN (residual on, physical params on a
    # faster LR with a longer schedule — the best the PINN was made to do).
    plain = bm.FitConfig(epochs=args.epochs)
    tuned = bm.FitConfig(epochs=1500, lr=1e-3, physics_weight=0.3, lr_physics=3e-2)
    arms = {
        "mlp": (bm.MLPBatteryModel, plain),
        "pinn-hard": (bm.PINNBatteryModel, plain),
        "pinn-soft": (bm.PINNBatteryModel, tuned),
    }
    records = []

    for regime, sampler in samplers.items():
        print(f"\n=== coverage: {regime} (test = full envelope) ===")
        header = f"{'N':>5} {'model':10} {'pooled':>8} {'dsoc':>7} {'dsoh':>7} {'grid':>7} {'freeDeg':>8} {'wrongGrid':>10}"
        print(header)
        print("-" * len(header))
        for n in args.budgets:
            x, y = sampler(n, 0)
            for name, (Cls, fit_cfg) in arms.items():
                agg: dict[str, list[float]] = {}
                for s in args.seeds:
                    m = Cls(cfg, hidden=args.hidden, seed=s)
                    bm.fit(m, x, y, fit_cfg, seed=s)
                    with torch.no_grad():
                        pred = m.forward(torch.from_numpy(xt)).numpy()
                    for k, v in evaluate(pred, yt, pf_t, scale).items():
                        agg.setdefault(k, []).append(v)
                mean = {k: float(np.mean(v)) for k, v in agg.items()}
                records.append({"regime": regime, "n": n, "model": name, **mean})
                print(
                    f"{n:>5} {name:10} {mean['pooled']:>8.4f} {mean['dsoc']:>7.3f} "
                    f"{mean['dsoh']:>7.3f} {mean['grid']:>7.3f} {mean['free_deg']:>8.4f} "
                    f"{mean['wrong_grid']:>10.4f}"
                )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    keys = ["regime", "n", "model", "pooled", "dsoc", "dsoh", "grid", "free_deg", "wrong_grid"]
    np.savez(
        args.out,
        **{k: np.array([r[k] for r in records]) for k in keys},
    )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
