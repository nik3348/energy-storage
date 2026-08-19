#!/usr/bin/env python
"""Wrong-constraints ablation for the PINN battery surrogate (RQ5 follow-up).

The main comparison (evaluate_battery_model.py) found the tuned
soft-residual PINN beats the MLP only at the largest budgets. This ablation
asks *why*: is that win physics content, or would any constraint-shaped
regularizer do? We refit the same tuned soft-residual PINN with deliberately
wrong physics and watch what the win does:

  soft-eta-true    one-way efficiency frozen at the true value (0.949) —
                   controls for freezing itself, separately from wrongness
  soft-eta-wrong   efficiency frozen wrong (0.60): the energy identity now
                   pulls grid-energy magnitudes ~37% off
  soft-cap-wrong   nameplate capacity halved in the residual (50 vs 100 kWh);
                   trainable eta can compensate on charge (eta halves) but not
                   on discharge (it would need eta ~1.9 > 1)
  soft-cal-wrong   calendar-aging law flipped: (1.5 - soc) instead of
                   (0.5 + soc), i.e. an empty battery ages fastest

against the correct tuned PINN and the MLP, on identical data draws. If the
correct residual's large-N edge comes from real physics information, wrong
physics should erase or invert it; if wrong constraints do just as well, the
residual is acting as a generic smoother and the physics content is
irrelevant. Uniform coverage only (the regime where the soft PINN wins at
all).

Usage:
    uv run --extra train python scripts/results/evaluate_wrong_constraints.py
"""

import argparse
from pathlib import Path

import numpy as np

from energy_storage.battery import BatteryConfig
from energy_storage import battery_model as bm
from evaluate_battery_model import evaluate


def main() -> None:
    import torch
    import torch.nn as nn

    class FrozenEtaPINN(bm.PINNBatteryModel):
        """Soft PINN with one-way efficiency frozen (not learned) at a given
        value. logit_eta stays a parameter but is never used, so the residual
        holds the network to the frozen efficiency."""

        def __init__(self, cfg=None, hidden=64, seed=0, eta_value=0.6):
            super().__init__(cfg, hidden, seed)
            self.register_buffer("eta_frozen", torch.tensor(float(eta_value)))

        @property
        def eta(self):
            return self.eta_frozen

    class WrongCapPINN(bm.PINNBatteryModel):
        """Soft PINN whose energy residual believes a wrong nameplate capacity."""

        def __init__(self, cfg=None, hidden=64, seed=0, cap_value=50.0):
            super().__init__(cfg, hidden, seed)
            self.cap.fill_(float(cap_value))

    class WrongCalendarPINN(bm.PINNBatteryModel):
        """Soft PINN whose degradation residual flips the calendar-aging law:
        (1.5 - soc) instead of the true (0.5 + soc)."""

        def physics_residual(self, x):
            out = self.forward(x)
            dsoc, dsoh, grid = out[:, 0:1], out[:, 1:2], out[:, 2:3]
            soc, soh, pf = x[:, 0:1], x[:, 1:2], x[:, 2:3]
            charging = (pf >= 0.0).to(x.dtype)
            eta = self.eta
            stored = dsoc.abs() * self.cap * soh
            expected_grid = stored * (charging / eta + (1.0 - charging) * eta)
            res_energy = (grid.abs() - expected_grid) / self.target_scale[2]
            kcyc, kcal, stress = (
                torch.exp(self.log_kcyc),
                torch.exp(self.log_kcal),
                torch.exp(self.log_stress),
            )
            soh_loss_phys = stored * kcyc * (1.0 + stress * pf.abs()) + kcal * (1.5 - soc)
            res_deg = ((-dsoh) - soh_loss_phys) / self.target_scale[1]
            return torch.cat([res_energy, res_deg], dim=1)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budgets", type=int, nargs="+", default=[500, 200, 100, 50, 25, 15, 10])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 11)))
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--test-size", type=int, default=8000)
    parser.add_argument("--out", type=Path, default=Path("data/battery-model-wrong-constraints.npz"))
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip (budget, arm) cells already present in --out (same seed count only)",
    )
    args = parser.parse_args()

    cfg = BatteryConfig()
    true_eta_one_way = float(np.sqrt(cfg.round_trip_efficiency))
    xt, yt = bm.sample_transitions(args.test_size, cfg, seed=99)  # same test set as the main run
    pf_t = xt[:, 2]
    scale = np.maximum(yt.std(0), 1e-8)

    plain = bm.FitConfig(epochs=400)
    tuned = bm.FitConfig(epochs=1500, lr=1e-3, physics_weight=0.3, lr_physics=3e-2)
    arms: dict[str, tuple] = {
        "mlp": (lambda s: bm.MLPBatteryModel(cfg, hidden=args.hidden, seed=s), plain),
        "soft-correct": (lambda s: bm.PINNBatteryModel(cfg, hidden=args.hidden, seed=s), tuned),
        "soft-eta-true": (
            lambda s: FrozenEtaPINN(cfg, hidden=args.hidden, seed=s, eta_value=true_eta_one_way),
            tuned,
        ),
        "soft-eta-wrong": (
            lambda s: FrozenEtaPINN(cfg, hidden=args.hidden, seed=s, eta_value=0.60),
            tuned,
        ),
        "soft-cap-wrong": (
            lambda s: WrongCapPINN(cfg, hidden=args.hidden, seed=s, cap_value=cfg.capacity_kwh / 2),
            tuned,
        ),
        "soft-cal-wrong": (
            lambda s: WrongCalendarPINN(cfg, hidden=args.hidden, seed=s),
            tuned,
        ),
    }

    keys = ["n", "model"] + [
        k + suffix
        for k in ["pooled", "dsoc", "dsoh", "grid", "free_deg", "wrong_grid"]
        for suffix in ("", "_std")
    ]

    def save(records: list[dict]) -> None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.out,
            n_seeds=np.array(len(args.seeds)),
            **{k: np.array([r[k] for r in records]) for k in keys},
        )

    # This grid takes hours; checkpoint every cell and allow --resume so a crash
    # or a sleeping machine costs at most one (budget, arm) cell.
    records: list[dict] = []
    done: set[tuple[int, str]] = set()
    if args.resume and args.out.exists():
        prior = np.load(args.out)
        # Only reuse a checkpoint fitted with the same number of seeds, so a
        # cheap smoke run can never silently contaminate the real grid.
        prior_seeds = int(prior["n_seeds"]) if "n_seeds" in prior else -1
        if prior_seeds != len(args.seeds):
            print(
                f"ignoring checkpoint at {args.out}: fitted with {prior_seeds} seeds, "
                f"this run uses {len(args.seeds)}",
                flush=True,
            )
        else:
            for i in range(len(prior["n"])):
                rec = {k: prior[k][i] for k in keys}
                rec["n"], rec["model"] = int(rec["n"]), str(rec["model"])
                records.append(rec)
                done.add((rec["n"], rec["model"]))
            print(f"resuming: {len(done)} cells already done", flush=True)

    header = f"{'N':>5} {'model':16} {'pooled mean±std':>16} {'freeDeg':>8} {'wrongGrid':>10}"
    print("=== wrong-constraints ablation, uniform coverage, full-envelope test ===", flush=True)
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for n in args.budgets:
        x, y = bm.sample_transitions(n, cfg, seed=0)  # same training draws as the main run
        for name, (make, fit_cfg) in arms.items():
            if (n, name) in done:
                continue
            agg: dict[str, list[float]] = {}
            for s in args.seeds:
                m = make(s)
                bm.fit(m, x, y, fit_cfg, seed=s)
                with torch.no_grad():
                    pred = m.forward(torch.from_numpy(xt)).numpy()
                for k, v in evaluate(pred, yt, pf_t, scale).items():
                    agg.setdefault(k, []).append(v)
            mean = {k: float(np.mean(v)) for k, v in agg.items()}
            std = {f"{k}_std": float(np.std(v, ddof=1)) for k, v in agg.items()}
            records.append({"n": n, "model": name, **mean, **std})
            save(records)
            print(
                f"{n:>5} {name:16} {mean['pooled']:>8.4f}±{std['pooled_std']:<7.4f} "
                f"{mean['free_deg']:>8.4f} {mean['wrong_grid']:>10.4f}",
                flush=True,
            )

    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
