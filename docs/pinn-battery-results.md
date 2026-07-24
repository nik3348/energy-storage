# PINN battery surrogate: results (negative)

Companion to `docs/pinn-battery-design.md`. **Headline: modelling the battery as
a physics-informed network does not improve sample efficiency over a plain MLP
surrogate on this problem. The battery input–output map is smooth and
low-dimensional enough that an unconstrained MLP already learns it from a handful
of transitions; the physics prior can match that accuracy but not beat it, and
its soft-constraint form actively competes with the data fit on the easy
targets. The PINN's only robust advantage is qualitative — guaranteed physical
admissibility — not sample efficiency.** Reproduce with
`scripts/evaluate_battery_model.py` (→ `data/battery-model-stage1.npz`).

## What was compared

All arms predict `(dsoc, dsoh, grid_energy)` from `(soc, soh, power_frac)` and
are scored on held-out relative MAE (per-target absolute error / target std) on
an 8,000-point full-envelope test set, averaged over 3 fit seeds.

- **`mlp`** — plain 2×64 tanh MLP, three free heads, scale-balanced MSE.
- **`pinn-hard`** — same trunk/heads, but physics as *hard admissibility*: SoH
  monotone (`dsoh ≤ 0`), sign consistency (`sign grid = sign dsoc = sign
  power`), and SoC saturation at the bounds. No residual, no extra parameters.
- **`pinn-soft`** — `pinn-hard` plus a label-free physics residual on
  collocation points (energy/efficiency conservation with known nameplate
  capacity + learned efficiency; the degradation structural equation with
  learned cycle/calendar/stress rates), weighted by λ. Physical parameters get
  a faster learning rate on a longer schedule (the tuning below).

Two coverage regimes: **uniform** (training transitions drawn across the whole
envelope) and **narrow** (a gentle mid-SoC, low-power operating band, tested on
the full envelope — the realistic case where the agent later queries states the
log never covered).

## The three things tried (per the review), and what happened

1. **Constraint form and weight.** Four functional forms were built:
   (a) a soft residual between *free* grid and dsoc predictions — **degenerate**,
   its minimizer collapses both toward zero; (b) fully hard-encoded conservation
   deriving grid and soh_loss from a single magnitude head — **brittle**, the
   multiplicative parameters fail to co-identify across the 1e5 spread in target
   scales; (c) hard admissibility only (`pinn-hard`); (d) hard admissibility +
   soft residual with a *known* capacity so the energy term is well-posed
   (`pinn-soft`). The weight was swept λ ∈ {0, 0.1, 0.3, 1.0}. **λ = 0 was the
   best PINN variant at every budget; every λ > 0 made pooled accuracy worse.**

2. **Hyperparameter retuning.** With the residual on, a shared learning rate
   left the physical parameters stuck at their (wrong) init, so the residual
   enforced *wrong* physics. Giving them a 10× faster rate on a 1500-epoch
   schedule fixed identification — `eta` → 0.964 (true 0.949), `kcyc` → 3.3e-7
   (true 2.0e-7) — and recovered most of the loss the residual had caused. It
   did not overturn the ranking.

3. **Restricting samples.** The budget was pushed down to N ∈ {50, 25, 15, 10,
   5}. The PINN never closed the gap. The one place physics helped was the
   *degradation* target (`dsoh`) at the smallest N (e.g. N = 10: PINN ≈ 0.72 vs
   MLP ≈ 0.95 relative MAE) — the tiny-signal target the MLP cannot learn from a
   few points — but the PINN paid it back on `dsoc`/`grid`, the near-linear
   targets the MLP nails from two samples, so overall accuracy stayed behind.

## Headline numbers (uniform coverage, pooled relative MAE, 3 fit seeds)

| N | mlp | pinn-hard |
|---|-----|-----------|
| 100 | 0.044 | 0.061 |
| 50 | 0.072 | 0.121 |
| 25 | 0.142 | 0.147 |
| 15 | 0.355 | 0.492 |
| 10 | 0.450 | 0.632 |

The MLP leads at every budget. (The tuned `pinn-soft` arm, and the narrow-coverage
regime, are produced by the same sweep script — `pinn-soft`'s only edge is on the
`dsoh` target at N ≤ 10, never on pooled error.)

Physical-violation rates tell the qualitative half of the story: across all N
the PINN's wrong-sign-grid and rising-SoH rates are exactly **0.0** (structural),
while the MLP predicts wrong-sign grid up to ~2% of the time at N = 25 and
occasionally rising SoH — small, but the kind of exploitable "free lunch" a
Dyna-style agent actively seeks.

## Why the result is what it is (diagnosed cause)

The battery step is a smooth, near-piecewise-linear map of three bounded inputs.
Its hardest feature — saturation at the SoC bounds — is a kink an MLP fits from a
few points either side. There is simply little function-approximation difficulty
for a physics prior to relieve, which is the regime where PINNs are known *not*
to pay off (they shine on stiff/high-dimensional/sparsely-observed operators).
The soft residual, being a competing objective, can only match the data fit in
the best case and distorts the easy targets in practice.

The corollary is the honest positive claim the experiment *does* support: the
value of physics here is **admissibility, not accuracy**. A physics-structured
surrogate cannot hallucinate free energy or free degradation, which is a
correctness property a world model handed to an optimizing agent should have —
independent of its average error. Whether that property changes a downstream
SAC's transfer (the model-exploitation question) is left as future work; it is
*not* answered by predictive accuracy, and the accuracy result alone does not
motivate the expense.

## Status

Stage 1 (this document) is complete and gates Stage 2 (the SAC sample-efficiency
arms in the design doc) closed on the sample-efficiency hypothesis: there is no
predictive crossover to carry forward. The surrogate module
(`energy_storage/battery_model.py`), the `EnvConfig.battery_factory` hook, and
tests remain in the tree as the reusable substrate for the admissibility /
model-exploitation follow-up, should it be pursued.
