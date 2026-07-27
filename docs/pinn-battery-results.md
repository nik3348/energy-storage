# PINN battery surrogate: results (negative)

Companion to `docs/pinn-battery-design.md`. **Headline: modelling the battery as
a physics-informed network does not improve sample efficiency over a plain MLP
surrogate on this problem. The battery input–output map is smooth and
low-dimensional enough that an unconstrained MLP already learns it from a handful
of transitions. A well-tuned soft physics residual edges the MLP only at the
*largest* budgets (N ≥ 200) and under favourable coverage; in the low-data
regime a PINN is supposed to own — and under realistic narrow coverage — it does
not help and often hurts. The PINN's only unconditional advantage is qualitative:
guaranteed physical admissibility (never a wrong-sign or free-degradation
prediction), which the MLP violates at rates that climb to 8–15% as data thins.**
Reproduce with
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
   (`pinn-soft`). The weight was swept λ ∈ {0, 0.1, 0.3, 1.0}. With default
   hyperparameters every λ > 0 made pooled accuracy worse; only after the
   retuning below did the residual ever help — and then only at the *largest*
   budgets.

2. **Hyperparameter retuning.** With the residual on, a shared learning rate
   left the physical parameters stuck at their (wrong) init, so the residual
   enforced *wrong* physics. Giving them a 10× faster rate on a 1500-epoch
   schedule fixed identification — `eta` → 0.964 (true 0.949), `kcyc` → 3.3e-7
   (true 2.0e-7). This turned the residual into a mild regulariser that **edges
   the MLP at N = 500 and 200** — but did not help, and slightly hurt, at small N.

3. **Restricting samples.** Pushed down to N ∈ {50, 25, 15, 10}. **The PINN does
   not win in the low-data regime** — the opposite of the usual PINN motivation.
   At small N the physical parameters are themselves under-identified, so the
   residual regularises toward *slightly wrong* physics, while the MLP already
   interpolates this smooth 3-D map from a handful of points. The only clear
   low-N help is on the tiny-signal degradation target (`dsoh`) at N = 10
   (`pinn-soft` 0.712 vs MLP 0.830), paid back on `dsoc`/`grid`.

## Headline numbers (pooled relative MAE, 3 fit seeds; **bold** = best at that N)

Uniform coverage (train and test span the envelope):

| N | mlp | pinn-hard | pinn-soft (tuned) |
|-----|-------|-----------|-------------------|
| 500 | 0.018 | 0.019 | **0.014** |
| 200 | 0.033 | 0.038 | **0.030** |
| 100 | **0.044** | 0.061 | 0.066 |
| 50 | **0.072** | 0.121 | 0.162 |
| 25 | **0.142** | 0.147 | 0.198 |
| 15 | **0.355** | 0.492 | 0.518 |
| 10 | **0.450** | 0.632 | 0.504 |

Under **narrow** (realistic) coverage the MLP wins at every N and the soft
residual is consistently worse (it over-constrains under biased data). So the
large-N edge is fragile: there is **no budget at which a physics prior gives a
robust sample-efficiency win**, and certainly none in the low-N regime PINNs are
supposed to own.

## Admissibility: the one unconditional advantage

The PINN's wrong-sign-grid and rising-SoH rates are exactly **0.0** at every N
and under both coverage regimes (structural). The MLP's climb steeply as data
thins — precisely the low-N regime — and become substantial:

| N (uniform) | MLP wrong-sign grid | MLP rising SoH (free degradation) |
|-----|------|------|
| 50 | 1.5% | 0.0% |
| 25 | 2.1% | 0.1% |
| 15 | 2.2% | 4.2% |
| 10 | 8.6% | 0.0% |

(Under narrow coverage the MLP's free-degradation rate reaches 15% at N = 10.)
These are exactly the exploitable "free lunches" — phantom arbitrage profit,
cost-free cycling — a Dyna-style optimiser would seek out. That the PINN forbids
them by construction is its real, if narrow, value here: **admissibility, not
accuracy.**

## Why the result is what it is (diagnosed cause)

The battery step is a smooth, near-piecewise-linear map of three bounded inputs.
Its hardest feature — saturation at the SoC bounds — is a kink an MLP fits from a
few points either side. There is simply little function-approximation difficulty
for a physics prior to relieve, which is the regime where PINNs are known *not*
to pay off (they shine on stiff/high-dimensional/sparsely-observed operators).
The soft residual is a competing objective: where labels are plentiful (large N)
and evenly spread it acts as a mild regulariser and can edge the MLP, but where
they are scarce (small N) or biased (narrow coverage) the physical parameters it
depends on are themselves under-identified, so it regularises toward slightly
wrong physics and hurts. Neither regime is the one the sample-efficiency
hypothesis needed.

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
