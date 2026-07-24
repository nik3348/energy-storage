# PINN battery surrogate: sample-efficiency experiment

Status: agreed design, pre-implementation. Companion experiment to the Dyna
market-model pipeline (`docs/dyna-design.md`).

## Framing

The Dyna pipeline is built on the structural insight that **battery dynamics are
known** — the operator owns the asset, so `Battery.step` is exact and there is
nothing to learn; only the *market* is modelled. This experiment deliberately
**inverts that assumption**: suppose the operator has *not* calibrated their
battery and must learn its input–output behaviour from a limited log of observed
transitions. Then a real question appears:

> Does baking the known battery physics into the surrogate as a **physics-informed
> neural network (PINN)** buy sample efficiency over a plain **MLP** surrogate —
> both as a predictive model and, downstream, as the world model a SAC agent
> trains against?

The hypothesis: the physics residuals are label-free (evaluable at any
`(soc, soh, power)` point in the operating envelope), so they substitute for
labels. The PINN should therefore reach a given accuracy — and train an equally
good policy — from **fewer real transitions** than the MLP. The experiment
sweeps the transition budget *down* and looks for the crossover N below which the
PINN wins.

To isolate the battery surrogate as the only variable, the **market stays exact**
(the real simulator) in every arm. This is the mirror image of the Dyna
experiment, which held the battery exact and learned the market.

## What the surrogate predicts

The env only reads three things out of a battery step (see `env.step`):

- `grid_energy_kwh` → profit (`-price · grid_energy / 1000`),
- `soh_loss` → monetized degradation cost, and
- the updated `soc`, `soh` → the next observation and the EoL termination check.

So the surrogate is a map

```
(soc, soh, power_frac)  ->  (Δsoc, Δsoh, grid_energy_kwh)
```

with `power_frac ∈ [-1, 1]` the env's action (fraction of max power), `dt = 1h`
fixed. Predicting **deltas** for the state (not absolute next values) keeps the
targets O(1e-3) and well-scaled; `soh_loss = -Δsoh`. Working in `soc`/`Δsoc`
(dimensionless) rather than kWh means the surrogate never needs the nameplate
capacity to advance state, which matters because under the "battery is unknown"
premise capacity, efficiency and the aging rates are exactly the quantities being
learned.

`LearnedBattery` wraps a fitted surrogate to duck-type `Battery`: `reset(soc,
soh)`, mutable `.soc`/`.soh`, and `step(power_kw, dt_h) -> BatteryStepResult`
(it converts `power_kw` back to a fraction with the known max-power rating, which
*is* known — it is a nameplate, not a learned dynamic). Same interface the env
already calls, so no policy or observation change.

## MLP baseline

Plain feed-forward net, `(3) -> hidden -> hidden -> (3)`, tanh activations,
regression on the three targets with a plain MSE loss over labelled transitions
only. No structural constraints on the outputs, no physics term. This is the
honest control: "just fit a black box to your transition log."

## PINN

Same trunk and parameter count, but physics enters two ways:

1. **Hard structural constraints (architecture).** The raw heads are passed
   through activations that make every prediction *physically admissible by
   construction*, so the network can never represent an impossible transition:
   - `Δsoh ≤ 0` always (SoH is monotone non-increasing): `Δsoh = -softplus(·)`.
   - Sign consistency `sign(grid_energy) = sign(power)` and `sign(Δsoc) =
     sign(power)`: charge magnitude and discharge magnitude come from
     non-negative (`softplus`) heads, gated by the sign of the input power.
   - Saturation near the SoC bounds: `Δsoc` is capped by the available
     head/foot-room `(soc_max − soc)` / `(soc − soc_min)` via a smooth min, so
     the model cannot push SoC out of `[soc_min, soc_max]`.

2. **Soft physics residuals (label-free loss).** On a batch of **collocation
   points** sampled uniformly across the envelope `(soc, soh, power_frac)` —
   *no labels needed* — penalize violation of the conservation identities the
   real battery obeys, with round-trip efficiency `η²` and calendar/cycle rates
   as trainable physical scalars:
   - **Energy/efficiency conservation.** Charging: energy into cells
     `= grid_energy · η`; discharging: energy out of grid `= drawn · η`. In soc
     terms this pins `Δsoc · capacity · soh` to `grid_energy · η^{±1}`, i.e. the
     ratio of grid energy to stored energy is fixed by a single efficiency, the
     same one on charge and discharge (round-trip = η²).
   - **Round-trip consistency.** A charge of `e` then a discharge of `e`
     returns `η² · e` to the grid — a two-point residual tying the charge and
     discharge branches through the *same* η.
   - **Degradation floor.** `soh_loss ≥ calendar_floor(soc)` even at
     `power = 0`: there is always calendar aging, and it is affine in soc.

   The MLP sees none of these; the PINN's total loss is
   `MSE_data(labelled) + λ · residual(collocation)`.

The contrast is deliberately clean: **identical capacity and optimizer; the only
difference is the physics** (hard admissibility + label-free residual). Any
sample-efficiency gap is attributable to the physics prior, not to a bigger or
better-tuned network.

## Stage 1 — surrogate accuracy vs sample budget (cheap, decisive)

Fit both surrogates on `N` real transitions sampled across the operating
envelope (Latin-hypercube-ish over `soc ∈ [soc_min, soc_max]`,
`soh ∈ [eol_soh, 1]`, `power_frac ∈ [-1, 1]`, labelled by the exact
`Battery.step`). Evaluate held-out prediction error on a large fixed test grid,
per target (`grid_energy`, `soh_loss`, `next_soc`) and pooled.

Sweep `N ∈ {2000, 1000, 500, 200, 100, 50, 25}` (multiple fit seeds each) and
report error vs N for both models. The deliverable is the **crossover N** below
which the PINN's held-out error is materially lower — and *where* in the envelope
it wins (expected: near the SoC bounds and at high power/stress, the sparsely
sampled corners the MLP cannot interpolate but the residual constrains).

**Gate on Stage 2:** if the PINN does not beat the MLP on held-out accuracy at
small N, the downstream SAC experiment cannot benefit and we stop here (write it
up as a negative result, as with the adaptation experiment).

## Stage 2 — SAC sample efficiency (expensive; gated on Stage 1)

Add `EnvConfig.battery_factory: Callable[[BatteryConfig], BatteryLike] | None`,
mirroring the existing `market_factory`: when set, `reset()` builds the battery
from it instead of `Battery(config.battery)`. No other env change.

Arms, each at transition budget `N` from the sweep, **market exact** throughout:

| arm | battery in training env | purpose |
|---|---|---|
| `exact` | real `Battery` | upper bound / sanity (recovers the Dyna baseline) |
| `mlp` | `LearnedBattery(MLP fit on N)` | black-box surrogate control |
| `pinn` | `LearnedBattery(PINN fit on N)` | the thesis mechanism |

Train SAC identically in each (same seeds, timesteps, warmup), then **evaluate on
the real env with the real battery** — paired same-seed, ≥20 episodes, reported
as % of the hindsight optimum on the 14-day ladder (the project's standard
metric; per-episode variance is huge, so always paired). Plot % hindsight vs N
for `mlp` vs `pinn`; the claim is a crossover N where `pinn` tracks `exact` while
`mlp` has already degraded (a mis-specified degradation surrogate makes SAC
over- or under-cycle).

## Integration & tests

- `energy_storage/battery_model.py` (torch, **train extra only** — core package
  must not import it, same rule as `market_model.py`/`viz.py`): dataset sampler
  from `Battery`, `MLPBatteryModel`, `PINNBatteryModel`, `LearnedBattery`.
- `scripts/evaluate_battery_model.py`: the Stage 1 sweep + report/plot.
- `scripts/train_battery_dyna.py`: the Stage 2 arms (Stage 2 only).
- Tests (`tests/test_battery_model.py`): output shapes/finiteness; seeded
  determinism; **physical admissibility of the PINN** (Δsoh ≤ 0, sign
  consistency, SoC stays in bounds) on random inputs; `LearnedBattery`
  round-trips through the env (`market_factory` default path unchanged).

## Risks

- **The physics prior can also be a ceiling.** If the hard constraints are even
  slightly wrong (e.g. the real re-expression on capacity fade), the PINN carries
  a floor of bias the MLP could in principle fit away with enough data — so the
  MLP may *win at large N*. That crossover is exactly the thing being measured,
  not a bug; report both ends.
- **Trivial dynamics.** The battery map is low-dimensional and smooth, so even
  the MLP may need very few samples — the crossover could sit at an
  uninterestingly small N. If so, the honest finding is "physics priors buy
  little when the system is this simple", and the experiment's value is the
  methodology + the accuracy curves, reported as a (mild) negative result.
- **Residual weight λ** trades data-fit against physics-fit; swept as a
  robustness check so the PINN isn't hand-tuned to win.
