# Dyna pipeline design

Status: agreed design, pre-implementation. Companion experiment to the model-free
SAC baseline (`models/sac-year-v2`).

## Framing

The simulator stands in for the real world. Training an RL agent against a real
battery + market is expensive and has consequences, so the experiment asks: given
a **limited budget of real-world data**, can a Dyna-style agent — one that learns
a generative model of the market and trains on synthetic experience — match or
beat a model-free agent trained on far more real data, and degrade more
gracefully under unseen freak scenarios?

Scenarios (cold snap, heat wave, fuel shock, drought, outage) stay **unseen
during training** in every arm. Robustness is evaluated zero-shot with the paired
harness in `energy_storage/robustness.py`.

## The structural insight the design is built on

The environment decomposes cleanly because the agent is a **price-taker**:

1. **Battery dynamics are known.** An operator knows their own asset; the planner
   already mirrors `Battery.step` exactly. Nothing to learn.
2. **Calendar is known.** Day-of-year, weekends, holidays are free.
3. **Market and weather are exogenous** — the agent's actions never influence
   prices or weather.

Consequences:

- The only thing to learn is a **generative model of market days**
  (prices + the weather features the observation exposes).
- **Model error cannot compound through actions** (the MBPO failure mode).
  Imagined rollouts are exact battery physics driven by a synthetic exogenous
  sequence; error lives only in that sequence.
- **The data budget is policy-independent.** N hours of interaction reveal
  N/24 market days no matter which policy ran. So the budget is denominated in
  **days of observed market history (D)**, and the "safe behavior policy"
  question (you can't let an untrained agent loose on a real asset) decouples
  from model quality: demos from the rolling-horizon planner — the
  industry-standard controller, real-world-legal — provide the real transitions
  for the replay buffer without costing model fidelity anything.

## Experiment arms

All arms see exactly D days of real market history and are evaluated identically
(paired seeds on the real simulator: % of hindsight optimum on the 14-day ladder,
plus the zero-shot scenario table; ≥20 episodes).

| arm | training data | purpose |
|---|---|---|
| A `online` | standard SAC, 24·D real env steps, then stop | classic sample-efficiency baseline |
| B `replay` | SAC on the D stored days replayed verbatim (unlimited epochs, random episode offsets) | the honest control: "just refit on your history" — separates *more gradient steps* from *model generalization* |
| C `dyna` | ensemble generative model fit on the D days; SAC trains on unlimited synthetic days; real planner transitions seeded into replay | the thesis mechanism |

Hypotheses: C ≥ B > A at small D; C's margin over B is largest on the unseen
scenario regimes and in worst-decile episodes (model-space randomization by the
ensemble is the robustness mechanism).

Budget sweep: **D ∈ {90, 365, 1095}** with D = 365 as the headline. Training
episodes are 14 days (windows over history / synthetic days) so every D covers
all its seasons across episodes; the year-long-episode SAC baseline remains the
unlimited-data reference point.

## The market model

**Unit of generation: one day** (24 prices + 24×3 weather), matching the env's
two-day `DayResult` buffer and the day-ahead information structure.

- **Target space**: prices in the env's log-normalized space
  (`sign(p)·log1p(|p|)/log1p(cap)`), temperature `/TEMP_SCALE_C`, wind
  `/WIND_SCALE_MS`, solar_cf raw. Everything O(1); spikes compressed.
- **Conditioning**: `[sin/cos doy, is_weekend, is_holiday,` compressed previous-day
  stats `(mean/min/max price, mean temp, mean wind, mean solar)]`. Prev-day
  dependence is deliberately low-dimensional to limit autoregressive drift.
- **Architecture v1**: MLP with diagonal-Gaussian heads over the 96 output dims
  (MBPO-standard probabilistic net). Known limitation: diagonal residuals
  under-generate *correlated* spike blocks (scarcity evenings). Upgrade path if
  the validation gate fails: small CVAE (≈8-dim latent) or hour-autoregressive
  head.
- **Ensemble**: K = 5, bootstrap-resampled days + different init seeds. Each
  imagined episode draws one member; disagreement is retained as a future OOD
  signal.
- Torch only under the `train` extra (module must not be imported by the core
  package, same rule as `viz.py`).

**Validation gate (before any RL on it)**: generate synthetic years and compare
against the real market's calibration statistics — median price, winter/summer
median and mean gaps, evening>overnight, daily spread distribution, spike
frequency, negative-hour share. The market tests in `tests/test_market.py`
define the metric list; the model report reuses them as soft metrics. A model
that fails the spread/spike stats will silently gut arbitrage value in
imagination — this gate is what makes the Dyna result interpretable.

## Integration

- `energy_storage/market_model.py`: dataset builder (days → tensors),
  `MarketDayModel`, `MarketModelEnsemble` (fit / sample_day / disagreement).
- `LearnedMarket`: duck-types `Market` (`simulate_day() -> DayResult`, settable
  `day`). Returns real calendar flags, synthetic prices + `WeatherDay`; dispatch
  and fuel fields left empty/dummy — nothing downstream of the env reads them.
- `ReplayedMarket` (arm B): serves stored days verbatim, wrapping at the end.
- `EnvConfig.market_factory: Callable[[MarketConfig, int], MarketLike] | None`
  — `reset()` calls it instead of constructing `Market` when set. No other env
  change; observation layout untouched, so policies transfer between real and
  learned envs by construction.
- `scripts/collect_history.py`: simulate D real days (+ planner transitions for
  replay seeding), save npz.
- `scripts/train_dyna.py`: `--arm {online,replay,dyna} --budget-days D`; fits
  the ensemble (arm C), builds the vec env, trains SAC, runs the standard eval.

## Tests

- Model: fit on a tiny dataset, output shapes/finiteness, seeded determinism.
- `LearnedMarket`/`ReplayedMarket` produce `DayResult`s the env accepts;
  full env round-trip; replay serves the exact stored prices.
- `market_factory` default path unchanged (existing env tests already pin it).

## Risks

- **Spike realism** is the main model risk (see gate above). Most arbitrage value
  is scarcity capture; a model that misses spikes trains a timid agent, one that
  over-generates them trains a reckless one. The gate quantifies this before RL.
- **Autoregressive drift** over long synthetic horizons — mitigated by
  calendar-dominant conditioning and 14-day training episodes.
- **Fuel prices are unobserved** by design (removed from the observation); the
  model learns price *levels* implicitly through the conditioning + residual
  variance. Fuel-shock robustness therefore tests level-shift generalization —
  exactly the point.
