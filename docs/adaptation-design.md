# Rapid adaptation to persistent regime shifts

Status: implemented (`energy_storage/adaptation.py`,
`scripts/run_adaptation.py`, `scripts/plot_adaptation.py`); results pending.
Sequel to the Dyna budget experiment (docs/dyna-design.md); reuses its
arms' checkpoints and the market-model machinery.

## Framing

The budget experiment showed the world model substitutes for real data the
agent never had. This experiment applies the same structural insight along
the **time axis**: a well-trained controller is operating; at day T the
market shifts **persistently** (not a transient shock); how fast can each
method recover, and at what cumulative cost?

Zero-shot robustness (measured: dyna-365 capture 64–89% under transient
shocks, margin over replay roughly doubling) is the *floor*. Adaptation is
the mechanism that closes the remaining gap — and it is the thing that
"just train longer on calm data" provably cannot buy (the unlimited-data
baseline scores *below* dyna-365 under fuel-shock: 56.4% vs 64.2%).

## Why the decomposition makes adaptation cheap

1. **Adaptation data is free and policy-independent.** Prices are exogenous
   and observed whether or not the agent acts: k days after the shift the
   operator holds k shifted days — no exploration cost, no risk to the asset.
2. **Only the small thing shifted.** Battery physics and calendar are
   unchanged; only the market model (a small GRU, minutes to fine-tune on
   day-level data) needs updating. The policy then fine-tunes in
   *imagination* overnight, not on the asset.
3. **Detection is already implemented.** `MarketModelEnsemble.disagreement()`
   (unused so far) should spike on out-of-distribution days; per-day NLL of
   the observed day under the ensemble is a second, sharper signal.

The model-free alternative must fine-tune on real transitions arriving at
24/day — the same asymmetry as the budget experiment's C vs B, now measured
as **recovery speed**.

## Shift catalogue

Persistent shifts, expressible today as long-duration scenarios (set
`duration_days` to cover the rest of the run):

| shift | mechanism | character |
|---|---|---|
| fuel step | `FuelShock`, magnitude ~2x, indefinite | pure price-*level* shift; obs carries most of it — frozen policies should cope best here |
| capacity loss | `PlantOutage` of a large baseload unit, indefinite | structural: scarcity frequency and timing change |
| cold regime | `ColdSnap`, indefinite | demand-level + winter-pattern shift |

Structural shifts are where frozen and adaptive policies should separate
most: they change *when* spikes happen, not just how high prices sit, and
the observation window alone doesn't re-teach hold/release timing.

## Protocol

One evaluation "deployment" = a continuous market run, same seed across
arms (paired):

- days 0–30: pre-shift (calm), establishes each arm's baseline capture;
- day 30 = T: shift begins, persists to day 120;
- per-day net profit recorded for every arm **and for the rolling-horizon
  oracle on the same market** (the oracle re-plans on observed prices, so
  it adapts instantly by construction — the fair "instant adaptation"
  normalizer).

Score: **capture(t) = 7-day-windowed policy net / oracle net** (single-day
capture is too noisy; windows smooth without hiding the recovery edge).
≥10 paired seeds; report the mean capture curve with seed band.

### Arms

| arm | behavior after T |
|---|---|
| frozen-dyna | dyna-365 checkpoint, never updated (zero-shot floor) |
| frozen-ideal | sac-year-v3 best, never updated (the "fully trained" strawman) |
| adaptive-dyna | nightly: fine-tune ensemble on trailing window of observed days; fine-tune SAC in imagination (~10k steps); deploy tomorrow |
| online-ft | sac-year-v3 best, fine-tuned on the real stream (24 transitions/day) — the model-free adaptation baseline |
| heuristic | fixed rule (context) |

### Metrics

- **Recovery lag**: days until capture regains 90% of the arm's own
  pre-shift level.
- **Cumulative regret**: sum over post-shift days of (oracle − policy) net.
- **Detection trace**: ensemble disagreement / per-day NLL vs time — did
  the model notice the shift at T? (Reported even though phase-1
  adaptation is nightly-always; a disagreement-*triggered* variant is the
  phase-2 refinement.)

## Adaptive-dyna loop (nightly)

1. Append today's observed (shifted) day to the history.
2. Fine-tune each ensemble member on a trailing window (e.g. last 90 days,
   bootstrap preserved) for a few epochs from current weights — do NOT
   refit from scratch: pre-shift structure (diurnal shape, calendar) still
   holds and must not be forgotten.
3. Regenerate imagination: LearnedMarket now samples shifted days.
4. Fine-tune SAC for M imagined steps (M ~ 10k) from current weights,
   mixing planner transitions on the *recent real* days into replay.
5. Deploy the updated policy for tomorrow. All of this is off-asset.

Guardrails: keep a fraction of pre-shift days in the fine-tune window
(anti-forgetting); track the gate stats of the refit model on a rolling
basis — if the refit model degrades the *calm* statistics badly, the
imagination is drifting and the run is flagged.

## Hypotheses

1. adaptive-dyna recovers in days; online-ft in weeks-to-never within the
   90-day horizon (24 real transitions/day is the binding rate).
2. frozen arms never fully recover on *structural* shifts, but roughly
   cope with the pure level shift (the price window carries level).
3. The disagreement signal spikes within a few days of T (usable trigger).
4. adaptive-dyna's advantage is largest on capacity-loss (structural),
   smallest on fuel step — mirroring which shifts the observation itself
   can already convey.

## Infrastructure (implemented)

- **Streaming evaluation harness** (`energy_storage/adaptation.py`):
  `run_stream(arm, config, seed)` drives one long episode (episode_days =
  horizon) day-by-day; after each settled day the arm's `end_of_day(record)`
  hook fires (where adaptive arms retrain) and may return per-day logs
  (NLL, disagreement, adapted flag) that land in the CSV. Arms sharing a
  reset seed see identical markets — pairing is automatic because prices
  are exogenous.
- `MarketModelEnsemble.fine_tune(x, y, epochs, lr)` — continues every
  member from current weights (fresh bootstrap per member); and
  `MarketModelEnsemble.day_nll(x, y)` — exact chain-rule mixture NLL of an
  observed day, the detection signal.
- `AdaptiveDynaArm` implements the nightly loop below; the imagination env
  holds the ensemble by reference, so every generated day uses the latest
  fine-tuned weights without rebuilding envs. SAC warm-starts via
  `model.learn(reset_num_timesteps=False)` on a fresh small replay buffer
  (100k — stale imagination ages out), reseeded nightly with planner demos
  on the trailing real days. An optional `nll_trigger` turns the phase-2
  triggered variant on; default adapts every night.
- `OnlineFinetuneArm`: loaded SAC refills a fresh buffer at 24 real
  transitions/day, N gradient steps nightly (training starts once a full
  batch exists, ~day 11, still pre-shift).
- `scripts/run_adaptation.py`: `--shift {fuel-step,capacity-loss,cold-regime}
  --arm {oracle,frozen-dyna,frozen-ideal,adaptive-dyna,online-ft,heuristic}
  --seeds N`, one CSV per seed under results/adaptation/{shift}/;
  `scripts/plot_adaptation.py --shift ...` renders the capture curves +
  detection trace and prints recovery lag / cumulative regret. The
  capacity-loss shift outages `coal-1` (the cheap 100 MW baseload with a
  min-run block) so the merit order reshapes, not just the level.

## Risks

- **Compute**: adaptive-dyna = 90 nightly fine-tunes × seeds × shifts.
  Budget by shrinking M and ensemble fine-tune epochs; the loop is fully
  simulated, so wall-clock is the only constraint.
- **Instability of nightly RL fine-tuning** (catastrophic forgetting,
  policy churn): mitigations — small M, frozen critic option, keep calm
  imagination mixed in. If SAC fine-tuning proves too twitchy, the
  fallback arm is "refit model + retrain from the budget-experiment
  checkpoint's replay buffer", slower but stable.
- **Oracle normalization under shift**: the rolling oracle is near-optimal
  only because prices are fully observed; that is exactly why it is the
  right instant-adaptation ceiling here, but state this explicitly in the
  write-up (in a forecast-based market the ceiling itself would lag).
