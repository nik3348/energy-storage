# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A battery storage arbitrage environment for training a SAC agent. Everything is synthetic and seeded — a simulated day-ahead power market generates prices from weather, demand, and fuel-price drivers; a battery with degradation physics trades against it as a price-taker.

## Commands

Everything runs through uv:

```bash
uv run pytest -q                                   # all tests
uv run pytest tests/test_market.py -q              # one file
uv run pytest tests/test_battery.py::test_soh_floors_at_eol  # one test

# Training/diagnostics need the 'train' extra (torch, sb3, wandb, matplotlib):
uv run --extra train python scripts/train_sac.py            # trains, evals vs baselines
uv run --extra train python scripts/diagnose.py --model models/best_model.zip
```

Both scripts log to the `energy-storage` wandb project by default; pass `--no-wandb` to skip.

## Architecture

Three layers, each usable standalone:

1. **`energy_storage/battery.py`** — battery physics. Conventions: power > 0 charges; SoC is a fraction of *current degraded* capacity (`capacity_kwh * soh`); SoH declines from throughput cycling (scaled up at high power by `cycle_stress`) plus calendar aging (scaled up at high SoC), and floors at `eol_soh`. Stored energy is conserved when capacity shrinks (SoC is re-expressed).

2. **`energy_storage/market/`** — the price simulator, calibrated to a **European (heating-dominated) profile**. `Market.simulate_day()` in `day_ahead.py` orchestrates one day: Calendar/Weather drivers → Demand (heating-dominated, U-shaped over the year: seasonal base load and heating peak midwinter, cooling is modest) and FuelMarket (OU processes) → each Generator submits `(price, quantity)` offers per hour → `clearing.clear_hour()` runs a uniform-price merit-order auction → `DayResult` (24 prices + dispatch by tech). Price formation is causal: shocks hit drivers, never prices directly. The **winter price premium emerges from consumption**: the thermal fleet is deliberately granular (units of varying efficiency make the merit order slope), so seasonal demand moves the marginal cost. Intra-day spreads come largely from the **solar duck curve** (500 MW solar: cheap middays, steep evening ramps). Negative prices emerge from subsidy-style negative renewable bids, coal min-run blocks, and must-run geothermal — mostly sunny summer middays; scarcity hours clear at `price_cap` and concentrate in winter/early spring (high demand + pre-melt reservoir + outage coincidences). Hydro bids its opportunity cost as a function of reservoir fill (self-stabilizing).

3. **`energy_storage/env.py`** — `BatteryArbitrageEnv` (Gymnasium). Hourly steps; action is `Box(-1, 1)` = fraction of max charge/discharge power. The env keeps a two-day `DayResult` buffer so the observation always contains the **next 24 day-ahead prices** (log-normalized) — the agent is supposed to know future prices, that's how day-ahead markets work. Reward = profit − monetized degradation (`soh_loss × replacement_cost × capacity`), scaled by `reward_scale`; there is deliberately no tunable degradation penalty weight. The observation layout is documented at the top of the file.

Scenarios (`market/scenarios.py`: ColdSnap, HeatWave, FuelShock, Drought, PlantOutage) perturb drivers inside a day window and **consume no randomness**, so a scenario run and a same-seed baseline run are identical outside the shock window — they're designed for counterfactual evaluation. In `Market.simulate_day()` they must apply before demand is computed (cold snap → heating load) and before generator `new_day` (drought → inflows).

Determinism: one `np.random.default_rng(seed)` is shared by all market components; reproducibility depends on the call order inside `simulate_day()` staying fixed.

Units: market is MW/MWh and $/MWh; battery is kW/kWh. The env divides by 1000 when settling.

`energy_storage/baselines.py` has the idle/heuristic policies and rollout/eval helpers shared by both scripts — extend there, not in the scripts. `energy_storage/oracle.py` has the DP benchmarks: `RollingHorizonOracle` (optimal policy over the agent's exact 24h information set, replanned hourly) and the hindsight optimum (full-episode plan, valid because the battery is a price-taker). Its `SocGridPlanner.step_economics` must mirror `Battery.step` exactly — a test pins them together.

## Calibration and tests

Market tests are statistical assertions over a seeded simulated year (fixture in `tests/test_market.py`): median price ~$64/MWh, **winter median > summer median + $6 and winter mean > summer mean + $15 (the European gate — the premium is consumption-driven; windy winter nights cap the median gap)**, evening peak > overnight, scarcity < 1.5% of hours, negative prices rare but present. If you touch fleet capacities, weather parameters, bid levels, or the seasonal amplitudes, these tests are the calibration gate — a failure usually means the market drifted, not that the test is wrong. The generator fleet is sized so scarcity comes from outage+calm+peak coincidences, not routine winter evenings.

Benchmark context for evaluating agents (net $/14-day episode, 20 paired seeds at seed0=1,000,000, European market): hindsight optimum ~+$90 (=100%), rolling-horizon oracle ~+$90 (≈100% — the 24h window is near-sufficient information), heuristic ~+$45 (50%), idle ~−$13 (calendar aging is unavoidable). SAC models trained before the European market redesign (July 2026) are stale — retrain before quoting numbers. Report policies as % of the hindsight optimum, not raw dollars: per-episode variance across market-weeks is huge (scarcity weeks dominate), so always compare paired same-seed over ≥20 episodes.
