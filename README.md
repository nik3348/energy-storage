# energy-storage

A battery storage arbitrage environment for reinforcement learning. Everything is
synthetic and seeded: a simulated day-ahead power market generates hourly prices
from weather, demand, and fuel-price drivers, and a battery with degradation
physics trades against it as a price-taker.

## Layout

Three layers, each usable standalone:

- **`energy_storage/battery.py`** — battery physics: SoC tracking, round-trip
  losses, and SoH degradation from cycling (worse at high power) and calendar
  aging (worse at high SoC).
- **`energy_storage/market/`** — the price simulator. `Market.simulate_day()`
  runs one day: calendar/weather drivers → temperature-sensitive demand and
  mean-reverting fuel prices → generators submit `(price, quantity)` offers →
  a uniform-price merit-order auction clears each hour. Negative prices emerge
  from subsidised renewable bids and coal min-run blocks; scarcity hours clear
  at the price cap. Scenarios (`ColdSnap`, `HeatWave`, `FuelShock`, `Drought`,
  `PlantOutage`) layer deterministic shocks on top for counterfactual
  evaluation.
- **`energy_storage/env.py`** — `BatteryArbitrageEnv`, a Gymnasium env. Hourly
  steps; the action is a `Box(-1, 1)` fraction of max charge/discharge power;
  the observation includes the next 24 day-ahead prices (that's how day-ahead
  markets work). Reward is profit minus monetized SoH degradation.

`energy_storage/baselines.py` has the idle/heuristic reference policies and
the rollout/eval helpers shared by the scripts. `energy_storage/oracle.py`
has the optimal-dispatch benchmarks (dynamic programming on a SoC grid): a
rolling-horizon oracle using exactly the agent's information set, and a
hindsight upper bound over the full episode.

## Usage

Everything runs through [uv](https://docs.astral.sh/uv/):

```bash
uv run pytest -q                                   # tests

# Training and diagnostics need the 'train' extra (torch, sb3, wandb, matplotlib):
uv run --extra train python scripts/train_sac.py            # train SAC, eval vs baselines
uv run --extra train python scripts/diagnose.py --model models/best_model.zip
```

Both scripts log to the `energy-storage` wandb project by default; pass
`--no-wandb` to skip.

## Benchmarks

Net $/episode over 20 paired held-out seeds (14-day episodes):

| policy | net | % of optimum |
|---|---|---|
| hindsight optimum | +$62 | 100% |
| rolling-horizon oracle (24h) | +$62 | 99.8% |
| SAC | +$52 | 83% |
| overnight/evening heuristic | +$16 | 25% |
| idle | −$13 | — |

Calendar aging makes idling a losing strategy, and the 24h day-ahead window
turns out to be near-sufficient information (the rolling oracle ≈ hindsight).
Per-episode variance across market-weeks is huge — always compare policies as
paired same-seed differences over at least 20 episodes, ideally as % of the
hindsight optimum.

Market calibration (seeded simulated year): median price ~$58/MWh, evening
peak above overnight, scarcity in <1.5% of hours, negative prices rare but
present. The statistical tests in `tests/test_market.py` are the calibration
gate for any change to fleet capacities, weather parameters, or bid levels.
