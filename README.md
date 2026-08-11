# Energy Storage Arbitrage

Supporting material for the dissertation. Everything in the paper is produced
by a script in `scripts/`, and this file says which script produces which
table or figure.

The project is entirely synthetic and seeded: a simulated day-ahead power
market generates hourly prices from weather, demand and fuel drivers, and a
battery with degradation physics trades against it as a price-taker. No
external data is downloaded and no API keys are needed.

## 1. Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/). All commands are
run from the repository root.

```bash
uv sync --extra train        # installs numpy, gymnasium, torch, stable-baselines3, matplotlib, wandb
uv run pytest -q             # 84 tests, ~25s, confirms the install works
```

`uv run` creates the environment on first use, so `uv sync` is optional. The
core simulator needs only numpy and gymnasium; everything that trains or plots
needs the `train` extra, which is why every command below carries
`--extra train`.

Scripts that can log to Weights & Biases (`train_sac.py`, `train_dyna.py`,
`validate_market_model.py`, `run_adaptation.py`) do so by default. Pass
`--no-wandb` to run fully offline.

## 2. Contents

| Path | Contents |
|---|---|
| `energy_storage/` | The library: market simulator, battery, Gymnasium env, oracles, world model, adaptation loop |
| `energy_storage/market/` | Merit-order market: demand, weather, fuels, generators, clearing, regime-shift scenarios |
| `scripts/` | One script per experiment, listed in Section 4 |
| `tests/` | 84 tests, including the statistical calibration gate for the market |
| `data/` | Cached market histories and PINN result archives |
| `results/` | Per-day adaptation CSVs and the budget-sweep JSON that the figures read |

## 3. Benchmarks

The benchmark ladder is what every percentage in the paper is measured
against. It needs no training and is the quickest way to confirm a working
install:

```bash
uv run --extra train python scripts/measure_ladder.py
```

## 4. Results

### Prerequisites (run once)

```bash
# The real-data budget every arm is allowed to see
uv run --extra train python scripts/collect_history.py --days 365 --seed 0
uv run --extra train python scripts/collect_history.py --days 90   --seed 0
uv run --extra train python scripts/collect_history.py --days 1095 --seed 0

# The unlimited-data model-free reference policy
uv run --extra train python scripts/train_sac.py --models-dir models/sac-year-v3
```

### Table I: validation-gate sweep (Section VII-A)

```bash
uv run --extra train python scripts/validate_market_model.py --budget-days 90   --history data/history-d90-s0.npz
uv run --extra train python scripts/validate_market_model.py --budget-days 365  --history data/history-d365-s0.npz
uv run --extra train python scripts/validate_market_model.py --budget-days 1095 --history data/history-d1095-s0.npz
```

Prints the gate table and exits 0 on pass, 1 on fail. The fitted ensemble is
saved to `models/market-model-d{D}-s0.pt` **only if the gate passes**, which
is how the gate is enforced: `train_dyna.py --arm dyna` refuses to run without
that file. Only `D=365` passes, which is why the dyna arm exists at one budget.

### Figure 1: sample efficiency versus real-data budget (Section VII-A)

Train the arms (long; hours per arm on CPU):

```bash
for d in 90 365 1095; do
  uv run --extra train python scripts/train_dyna.py --arm online --budget-days $d
  uv run --extra train python scripts/train_dyna.py --arm replay --budget-days $d
done
uv run --extra train python scripts/train_dyna.py --arm dyna --budget-days 365
```

Then score every checkpoint on the same 20 paired held-out episodes and plot:

```bash
uv run --extra train python scripts/measure_budget_sweep.py   # writes results/budget_sweep.json
uv run --extra train python scripts/plot_dyna_results.py      # writes diagnostics/dyna_sample_efficiency.png
```

The figure reads the JSON rather than hardcoding numbers, so it cannot drift
away from the tables.

### Table II: infeasible on-asset action requests (Section VII-B)

```bash
uv run --extra train python scripts/measure_safe_exploration.py    # top block: training-time behaviours
uv run --extra train python scripts/measure_deployed_clipping.py   # bottom block: trained policies as deployed
```

### Table III: zero-shot robustness to transient shocks (Section VII-C)

```bash
uv run --extra train python scripts/measure_scenario_robustness.py
```

### Tables IV and V, Figures 2 to 4: adaptation to persistent shifts (Section VII-D)

Run the deployments. This is the most expensive experiment in the paper:
3 shifts x 6 arms x 10 paired 120-day deployments.

```bash
for shift in fuel-step capacity-loss cold-regime; do
  for arm in oracle heuristic frozen-dyna frozen-ideal online-ft adaptive-dyna; do
    uv run --extra train python scripts/run_adaptation.py --shift $shift --arm $arm --seeds 10
  done
done
```

Each run writes one CSV per seed to `results/adaptation/{shift}/{arm}-s{seed}.csv`
and checkpoints nightly, so a killed run resumes rather than restarting. The
`oracle` arm is the normaliser every capture curve is divided by, so it must be
run for the same seeds as every other arm.

Then the tables and figures:

```bash
uv run --extra train python scripts/adaptation_stats.py            # Tables IV and V
uv run --extra train python scripts/plot_adaptation.py --shift fuel-step      # Figure 2
uv run --extra train python scripts/plot_adaptation.py --shift capacity-loss  # Figure 3
uv run --extra train python scripts/plot_adaptation.py --shift cold-regime    # Figure 4
```

The pre-computed CSVs are included in `results/adaptation/`, so the three
commands above reproduce the tables and figures without rerunning the
deployments.

### Ensemble drift trace (Section VII-D, "Where the shortfall comes from")

Replays the adaptive arm's nightly refit offline and re-gates the ensemble
after every night, which is the measurement showing the refit leaves the gate
on night three and never returns:

```bash
uv run --extra train python scripts/measure_ensemble_drift.py
```

### Table VI: PINN versus MLP battery surrogate (Section VII-E)

```bash
uv run --extra train python scripts/evaluate_battery_model.py --seeds 1 2 3 4 5 6 7 8 9 10
```

Writes `data/battery-model-stage1.npz` and prints the accuracy and
physical-violation tables for both coverage regimes.

### Table VII: wrong-constraints ablation (Section VII-E1)

```bash
uv run --extra train python scripts/evaluate_wrong_constraints.py
```

Uses ten fit seeds and the same training draws and test set as Table VI.
Writes `data/battery-model-wrong-constraints.npz`.
