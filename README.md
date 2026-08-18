# Energy Storage Arbitrage

Supporting material for the dissertation. Almost every table and figure in the
paper is produced by a script in `scripts/`, and this file says which script
produces which. The one exception is noted in Section 4.

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
| `scripts/` | One script per experiment, mapped to tables and figures in Section 4 |
| `tests/` | 84 tests, including the statistical calibration gate for the market |
| `data/` | Pinned market histories and the battery-surrogate result archives |
| `results/` | Per-day adaptation CSVs, and the sweep JSONs the tables and in-text numbers are read from |
| `docs/` | `main.tex` and `main.pdf`, the reflective essay, and the one paper figure |

## 3. Benchmarks

The benchmark ladder is what every percentage in the paper is measured
against. It needs no training and is the quickest way to confirm a working
install:

```bash
uv run --extra train python scripts/measure_ladder.py
```

## 4. Results

The paper has twelve tables and one figure. Tables I and II are in the body;
Tables III to IX and Figure 1 are collected in Appendix A, and Tables X to XII
in Appendix B. The headings below give the section each result is *discussed*
in, which is where the claim it supports is made.

Table III is a list of constants, and Table XI is covered at the end of this
section. Everything else is produced by the commands here.

### Prerequisites (run once)

```bash
# The real-data budget every arm is allowed to see
uv run --extra train python scripts/collect_history.py --days 365 --seed 0
uv run --extra train python scripts/collect_history.py --days 90   --seed 0
uv run --extra train python scripts/collect_history.py --days 1095 --seed 0

# The unlimited-data model-free reference policy
uv run --extra train python scripts/train_sac.py --models-dir models/sac-year-v3
```

### Table IV: the calibration contract (Section III-C)

Asserted directly by the test suite over a seeded simulated year, so it is
verified by `uv run pytest -q` rather than by a script:

```bash
uv run pytest tests/test_market.py -q
```

### Table V: calibration gate across real-data budgets (Section VII-A)

```bash
uv run --extra train python scripts/validate_market_model.py --budget-days 90   --history data/history-d90-s0.npz
uv run --extra train python scripts/validate_market_model.py --budget-days 365  --history data/history-d365-s0.npz
uv run --extra train python scripts/validate_market_model.py --budget-days 1095 --history data/history-d1095-s0.npz
```

Prints the gate table and exits 0 on pass, 1 on fail. The fitted ensemble is
saved to `models/market-model-d{D}-s0.pt` **only if the gate passes**, which
is how the gate is enforced: `train_dyna.py --arm dyna` refuses to run without
that file. Only `D=365` passes, which is why the dyna arm exists at one budget.

### Table VI: gate outcome across independent draws of the year (Section VII-A)

Whether a year is usable depends on which year. This varies the seed over the
whole pipeline, so each seed collects its own 365 days, fits its own ensemble
and is re-gated before any policy trains. It is the source of the one-in-five
pass rate.

The table itself is the gate verdict per draw, so it comes from running the
gate once per seed:

```bash
for s in 0 1 2 3 4; do
  uv run --extra train python scripts/collect_history.py --days 365 --seed $s
  uv run --extra train python scripts/validate_market_model.py \
      --budget-days 365 --seed $s --history data/history-d365-s$s.npz \
      --out diagnostics/gate-s$s --no-wandb
done
```

Each call prints the per-criterion table, including the winter mean price gap
that decides the gate, and exits non-zero on a failing draw.

`scripts/run_seed_sweep.sh 1 2 3 4` wraps that loop and additionally trains the
dyna and replay arms per seed, skipping the dyna arm where the gate failed,
which is the deployed protocol. Logs land in `logs/seeds/`, and it is safe to
re-run because completed stages are skipped. `scripts/run_gate_sweep.sh 5 6 7 8 9`
is the gate-only companion, fitting and scoring an ensemble per seed without
training any policy, which is the cheap way to extend the pass-rate estimate to
further draws.

`results/seed_sweep_summary.json` and `results/seed_sweep_d365.json` are the
collected outputs of that sweep, kept for reference. They were assembled from
the run logs rather than written by a script, so re-running the sweep will not
regenerate them.

### Table I: fixed-history training-seed sweep (Section VII-A)

The headline dyna-versus-replay result. This holds the real-data draw fixed at
seed 0's history and seed 0's gated ensemble, so the seed indexes the training
run only and both arms train at every seed.

```bash
scripts/run_fixed_history_sweep.sh 1 2 3 4    # seed 0 was trained on the same pinned artifacts

uv run --extra train python scripts/measure_fixed_history_sweep.py \
    --checkpoint sac_final.zip --out results/fixed_history_sweep_final.json
```

Table I reports the `sac_final.zip` checkpoint, which is why the command names
it explicitly; the script defaults to `best_model.zip`. Both are committed, as
`results/fixed_history_sweep_final.json` and `results/fixed_history_sweep_best.json`,
and the conclusion in Section VIII quotes the seed-0 `best_model.zip` figure.

### In-text budget comparison and the unlimited-data reference (Section VII-A)

Scores every checkpoint on the same 20 paired held-out episodes and writes the
full ladder, including the unlimited-data reference policy the dyna arm is
compared against:

```bash
uv run --extra train python scripts/measure_budget_sweep.py   # writes results/budget_sweep.json
```

Training the arms it scores is long, hours per arm on CPU:

```bash
for d in 90 365 1095; do
  uv run --extra train python scripts/train_dyna.py --arm online --budget-days $d
  uv run --extra train python scripts/train_dyna.py --arm replay --budget-days $d
done
uv run --extra train python scripts/train_dyna.py --arm dyna --budget-days 365
```

### Table VII: infeasible on-asset action requests (Section VII-B)

```bash
uv run --extra train python scripts/measure_safe_exploration.py    # top block: training-time behaviours
uv run --extra train python scripts/measure_deployed_clipping.py   # bottom block: trained policies as deployed
```

### Table II: zero-shot robustness to transient shocks (Section VII-C)

```bash
uv run --extra train python scripts/measure_scenario_robustness.py
```

### Tables VIII and IX, Figure 1: adaptation to persistent shifts (Section VII-D)

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

Then the tables and the figure:

```bash
uv run --extra train python scripts/adaptation_stats.py                       # Tables VIII and IX
uv run --extra train python scripts/plot_adaptation.py --shift fuel-step      # Figure 1
```

The pre-computed CSVs are included in `results/adaptation/`, so the two
commands above reproduce the tables and the figure without rerunning the
deployments. `plot_adaptation.py` also accepts `--shift capacity-loss` and
`--shift cold-regime`; the paper reports fuel-step as the representative case
and the other two shifts in Table VIII.

### Ensemble drift trace (Section VII-D, "Where the shortfall comes from")

Replays the adaptive arm's nightly refit offline and re-gates the ensemble
after every night, which is the measurement showing the refit leaves the gate
on night three and never returns:

```bash
uv run --extra train python scripts/measure_ensemble_drift.py
```

### Table X: PINN versus MLP battery surrogate (Section VII-E)

```bash
uv run --extra train python scripts/evaluate_battery_model.py --seeds 1 2 3 4 5 6 7 8 9 10
```

Writes `data/battery-model-stage1.npz` and prints the accuracy and
physical-violation tables for both coverage regimes.

### Table XI: nonlinear ground truth (Section VII-E, "Harder physics")

Table XI repeats the Table X comparison against a battery whose degradation is
nonlinear, with quadratic cycle stress and an end-of-life knee. That variant of
the ground truth is not part of the simulator in this repository, so Table XI
is the one result here with no reproducing command.

### Table XII: wrong-constraints ablation (Section VII-E, "Wrong-constraints ablation")

```bash
uv run --extra train python scripts/evaluate_wrong_constraints.py
```

Uses ten fit seeds and the same training draws and test set as Table X.
Writes `data/battery-model-wrong-constraints.npz`.

## 5. Diagnostics not used in the paper

`scripts/plot_dyna_results.py` reads `results/budget_sweep.json` and plots
capture against real-data budget as `diagnostics/dyna_sample_efficiency.png`.
The paper reports that comparison in the text of Section VII-A rather than as a
figure, so the plot is kept only as a diagnostic.
