# A Calibration-Gated Market Model for Sample-Efficient Battery Storage Arbitrage

A synthetic day-ahead power market generates
hourly prices from weather, demand and fuel drivers, and a battery with
degradation physics trades against it as a price-taker.

`scripts/pipeline/` collects data and trains checkpoints; `scripts/results/`
has one script per experiment below. Every result is regenerated from
scratch; nothing is bundled.

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra train        # numpy, gymnasium, torch, stable-baselines3, matplotlib, wandb
uv run pytest -q             # 87 tests, ~25s
```

The core simulator needs only numpy and gymnasium; everything that trains or
plots needs `--extra train`. Scripts that can log to Weights & Biases do so by
default; add `--no-wandb` to run offline.

## Results

The paper asks five questions. Each subsection below is one of them, with the
commands that produce its numbers. First, the shared setup every question but
the last one needs: collect three budgets of real market days, fit and gate a
market model at each budget, then train the reference policy and the
budget-sweep arms.

```bash
# 1. Collect real-data budgets
for d in 90 365 1095; do
  uv run --extra train python scripts/pipeline/collect_history.py --days $d --seed 0
done

# 2. Fit and gate a market model at each budget (fails at 90 and 1095 by design)
for d in 90 365 1095; do
  uv run --extra train python scripts/results/validate_market_model.py \
      --budget-days $d --history data/history-d$d-s0.npz
done

# 3. Train the unlimited-data reference policy
uv run --extra train python scripts/pipeline/train_reference_policy.py --models-dir models/sac-year-v3

# 4. Train the budget-sweep arms (dyna only trains where the gate passed, i.e. D=365)
for d in 90 365 1095; do
  uv run --extra train python scripts/pipeline/train_dyna_arms.py --arm online --budget-days $d
  uv run --extra train python scripts/pipeline/train_dyna_arms.py --arm replay --budget-days $d
done
uv run --extra train python scripts/pipeline/train_dyna_arms.py --arm dyna --budget-days 365
```

### 1. Does a learned market substitute for real data under a fixed budget?

```bash
uv run --extra train python scripts/results/measure_budget_sweep.py
```

That's the headline in-text comparison. The gate table itself was already
printed by step 2 above. Two more pieces back it up: whether the same year of
data reliably yields a usable model, and whether the result holds up over
several training runs.

```bash
# Does the gate pass reliably across independent draws of the year?
for s in 1 2 3 4; do
  uv run --extra train python scripts/pipeline/collect_history.py --days 365 --seed $s
  uv run --extra train python scripts/results/validate_market_model.py \
      --budget-days 365 --seed $s --history data/history-d365-s$s.npz --no-wandb
done

# Does dyna beat replay across training seeds, holding the data fixed?
for s in 1 2 3 4; do
  uv run --extra train python scripts/pipeline/train_dyna_arms.py --arm dyna --seed $s \
      --history data/history-d365-s0.npz --model models/market-model-d365-s0.pt \
      --models-dir models/dyna/dyna-d365-fixedhist-s$s
  uv run --extra train python scripts/pipeline/train_dyna_arms.py --arm replay --seed $s \
      --history data/history-d365-s0.npz --model models/market-model-d365-s0.pt \
      --models-dir models/dyna/replay-d365-fixedhist-s$s
done
uv run --extra train python scripts/results/measure_fixed_history_sweep.py --checkpoint sac_final.zip
```

### 2. Does training in imagination avoid infeasible actions after deployment?

```bash
uv run --extra train python scripts/results/measure_safe_exploration.py
```

Prints both halves of the result: infeasible requests during training (top),
and the same trained checkpoints as deployed (bottom).

### 3. Does the gated model degrade gracefully under transient shocks?

```bash
uv run --extra train python scripts/results/measure_scenario_robustness.py
```

### 4. Can the adaptation loop recover from persistent regime shifts?

This one needs its own data: 3 shifts x 6 arms x 10 paired 120-day
deployments. It's the most expensive command in the repo and checkpoints
nightly, so a killed run resumes rather than restarting.

```bash
for shift in fuel-step capacity-loss cold-regime; do
  for arm in oracle heuristic frozen-dyna frozen-ideal online-ft adaptive-dyna; do
    uv run --extra train python scripts/pipeline/run_adaptation_deployments.py \
        --shift $shift --arm $arm --seeds 10
  done
done
```

Then the results and the figure:

```bash
uv run --extra train python scripts/results/plot_adaptation.py --shift fuel-step      # + writes Figure 1
uv run --extra train python scripts/results/plot_adaptation.py --shift capacity-loss
uv run --extra train python scripts/results/plot_adaptation.py --shift cold-regime
uv run --extra train python scripts/results/adaptation_stats.py
```

And the mechanism behind the negative result, why the nightly refit fails:

```bash
uv run --extra train python scripts/results/measure_ensemble_drift.py
```

### 5. Does a physics-informed battery surrogate need fewer transitions?

Self-contained; no shared setup needed.

```bash
uv run --extra train python scripts/results/evaluate_battery_model.py --seeds 1 2 3 4 5 6 7 8 9 10
uv run --extra train python scripts/results/evaluate_battery_model.py --nonlinear --seeds 1 2 3 4 5 6 7 8 9 10
uv run --extra train python scripts/results/evaluate_wrong_constraints.py
```

The first is the main comparison. `--nonlinear` reruns it against a battery
with convex cycle-stress and an end-of-life knee, asking whether the physics
was simply too easy to need a prior; both knobs default off everywhere else,
so no other result in the repo is affected. The wrong-constraints run asks
whether the PINN's edge is real physics or a generic regularizer, by refitting
it with deliberately broken physics.
