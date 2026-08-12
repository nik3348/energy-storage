#!/usr/bin/env bash
# Repeat the D=365 sample-efficiency comparison over independent training seeds.
#
# The seed is varied over the whole pipeline, not just SAC's initialisation:
# each seed draws its own 365 real days, fits its own market ensemble, and is
# re-gated before any policy trains. A seed whose ensemble fails the gate gets
# no dyna arm, which is the deployed protocol and is itself a reported number
# (the gate pass rate across seeds).
#
#   scripts/run_seed_sweep.sh 1 2 3 4
#
# Logs land in logs/seeds/. Safe to re-run: completed stages are skipped.
set -uo pipefail
cd "$(dirname "$0")/.."

SEEDS=("$@")
[ ${#SEEDS[@]} -eq 0 ] && SEEDS=(1 2 3 4)
D=365
MAX_CONCURRENT=${MAX_CONCURRENT:-3}

# Each run already parallelises over 4 envs; cap BLAS threads so concurrent
# seeds don't oversubscribe the 16 cores.
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

mkdir -p logs/seeds

run_seed() {
  local s=$1
  local log="logs/seeds/s${s}.log"
  {
    echo "=== seed $s started $(date -Is) ==="

    if [ ! -f "data/history-d${D}-s${s}.npz" ]; then
      echo "--- collect_history ---"
      uv run --extra train python scripts/collect_history.py --days $D --seed "$s" || return 1
    else
      echo "--- history exists, skipping collect ---"
    fi

    if [ ! -f "models/market-model-d${D}-s${s}.pt" ]; then
      echo "--- validate_market_model (gate) ---"
      uv run --extra train python scripts/validate_market_model.py \
        --budget-days $D --seed "$s" \
        --history "data/history-d${D}-s${s}.npz" \
        --out "diagnostics/gate-s${s}" --no-wandb || return 1
    else
      echo "--- gated ensemble exists, skipping fit ---"
    fi

    # dyna only runs if the gate wrote a certificate for this seed.
    if [ -f "models/market-model-d${D}-s${s}.pt" ]; then
      if [ ! -f "models/dyna/dyna-d${D}-s${s}/best_model.zip" ]; then
        echo "--- train dyna ---"
        uv run --extra train python scripts/train_dyna.py \
          --arm dyna --budget-days $D --seed "$s" --no-wandb || return 1
      else
        echo "--- dyna checkpoint exists, skipping ---"
      fi
    else
      echo "--- GATE FAILED at seed $s: no dyna arm (this is a result, not an error) ---"
    fi

    if [ ! -f "models/dyna/replay-d${D}-s${s}/best_model.zip" ]; then
      echo "--- train replay ---"
      uv run --extra train python scripts/train_dyna.py \
        --arm replay --budget-days $D --seed "$s" --no-wandb || return 1
    else
      echo "--- replay checkpoint exists, skipping ---"
    fi

    echo "=== seed $s finished $(date -Is) ==="
  } >>"$log" 2>&1
}

for s in "${SEEDS[@]}"; do
  while [ "$(jobs -rp | wc -l)" -ge "$MAX_CONCURRENT" ]; do sleep 20; done
  echo "launching seed $s (log: logs/seeds/s${s}.log)"
  run_seed "$s" &
done
wait
echo "all seeds done"
