#!/usr/bin/env bash
# Repeat the D=365 dyna-vs-replay comparison over training seeds, with the
# real-data draw HELD FIXED.
#
# This is the complement to run_seed_sweep.sh. That script varies the seed over
# the whole pipeline (each seed draws its own 365 days and refits its own
# ensemble), which confounds data-draw variance with training-run variance and
# lets the calibration gate block the dyna arm. Here every run reads seed 0's
# history and seed 0's gated ensemble, so the only thing the seed changes is
# the training run itself: SAC initialisation, env stream, and which stored
# days the planner demonstrations visit. Both arms therefore train at every
# seed, which is what makes the paired dyna - replay difference measurable more
# than once.
#
#   scripts/run_fixed_history_sweep.sh 1 2 3 4
#
# Seed 0 is deliberately absent: models/dyna/{dyna,replay}-d365-s0 were already
# trained on exactly these pinned artifacts and count as the seed-0 point.
# Logs land in logs/fixed-history/. Safe to re-run: completed runs are skipped.
set -uo pipefail
cd "$(dirname "$0")/.."

SEEDS=("$@")
[ ${#SEEDS[@]} -eq 0 ] && SEEDS=(1 2 3 4)
D=365
HISTORY="data/history-d${D}-s0.npz"
MODEL="models/market-model-d${D}-s0.pt"
MAX_CONCURRENT=${MAX_CONCURRENT:-3}

for f in "$HISTORY" "$MODEL"; do
  [ -f "$f" ] || { echo "missing pinned artifact: $f" >&2; exit 1; }
done

# Each run already parallelises over 4 envs; cap BLAS threads so concurrent
# runs don't oversubscribe the 16 cores.
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

mkdir -p logs/fixed-history

run_arm() {
  local arm=$1 s=$2
  local dir="models/dyna/${arm}-d${D}-fixedhist-s${s}"
  local log="logs/fixed-history/${arm}-s${s}.log"

  if [ -f "$dir/best_model.zip" ]; then
    echo "--- ${arm} s${s} already trained, skipping ---" >>"$log"
    return 0
  fi
  {
    echo "=== ${arm} seed ${s} started $(date -Is) ==="
    uv run --extra train python scripts/train_dyna.py \
      --arm "$arm" --budget-days $D --seed "$s" \
      --history "$HISTORY" --model "$MODEL" \
      --models-dir "$dir" --no-wandb
    echo "=== ${arm} seed ${s} finished $(date -Is) rc=$? ==="
  } >>"$log" 2>&1
}

for s in "${SEEDS[@]}"; do
  for arm in dyna replay; do
    while [ "$(jobs -rp | wc -l)" -ge "$MAX_CONCURRENT" ]; do sleep 20; done
    echo "launching ${arm} seed ${s} (log: logs/fixed-history/${arm}-s${s}.log)"
    run_arm "$arm" "$s" &
  done
done
wait
echo "all runs done"
