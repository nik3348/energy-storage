#!/usr/bin/env bash
# Estimate how often a D-day draw yields a gate-passing market model.
#
# Gate only: each seed draws its own history, fits its own ensemble, and is
# scored. No policy trains. A pass writes models/market-model-d{D}-s{seed}.pt,
# which is what a later dyna run would need, so a passing seed can be picked up
# by run_seed_sweep.sh without refitting.
#
#   scripts/run_gate_sweep.sh 5 6 7 8 9
#
# Results land in logs/gate-sweep/ and a one-line-per-seed summary in
# logs/gate-sweep/verdicts.txt. Safe to re-run: fitted seeds are skipped.
set -uo pipefail
cd "$(dirname "$0")/.."

SEEDS=("$@")
[ ${#SEEDS[@]} -eq 0 ] && SEEDS=(5 6 7 8 9)
D=${D:-365}
MAX_CONCURRENT=${MAX_CONCURRENT:-4}

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

mkdir -p logs/gate-sweep diagnostics

run_seed() {
  local s=$1
  local log="logs/gate-sweep/s${s}.log"

  if [ -f "models/market-model-d${D}-s${s}.pt" ]; then
    echo "s${s} PASS (already fitted)" >>logs/gate-sweep/verdicts.txt
    return 0
  fi
  {
    echo "=== seed $s started $(date -Is) ==="
    if [ ! -f "data/history-d${D}-s${s}.npz" ]; then
      uv run --extra train python scripts/collect_history.py --days $D --seed "$s" || return 1
    fi
    uv run --extra train python scripts/validate_market_model.py \
      --budget-days $D --seed "$s" \
      --history "data/history-d${D}-s${s}.npz" \
      --out "diagnostics/gate-s${s}" --no-wandb
    echo "=== seed $s finished $(date -Is) ==="
  } >>"$log" 2>&1

  # validate_market_model.py exits 1 on a clean gate failure as well as on a
  # crash, so the verdict is read from the log rather than the exit code.
  if grep -q "GATE: PASS" "$log"; then
    echo "s${s} PASS" >>logs/gate-sweep/verdicts.txt
  elif grep -q "GATE: FAIL" "$log"; then
    local failed
    failed=$(grep -E "FAIL$" "$log" | awk '{print $1}' | sort -u | tr '\n' ',' | sed 's/,$//')
    echo "s${s} FAIL (${failed})" >>logs/gate-sweep/verdicts.txt
  else
    echo "s${s} ERROR (no verdict in log)" >>logs/gate-sweep/verdicts.txt
  fi
}

for s in "${SEEDS[@]}"; do
  while [ "$(jobs -rp | wc -l)" -ge "$MAX_CONCURRENT" ]; do sleep 10; done
  echo "launching seed $s (log: logs/gate-sweep/s${s}.log)"
  run_seed "$s" &
done
wait
echo "=== all gate fits done ==="
sort -V logs/gate-sweep/verdicts.txt
