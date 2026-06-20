#!/usr/bin/env bash
# Phase 2 (Instruct model), sequential on a single A100-80GB. ~4h total.
#   1. recompute signals on the Instruct model (predictor side)
#   2. per cohort: GRPO train -> eval checkpoint -> commit lift
# Run inside tmux:  ./run_phase2.sh 2>&1 | tee phase2.log
# Optional arg = max_steps (default 120):  ./run_phase2.sh 80
set -uo pipefail

export REPORT_TO=wandb
export WANDB_PROJECT=grpo-cohorts

COHORTS="benchmark_slice harder_sibling synthetic_good synthetic_degraded random_control"
STEPS="${1:-120}"

echo "######## PHASE 2 START $(date) | steps=$STEPS ########"

# ---- Predictor side: signals on the Instruct model (overwrites broken-base ones)
echo "-------- extract_signals_math.py $(date) --------"
if python3 extract_signals_math.py; then
  git add cohorts/*_signals.jsonl cohorts/summary.jsonl results/math_signals.jsonl
  git commit -m "Phase 1: signals on Instruct" && git push
else
  echo "!!!! signal extraction FAILED — fix before trusting the predictor side !!!!"
fi

# ---- Outcome side: train + eval each cohort (config identical, only data varies)
for C in $COHORTS; do
  echo "================ START $C $(date) ================"
  export RUN_NAME="math_${C}_instruct_seed0"
  if python3 grpo_math.py --cohort "$C" --max_steps "$STEPS" --seed 0; then
    python3 eval_math.py --model "outputs/${RUN_NAME}" --label "$C" --n 200
    git add results/math_eval.jsonl
    git commit -m "eval: $C (Instruct)" && git push
  else
    echo "!!!! $C training FAILED — skipping eval, continuing !!!!"
  fi
  echo "================ DONE  $C $(date) ================"
done

echo "######## PHASE 2 COMPLETE $(date) ########"
echo "---- results/math_eval.jsonl (base + lifts) ----"
cat results/math_eval.jsonl
