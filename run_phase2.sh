#!/usr/bin/env bash
# Phase 2 — Qwen2.5-1.5B-Instruct, sequential on a single A100-80GB. ~4h.
# Order is fail-fast: validate the full train->save->eval path on ONE cohort
# before committing hours to signals + the other four.
#   0. base eval (headroom check — want ~0.55-0.70; if >0.78 stop & go smaller)
#   1. benchmark_slice: GRPO train -> eval  (GATE: exit if either fails)
#   2. extract_signals_math.py  (predictor side, all cohorts)
#   3. remaining 4 cohorts: train -> eval -> commit
# Run inside tmux:  ./run_phase2.sh 2>&1 | tee phase2.log
# Optional arg = max_steps (default 120):  ./run_phase2.sh 150
set -uo pipefail

export REPORT_TO=wandb
export WANDB_PROJECT=grpo-cohorts
STEPS="${1:-120}"

train_eval () {  # $1 = cohort name
  local C="$1"
  export RUN_NAME="math_${C}_seed0"
  echo "================ START $C $(date) ================"
  python3 grpo_math.py --cohort "$C" --max_steps "$STEPS" --seed 0 || return 1
  python3 eval_math.py --model "outputs/${RUN_NAME}" --label "$C" --n 200 || return 1
  git add results/math_eval.jsonl
  git commit -m "eval: $C (1.5B-Instruct)" && git push
  echo "================ DONE  $C $(date) ================"
}

echo "######## PHASE 2 START $(date) | steps=$STEPS ########"

# 0. base headroom check (eyeball the printed accuracy)
echo "-------- base eval (headroom check) $(date) --------"
python3 eval_math.py --label base --n 200

# 1. fail-fast gate: full pipeline on one cohort
if ! train_eval benchmark_slice; then
  echo "!!!! benchmark_slice train/eval FAILED — fix before running the rest. Exiting."
  exit 1
fi
echo ">>>> Pipeline validated. Continuing with signals + remaining cohorts."

# 2. predictor side: signals on the Instruct model (overwrites broken-base ones)
echo "-------- extract_signals_math.py $(date) --------"
if python3 extract_signals_math.py; then
  git add cohorts/*_signals.jsonl cohorts/summary.jsonl results/math_signals.jsonl
  git commit -m "Phase 1: signals on 1.5B-Instruct" && git push
else
  echo "!!!! signal extraction FAILED — predictor side incomplete, but lifts will continue."
fi

# 3. remaining cohorts (continue even if one errors)
for C in harder_sibling synthetic_good synthetic_degraded random_control; do
  train_eval "$C" || echo "!!!! $C FAILED — continuing !!!!"
done

echo "######## PHASE 2 COMPLETE $(date) ########"
echo "---- results/math_eval.jsonl (base + lifts) ----"
cat results/math_eval.jsonl
