#!/usr/bin/env bash
# run_code_track.sh — Prime Intellect driver for the CODE track.
# Run INSIDE tmux + an activated venv. Each stage is independently runnable.
#
#   bash run_code_track.sh check      # (no GPU) offline logic tests
#   bash run_code_track.sh inference  # (GPU) load model, generate, verify — smoke_test_code
#   bash run_code_track.sh grpo1      # (GPU) ONE short GRPO run + eval — the "test one sample" step
#   bash run_code_track.sh phase1     # (GPU) base eval (acc_before) + signals on all cohorts
#   bash run_code_track.sh phase2     # (GPU) train+eval each cohort, then merge -> summary
#   bash run_code_track.sh all        # check -> inference -> grpo1 -> phase1 -> phase2
#
# Knobs (env): EVAL_N (default 200), SMOKE_STEPS (15), MAX_STEPS (250),
#              NUM_GENERATIONS (8), REPORT_TO=wandb WANDB_PROJECT=grpo-cohorts
set -euo pipefail

COHORTS="benchmark_slice harder_sibling synthetic_good synthetic_degraded random_control"
EVAL_N="${EVAL_N:-200}"

check() {
  echo "== STEP 1: offline logic tests (no GPU) =="
  python test_executor.py
  python test_pipeline.py
}

inference() {
  echo "== STEP 2: inference smoke (loads the model, generates, verifies) =="
  python smoke_test_code.py --n 3
}

grpo1() {
  echo "== STEP 3: ONE short GRPO run + eval (proves training end-to-end) =="
  MAX_STEPS="${SMOKE_STEPS:-15}" NUM_GENERATIONS="${NUM_GENERATIONS:-8}" \
    RUN_NAME=code_smoke python grpo_code.py --cohort benchmark_slice
  python eval_code.py --model outputs/code_smoke --label smoke_check --n 16
  echo ">> if a checkpoint saved in outputs/code_smoke and eval printed an accuracy, GRPO works."
}

phase1() {
  echo "== STEP 4a / Phase 1: base eval (acc_before) + signals on all cohorts (base model) =="
  python eval_code.py --label base --n "$EVAL_N"
  python extract_signals_code.py
}

phase2() {
  echo "== STEP 4b / Phase 2: one identical GRPO run per cohort, eval each =="
  for c in $COHORTS; do
    echo "--- cohort: $c ---"
    RUN_NAME=code_$c python grpo_code.py --cohort "$c"
    python eval_code.py --model "outputs/code_$c" --label "$c" --n "$EVAL_N"
  done
  python merge_summary.py --domain code
  echo ">> results/code_summary.jsonl ready. Now: git add results/ && git commit && git push"
}

case "${1:-all}" in
  check)     check ;;
  inference) inference ;;
  grpo1)     grpo1 ;;
  phase1)    phase1 ;;
  phase2)    phase2 ;;
  all)       check; inference; grpo1; phase1; phase2 ;;
  *) echo "usage: bash run_code_track.sh {check|inference|grpo1|phase1|phase2|all}"; exit 1 ;;
esac
