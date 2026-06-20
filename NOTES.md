# Progress notes

## Done
- `requirements.txt` — torch, transformers, trl, datasets, accelerate, vllm.
- `grpo_math.py` — Qwen2.5-Math-1.5B + GSM8K + GRPO, 50-step smoke test
  ran end-to-end on Prime Intellect H100. Loss/reward print, no OOM. 50 steps
  is too few for real lift (clip_ratio stayed 0) — that's expected.
- `slice_cohorts_math.py` — generated 5 cohorts (256 tasks each) in `cohorts/`.
  NOTE: this is the OLD (steps × qlen) cut. Game plan now wants source-diverse
  cohorts (benchmark slice / sibling dataset / synthetic / control) — to redo.
- `extract_signals_math.py` — perplexity + N-rollout reward stats per task, with
  pad-token fix and attention-mask passthrough. Ran on cohorts.
- `signals.py` — **shared, domain-agnostic signal module.** Pure functions on
  primitives (no GPU), imported by BOTH math and code tracks so a signal means
  the same thing in each. Implements baselines + both hero candidates
  (`sampling_headroom`, `gradient_coherence` + sketched variant) + your
  `benchmark_coverage` idea. `python signals.py` runs an offline fake-data
  demo — verified headroom peaks at medium difficulty as theorized.
- `PROBLEM.md` — restatement of the brief and our adapted plan.
- `GAMEPLAN.md` — strategy doc. **Read this first.** Rewritten: phase flow,
  source-diverse cohorts, metric bake-off, two-lane parallel schedule.
- `DASHBOARD.md` / `PRIME_INTELLECT.md` — demo spec + GPU runbook.
- `make_synthetic_math.py` — programmatic GSM8K corruption (wrong_answer /
  shuffled_steps / mismatched / trivial). ZERO tokens — builds the degraded /
  control cohorts without Claude.
- `plots.py` — the two deliverable figures (predicted-vs-actual scatter +
  cost-vs-ρ Pareto), numpy-only stats, runs offline on fake data. Verified.
- `test_pipeline.py` — 10 offline sanity checks (no GPU). `python
  test_pipeline.py` → 10/10. Run after editing signals/synthetic/plots.
- `grpo_math.py` — now W&B-aware via env vars (REPORT_TO / WANDB_PROJECT /
  RUN_NAME), defaults to terminal-only so it never breaks.
- `eval_math.py` — **the dependent variable.** Greedy, deterministic accuracy on a
  fixed GSM8K test slice. Run before (`--label base`) and after each cohort
  (`--model <checkpoint> --label <cohort>`); appends to
  `results/math_eval.jsonl` and prints lift vs base. Heavy imports are lazy so
  extraction logic is unit-tested offline. Smoke: `python eval_math.py --n 8`.

## Next (in order)
1. ~~Eval harness~~ ✅ DONE (`eval_math.py`).
2. **Rebuild cohorts** as source-diverse, size-matched (see GAMEPLAN
   "Training data vs. eval benchmark"): benchmark_slice / harder_sibling /
   synthetic_good / synthetic_degraded / random_control.
3. **Wire `extract_signals_math.py` to emit `signals.py` primitives** (`Task`
   objects) + add optional embedder & gradient hook for tiers 2.5 / 3.
4. **GRPO run per cohort**, 200–300 steps, `num_generations=8`, identical
   config. Seed-repeat if budget allows. Save → eval → `lift`.
5. Regression: metric → lift, leave-one-out, Spearman ρ (not R²).
6. Dashboard: predicted-vs-actual scatter + cost-vs-ρ Pareto plot.

## Open questions / risks
- vLLM + GRPO on a single small GPU — might have to disable `use_vllm`
  (currently False in baseline; flip on once we have headroom).
- Flash-attn wheel availability on Prime Intellect base image — removed
  the `attn_implementation` kwarg from the baseline, model loads fine without.
- 50 steps showed clip_ratio=0 → model barely moved. For real cohort runs
  bump to 200–300 steps and `num_generations=8`.
- With only 5 cohorts, R² is meaningless — report Spearman LOO instead.
- `benchmark_coverage` is both a candidate metric AND a confound control:
  source-diverse cohorts vary in distribution-match to the eval set, so we
  must regress lift on headroom *while controlling for* coverage.
- `gradient_coherence` needs a backward pass per task — heavier than the
  rollout-reuse signals. Use the sketched variant to keep it cheap; build it
  last (sharp-but-safe: baselines + headroom ship first).

## VS Code + SSH + Claude Code (your sidebar question)
- Install the **Remote - SSH** extension in VS Code.
- `Ctrl+Shift+P` → "Remote-SSH: Connect to Host" → add your Prime Intellect
  host (or paste the `ssh user@host -p PORT` line; VS Code parses it).
- Once connected the VS Code window is "on" the remote. Open the integrated
  terminal (`Ctrl+``) — that shell is already on the remote box.
- Run `claude` in that terminal. Claude Code runs remotely, sees the
  remote filesystem, and the VS Code editor pane shows the same files.
  No port forwarding needed.
- Tip: put your Prime Intellect SSH key in `~/.ssh/config` with a `Host`
  alias so VS Code remembers it. Keep a `tmux` session on the remote so
  long GRPO runs survive if your laptop sleeps:
  `tmux new -s grpo` then run `claude` (or training) inside it.
