# Progress notes

## 2026-06-20 (late) — Model: general 1.5B-Instruct, NOT math-Instruct (READ THIS)
Fixing the base model (below) got `Qwen2.5-Math-1.5B-Instruct` to **0.855** on the
GSM8K eval slice — but that's a **ceiling that would have produced a null result**:
- Good cohorts: model already ~85% right → no headroom → GRPO can't improve → lift ≈ 0.
- Corrupt cohorts: model never *outputs* the wrong/mismatched gold → reward almost
  never fires → near-zero gradient → model barely changes → lift ≈ 0.
- ⇒ every cohort lands at ~0 lift, inside eval noise. No variance to predict.

**Fix:** switched to the general `Qwen/Qwen2.5-1.5B-Instruct` (~0.55–0.70 on GSM8K).
Headroom lets good cohorts climb (positive lift) while corrupt cohorts stay flat →
the **measurable, predictable lift variance** the experiment depends on. `MODEL_ID`
is now centralized in `math_common.py` (one edit changes grpo/eval/signals).
Run order is fail-fast: base eval (headroom check, want 0.55–0.70; if >0.78 drop to
`Qwen2.5-0.5B-Instruct`) → validate one cohort end-to-end → signals → remaining 4.
See `run_phase2.sh`. Expected lift order: benchmark_slice ≥ synthetic_good ≥
harder_sibling > synthetic_degraded ≈ random_control; `reward_mean`/`reward_var`
should track it.

## 2026-06-20 — Phase 2 debug: model switch (READ THIS)
First full Phase-2 run produced a **flat, cohort-overlapping reward curve** and
base GSM8K acc of only **~0.20**. Root-caused with `sanity_dump.py`:

- **The base `Qwen2.5-Math-1.5B` is unusable zero-shot.** Not instruction-tuned →
  can't follow the chat prompt: it rambles past the answer (no EOS, hits
  `max_completion_length` every time), degenerates into repetition / non-English
  garbage, and never emits `\boxed{}` or a clean `<answer>`. It sometimes *solved*
  the problem but buried the number in junk, so it scored ~0.
- **Fix → `Qwen2.5-Math-1.5B-Instruct`** (commit 67bc893). Stops cleanly, boxes,
  ~0.80 on GSM8K. Switched `MODEL_ID`/`DEFAULT_MODEL` in grpo/eval/signals.
- **Extraction hardened** (`math_common.extract_pred`): read `\boxed{}` (last box)
  and require ≥1 digit inside `<answer>` — an echoed literal `<answer>...</answer>`
  template was matching and returning `"..."`.
- **Config:** `lr_scheduler_type="constant"` @ `2e-6` (linear decay was zeroing the
  LR by ~step 110, so the back third of each run did nothing).

**Update:** ~0.855 turned out to be a ceiling that risks a null result — see the
"general 1.5B-Instruct" note above, which supersedes the model choice. Still true:
**must re-run signals + retrain all cohorts + re-eval base** on the new model (old
results were on the broken base model).

## Done
- `requirements.txt` — torch, transformers, trl, datasets, accelerate, vllm.
- `grpo_math.py` — Qwen2.5-1.5B-Instruct + GSM8K + GRPO, 50-step smoke test
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
