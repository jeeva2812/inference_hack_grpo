# Progress notes

## Done
- `requirements.txt` — torch, transformers, trl, datasets, accelerate, vllm.
- `grpo_baseline.py` — Qwen2.5-Math-1.5B + GSM8K + GRPO, 50-step smoke test
  ran end-to-end on Prime Intellect H100. Loss/reward print, no OOM. 50 steps
  is too few for real lift (clip_ratio stayed 0) — that's expected.
- `slice_cohorts.py` — generated 5 cohorts (256 tasks each) in `cohorts/`.
- `extract_signals.py` — perplexity + N-rollout reward stats per task, with
  pad-token fix and attention-mask passthrough. Currently running.
- `PROBLEM.md` — restatement of the brief and our adapted plan.
- `GAMEPLAN.md` — strategy doc. **Read this first.**

## Next (in order)
1. Let `extract_signals.py` finish all 5 cohorts → `summary.jsonl`.
2. **Eval harness** — fixed GSM8K test slice, greedy decoding, returns
   accuracy. Will be called before and after each cohort run.
3. **Re-cohort by pass-rate** using the signal output: easy / medium / hard /
   mixed buckets. Replaces the current `(steps × qlen)` cut for the main
   experiment (we keep the original as a secondary axis).
4. **5 GRPO training runs**, 200 steps each, `num_generations=8`,
   identical config across cohorts. Save checkpoints.
5. Eval all 5 checkpoints + base = 6 accuracy numbers → 5 lifts.
6. Compute extras: gold-trajectory surprise, self-consistency gap.
7. Regression: metric → lift, leave-one-out. Spearman ρ, not R².
8. Pareto plot: (compute-cost, ρ) per metric.

## Open questions / risks
- vLLM + GRPO on a single small GPU — might have to disable `use_vllm`
  (currently False in baseline; flip on once we have headroom).
- Flash-attn wheel availability on Prime Intellect base image — removed
  the `attn_implementation` kwarg from the baseline, model loads fine without.
- 50 steps showed clip_ratio=0 → model barely moved. For real cohort runs
  bump to 200–300 steps and `num_generations=8`.
- With only 5 cohorts, R² is meaningless — report Spearman LOO instead.

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
