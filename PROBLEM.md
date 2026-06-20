# Inference-Time Compute Hackathon — Applied AI Track

## Question
What **task-level** and **dataset-level** metrics correlate with a model's
post-RL performance gain? Why? Where do these metrics sit on the
cost-quality Pareto frontier?

The judges care more about the **rationale** for choosing metrics than the
exact numbers.

## Primitives (suggested)
- `T` task string
- `M(T) -> y` model rollout
- `V(y) -> {criterion: score}` verifier
- `r(V(y)) -> float` reward reducer

## Recommended recipe
1. Pick a small open model + cheap benchmark (we picked **Qwen2.5-1.5B-Instruct**
   + **GSM8K**). The non-Instruct base couldn't follow the prompt zero-shot, and the
   *math*-Instruct variant was too strong (~0.855 → no headroom → null lift); the
   general 1.5B-Instruct leaves room for lift to vary. See NOTES.md 2026-06-20.
2. Partition GSM8K into **cohorts** of equal size, varying one property
   (difficulty, length, topic, synthetic vs real, etc.).
3. Run **GRPO** on each cohort with identical config — cohort is the only
   variable.
4. Eval each trained model: `lift = acc_after - acc_before`.
5. Propose cheap per-cohort metrics that should predict lift from first
   principles. Candidates worth trying:
   - **model-dependent**: pass rate, reward variance, rollout entropy,
     intermediate-difficulty (pass-rate near 0.5), learnability /
     reducible-loss, branching factor
   - **model-agnostic**: token length, Vendi diversity, redundancy
6. Fit `metric -> lift` on some cohorts, measure R² / RMSE / rank-corr
   on held-out cohorts. Show the scatter.

## Resources we actually have
- Prime Intellect: **$100** credits (NOT the 8×H100 the brief assumes).
- Anthropic API: $100.
- Single small GPU box is the realistic budget — every cohort run must be
  cheap. Likely 1×H100 or 1×A100, time-boxed.

## Implication for plan
- Cap cohort size small (e.g. 256–512 tasks).
- Cap GRPO steps per cohort (e.g. 100–200).
- Few generations per prompt (4).
- Use vLLM for rollouts to stay inside the budget.
- Reuse the same eval set (a fixed GSM8K test slice) for every cohort.
