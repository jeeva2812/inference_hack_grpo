# Game plan — winning this hackathon

## North star
The judges said it twice: **they care about the rationale**, not the absolute
numbers. So our deliverable is a *story*:

> "Here is a ladder of metrics from free to expensive. Here is *why* each
> one should predict GRPO lift, derived from first principles about how GRPO
> actually works. Here is where each one sits on the cost-quality frontier.
> Here are the ones that broke the frontier — and why we think they did."

If we ship 3 metrics with sharp reasoning > 10 metrics with handwavy hope.

---

## Two-person split: math (me) ∥ code (teammate)

We run **two domains in parallel**, sharing the *same metric definitions*
and the *same regression machinery*. This is not just "more data" — it is
the strongest validation we can offer:

> **If a metric predicts GRPO lift on BOTH math and code, it is
> domain-general. If it only works on one, it is a domain artifact.**

That cross-domain claim is what most teams won't have. It turns "we found a
correlation on GSM8K" into "we found a *property of learnable data* that
holds across modalities."

### Track A — Math (owner: me / this repo)
- Model: `Qwen2.5-Math-1.5B`
- Benchmark: GSM8K
- Verifier: regex on final `<answer>` numeric match
- Status: baseline trains, signals extracting, cohorts sliced.

### Track B — Code (owner: teammate)
- Model: small **code** model — recommend `Qwen2.5-Coder-1.5B`
  (matches our family/size, so cohort-size and step-count configs transfer
  with minimal retuning).
- Benchmark: **MBPP+** or **HumanEval+** (brief lists both as approved).
  Recommend **MBPP+** — more tasks (~400) → more room to slice cohorts.
- Verifier: **execute unit tests in a sandbox**, reward = fraction of tests
  passed. This is the one genuinely new piece — no regex, needs a safe
  subprocess runner with a timeout.
- Cohorts: same *kinds* of axes — pass-rate buckets (easy/med/hard) by
  base-model test-pass-rate.

### Shared contract (the thing that makes parallel work)
Both tracks must emit cohort summaries in the **same schema** so one
regression script consumes both:

```
{ "domain": "math" | "code",
  "cohort": str,
  "n_tasks": int,
  "signals": { "self_consistency_gap": float, "reward_var": float,
               "prompt_ppl": float, ... },
  "lift": float }      # acc_after - acc_before
```

Agree this JSON contract FIRST. Then the two of us can build independently
and the analysis just concatenates both files. Metric definitions
(self-consistency gap, reward variance) are **domain-agnostic** — they only
depend on the rollout reward, not on whether it came from regex or unit
tests. So the teammate reuses our `extract_signals.py` logic and only
swaps the verifier + dataset loader.

### What's genuinely different for code (teammate's TODO)
1. **Sandbox executor** — run generated code against test cases in a
   subprocess with a hard timeout + no network. This is the only real new
   infra. Everything else mirrors the math track.
2. **Prompt format** — code completion / function-signature style, not
   `<answer>` tags.
3. **Reward** — continuous (fraction of tests passed) rather than binary.
   Nice bonus: continuous reward gives *richer* `reward_var` signal.

### Division of GPU time
Two tracks = two model families loaded. **Do not run both on one A100 at
once** (OOM risk + contention). Either:
- (a) Time-share one A100: math runs, then code runs. Safer on $100 budget.
- (b) Spin a second A100 for the teammate if the credit pool allows
  (~$2/hr × ~6 hr = ~$12 each, still cheap). Faster wall-clock.
Recommend (b) if credits are shared and healthy — parallel wall-clock is
worth $12.

---

## What we have working
- [x] GRPO baseline trains (50-step smoke test, gradients flow, rewards non-zero).
- [x] 5 cohorts sliced by `(steps × qlen)` median split + a random control.
- [x] Signal extractor running: `prompt_ppl`, `reward_mean`, `reward_var`,
      `format_rate`, `mean_length`.

## What we still need
- [ ] Held-out **eval harness** — fixed GSM8K test slice, deterministic
      (greedy or temp=0), reports `acc_before` and `acc_after` per cohort.
      This is the dependent variable; without it nothing else matters.
- [ ] **Real cohort training runs** — 200–300 steps each, same config,
      `num_generations=8`. ~80 min/cohort × 5 cohorts ≈ 7 GPU-hours.
- [ ] **Metric → lift regression** — fit on 4 cohorts, leave-one-out.
- [ ] **Scatter plot** of predicted vs actual lift. This is the money slide.

---

## The metric ladder (cheap → expensive)

We are explicitly trying to span the Pareto frontier, not crowd one corner.

| Tier | Metric | Cost | Why it should predict lift |
|---|---|---|---|
| **Free** | `reasoning_steps` (count of `<<...>>`) | 0 | Difficulty proxy. Mid-difficulty = most learnable. |
| **Free** | `qlen`, `num_count` | 0 | Distractor / complexity proxy. |
| **Cheap (1 fwd)** | `prompt_ppl` | 1 forward pass | Tasks the model finds "novel" but not alien. |
| **Cheap (1 fwd)** | `gold_token_surprise` *(TODO)* | 1 forward pass on gold solution | Sum of `(1 - p_model[gold_tok])`. Measures the model's "headroom" on this exact answer. |
| **Med (N rollouts)** | `reward_var` | N generations | **Most mechanistically grounded.** GRPO's gradient ∝ within-group reward std. Zero variance = zero gradient. This *must* correlate or our understanding of GRPO is wrong. |
| **Med (N rollouts)** | `intermediate_frac` | N generations | Fraction of tasks with 0<pass-rate<1. The "Goldilocks pool." |
| **Med (N rollouts)** | `self_consistency_gap` *(TODO)* | N generations | `accuracy(majority-vote-of-N) − accuracy(greedy)`. Big gap = "model knows but can't commit" = ideal RL target. |
| **Med (N rollouts)** | `rollout_diversity` *(TODO)* | N generations + 1 embed | Mean pairwise embedding distance between completions. Captures *path* diversity, not just outcome. |
| **Expensive (grad probe)** | `learnability` *(stretch)* | 1 SGD step + eval | One-step RHO-loss: does fine-tuning on this task drop loss on a held-out slice? Ground-truth-ish predictor; expensive to compute. |

The pitch to the judges: the medium tier should dominate the frontier.
Cheap signals are too coarse; the expensive learnability probe approaches the
true answer but costs ~as much as just running the RL. The sweet spot is
`reward_var` + `self_consistency_gap` — they pay for themselves in 5 rollouts.

---

## Cohort design (see "Conservative scope" below for v1 cut)

The full ambition was 5 cohorts varying on different axes (difficulty,
length, synthetic). v1 ships just the 3 difficulty cohorts; v2 adds the
others *if and only if* v1 ran clean.

**Why varied axes matter (the v2 story):** a metric that predicts lift
only within a difficulty sweep is just a difficulty proxy. A metric that
predicts across difficulty + length + synthesis is capturing real
learnability. Save this argument for v2 / writeup.

**Stats honesty:** with 3–5 cohorts, R² is noise. Report **Spearman rank
correlation with leave-one-out CV**.

---

## Out-of-the-box ideas — pick 1, maybe 2

Brainstorm, ranked by "judge would say 'huh, interesting'":

1. **Self-consistency gap as a learnability oracle.**
   Pass-rate(majority-of-8) minus pass-rate(greedy). When this is large, the
   model *can* reach the answer with luck but doesn't consistently — exactly
   the gap RL closes. I'd bet this beats raw reward_var. Cheap: reuse the same
   rollouts.

2. **Gold-trajectory surprise.**
   Teacher-force the gold solution token-by-token, sum `(1 - p)`. Captures
   "how much novel info does this answer contain for the model" without ever
   generating. One forward pass per task. Predicted to correlate with lift
   because it's literally the cross-entropy budget RL is fighting.

3. **The "mixed cohort beats pure cohort" hypothesis.**
   Curriculum lit says diversity > purity for transfer. Train one cohort that
   is 60% medium / 40% mixed-hard. If it beats all four pure cohorts, that's
   a clean narrative win.

4. **Logit margin on the answer-emitting token.**
   When the model is about to emit the final number, what is `top-1 − top-2`?
   Low margin = uncertain = high learning potential. One generation per task,
   inspect logits at the answer position. Niche but cheap and novel.

5. **Negative-result honesty.**
   Pick one metric we *expect* to fail (e.g., raw token length) and report it
   failing. Shows we understand causation vs correlation. Judges love this.

6. **Cost on the X-axis.**
   Literally plot every metric on (compute-seconds, predictive Spearman ρ)
   axes — that's the Pareto frontier they explicitly asked about. Most teams
   will skip this and just hand-wave; we shouldn't.

**LOCKED: hero metric = self-consistency gap (#1).** Plus baselines we
already compute (`reward_var`, `prompt_ppl`) for the Pareto plot (#6).
Skip #2 unless we have GPU time left over. Skip #3, #4, #5 — scope discipline.

---

## Risks to watch

- **Lift is noisy at 200 steps.** Run each cohort twice with different seeds
  if budget allows; report mean lift.
- **Reward hacking on `<answer>` tag.** A model can game format reward
  without solving math. Already mitigated by weighting correctness 5× format,
  but worth eyeballing some final completions.
- **vLLM rollout might OOM on a single GPU.** If it does, fall back to HF
  generate (slower but works). Don't burn 2 hours debugging vLLM.
- **5 cohorts → R² is noise.** Pre-commit to reporting Spearman ρ with LOO.

---

## GPU discipline — when to bring the box up

You're on a single A100 with metered credits. Treat GPU time as the
scarce resource. Everything that doesn't need a GPU happens locally
(Windows box) while the GPU is **down**.

### Local-only work (GPU OFF)
- Editing scripts (this is what you're doing right now).
- Reading dataset rows, designing prompts, writing Claude synthetic-data
  prompts, writing regression + plot code on dummy data.
- Drafting the writeup against placeholder numbers.
- Anything involving me — I cost nothing GPU-wise.

### GPU-required work (bring it UP only for these)
1. **Signal extraction pass** (~1.5 hr) — `extract_signals.py` on all
   cohorts in one shot. Already in progress; let it finish.
2. **Pre-training eval** (~15 min) — base-model accuracy on the test slice.
   Cache the number.
3. **Training runs** (~1 hr × N cohorts) — back-to-back in one session,
   `tmux` so the laptop can sleep.
4. **Post-training eval** (~15 min × N) — same harness, on each saved
   checkpoint.

**Rule of thumb:** if you're about to spend >5 minutes editing code with
the GPU idle, shut the box down. Bring it back up only when you have a
batch of work queued.

## Conservative scope (driving + ~real-world day)

Trim from 5 cohorts to **3 cohorts** for v1. Ship a complete pipeline
end-to-end before considering expansion.

| Cohort (v1) | Selection rule |
|---|---|
| `easy_pass`    | pass-rate > 0.7 |
| `medium_pass`  | pass-rate ∈ [0.3, 0.7] |
| `hard_pass`    | pass-rate < 0.3 |

Add `long_context` and `synthetic_claude` only if v1 runs clean and you
still have budget. 3 cohorts × ~1 hr training = ~3 GPU-hours, leaves
plenty of cushion on $100 of credits.

### Time budget (revised)

| Block | Time | GPU? |
|---|---|---|
| Finish signal extraction (already running) | 1.5 h | ✅ |
| Write eval harness (offline, against fake data) | 0.5 h | ❌ |
| Re-cohort by pass-rate (offline script) | 0.2 h | ❌ |
| Pre-train eval on base model | 0.25 h | ✅ |
| 3 GRPO training runs, 200 steps, num_gen=8, tmux | 3 h | ✅ |
| Post-train eval × 3 checkpoints | 0.5 h | ✅ |
| Regression + Pareto plot (offline) | 1 h | ❌ |
| Writeup (offline) | 1.5 h | ❌ |
| **Total wall** | **~8.5 h** | **~5.5 GPU-h** |

A100 cost on Prime Intellect ≈ $1.5–2/hr → ~$10 of credit used. Huge
headroom for retries.

---

## What "winning" looks like

A 1-page writeup with:
1. One scatter: predicted lift vs actual lift, **math and code points on the
   same axes**. If a metric's points line up across both domains, that single
   plot makes the whole argument.
2. One Pareto plot: compute-cost vs predictive ρ for each metric.
3. Three paragraphs of *reasoning* explaining why `self_consistency_gap` and
   `reward_var` beat the cheaper metrics — and why the mechanism (GRPO
   gradient ∝ within-group reward variance) is *domain-independent*, which is
   why it should and does transfer from math to code.
4. One sentence of honesty: "n cohorts per domain, take this as a hypothesis
   not a conclusion." Judges respect that more than fake p-values.

The differentiator: most teams will show one correlation on one benchmark.
We show the *same* cheap metric predicting lift across two modalities. That's
the difference between "a correlation" and "a property of learnable data."
