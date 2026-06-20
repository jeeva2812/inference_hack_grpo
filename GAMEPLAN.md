# Game Plan

## TL;DR

We are answering: **which cheap, training-free metric best predicts how much
a model improves after GRPO?** Our edge is that we test the answer on **two
different domains with two different models and two different benchmarks** —
math and code. A metric that predicts RL lift on *both* is a property of
learnable data, not a quirk of one benchmark. That cross-domain claim is the
thing most teams won't have.

**We do not pre-pick a winner. We run a bake-off.** We compute a basket of
candidate metrics on each cohort and let the cross-domain regression crown
the one that actually predicts lift. The two we're most excited about (and
which most teams won't think of) are novel — but they earn the spotlight only
if the data backs them:

> - **Sampling headroom** = `pass@N − pass@1` — RL converts latent capability
>   into reliable capability. Cheap, sharp, our safe anchor.
> - **Gradient coherence** (dataset-level) — do the cohort's tasks pull the
>   model in the *same* direction? Captures inter-task synergy that no
>   per-task average can see. The high-ceiling bet.
> - **Baselines** (everyone has these): reward variance, prompt perplexity,
>   pass-rate, difficulty. These are the "cheap floor" we expect to beat.
> - **Deliverable:** one scatter (predicted vs actual lift, both domains) +
>   one Pareto plot (metric cost vs predictive power), with the winner chosen
>   empirically, not asserted.

The judges said twice they care about **rationale over numbers**. So every
candidate ships with a first-principles reason it *should* predict lift — and
we report which reasons survived contact with the data, including the ones
that didn't.

---

## The bet: why cross-domain wins

Most teams will pick one benchmark, slice it, train, and report one
correlation. That's "we found a correlation on GSM8K." It's fragile — the
metric might just be a difficulty proxy for that one dataset.

We instead run two independent tracks that share **only the metric
definitions and the analysis**:

| | Track A — Math | Track B — Code |
|---|---|---|
| Owner | me / this repo | teammate |
| Model | `Qwen2.5-Math-1.5B` | `Qwen2.5-Coder-1.5B` |
| Benchmark | GSM8K | MBPP+ (recommended) or HumanEval+ |
| Verifier | regex numeric match on `<answer>` | execute unit tests in a sandbox |
| Reward | binary (correct / not) | continuous (fraction of tests passed) |

The metrics (self-consistency gap, reward variance, perplexity) depend
**only on the rollout reward**, never on how the reward was produced. So the
same metric code runs on both tracks unchanged. If the metric → lift
relationship holds across both, we've shown something general. **That is the
win.**

---

## Training data vs. eval benchmark (read this — it's the core design call)

**The eval benchmark is FIXED. The training cohorts are the VARIABLE.**
`lift = accuracy_after − accuracy_before` is always measured on the same
held-out benchmark test slice (GSM8K test for math, MBPP+ test for code).
What changes between runs is *only the data we GRPO on*.

**Why we do NOT just slice the benchmark's own train split.**
If every cohort is a slice of GSM8K-train, all cohorts are the same quality,
same distribution — we're only varying *which* easy/in-distribution tasks we
pick. Lift will be similar across cohorts, the dependent variable barely
moves, and a metric has almost nothing to predict. We'd be fitting a line to
points that are all stacked on top of each other.

**What we do instead: cohorts from genuinely different sources.**
To get real spread in cohort *quality* (and therefore in lift), draw the
training pool from a mix:

| Cohort source | Math track | Code track | Expected quality |
|---|---|---|---|
| Benchmark train slice | GSM8K train | MBPP train | high (in-distribution) |
| Harder sibling dataset | MATH, GSM-Symbolic | HumanEval, LiveCodeBench | medium-high, distribution-shifted |
| Synthetic — good | Claude-generated, verified | Claude-generated + tests pass | variable |
| Synthetic — degraded | wrong/noisy CoT, shuffled steps | buggy tests, wrong refs | deliberately low |
| Random / off-task control | non-math text or trivial Qs | non-code or trivial snippets | near-zero (control) |

Each cohort stays **size-matched** (e.g. 256 tasks) so size is never a
confound — source/quality is the only variable. The degraded and control
cohorts are not filler: they anchor the *low-lift* end of the regression, and
they're exactly where a good metric should correctly predict "don't train on
this." A metric that only ranks good cohorts but can't flag bad data is half
a metric.

**The "coverage vs. eval" caveat → make it a signal, not a bug.**
When a cohort is from a shifted distribution, some lift (or lack of it) comes
from distribution match rather than learnability. Rather than fight this, we
*measure* it: `coverage` = mean embedding similarity between the cohort and
the eval set (Appendix A). It enters the basket as its own candidate and as a
control variable in the regression.

---

## Roles & ownership

- **Track A (math)** — this repo, owned by me. `grpo_baseline.py`,
  `slice_cohorts.py`, `extract_signals.py` already exist.
- **Track B (code)** — teammate. Forks the same three scripts, swaps the
  dataset loader + verifier (the sandbox executor is the only genuinely new
  piece), keeps everything else identical.
- **Shared, built once** — the cohort-summary JSON schema (Appendix C), the
  regression/analysis script, and the dashboard. These consume both tracks'
  outputs. Whoever finishes their track first builds these.

**Do this first:** both agree on the JSON contract in Appendix C *before*
writing track-specific code. If both tracks emit the same schema, the final
merge is a concatenation. If they don't, you lose the last hour reconciling
formats instead of analyzing.

---

## The flow

Five phases. Phases 1–2 need a GPU; 0, 3, 4 are offline. Each person runs
their own track through phases 1–2 in parallel on their own $100 box.

### Phase 0 — Contract & cohorts (OFFLINE, ~45 min)
**Goal:** lock the interface and assemble the training cohorts so the two
tracks can run independently.
- Agree the cohort-summary JSON schema (Appendix C).
- Build **source-diverse, size-matched cohorts** (256 tasks each) per the
  "Training data vs. eval benchmark" table. v1 minimum = 5 cohorts:
  `benchmark_slice`, `harder_sibling`, `synthetic_good`, `synthetic_degraded`,
  `random_control`. This deliberately spans the quality range so lift has
  spread to predict.
- Fix the eval test slice now (same one used before & after, every cohort).
- Each track writes its `slice_cohorts.py` equivalent → `cohorts/*.jsonl`.
  Synthetic cohorts use **Anthropic credits, not GPU** — generate them in
  this offline phase.
**Output:** cohort files on disk, fixed eval slice, agreed schema. No GPU yet.

### Phase 1 — Signals (GPU ON, ~1.5 h per track)
**Goal:** compute the cheap, training-free metrics that we'll later test as
lift predictors.
- Run `extract_signals.py` over every cohort: the full candidate basket
  (Appendix A). `prompt_ppl`, `reward_var`, `intermediate_frac`,
  `self_consistency_gap`, and `sampling_headroom` all reuse the same N
  rollouts — near-free. `gradient_coherence` adds N backward passes (do it in
  the same pass while the model + cohort are loaded).
- Aggregate to one row per cohort.
**Output:** `results/<domain>_signals.jsonl`. This is the **predictor** side
of the regression. Computed *before any training* — that's the whole point,
the metric must be cheap.

### Phase 2 — Train & measure lift (GPU ON, ~4 h per track)
**Goal:** get the ground-truth `lift` for each cohort — the thing the metric
is trying to predict.
1. **Pre-train eval** — base model accuracy on a fixed held-out test slice
   (greedy decoding, deterministic). Cache `acc_before`. ~15 min.
2. **One GRPO run per cohort** — identical config across cohorts so *cohort
   is the only variable*. 200–300 steps, `num_generations=8`, in `tmux`.
3. **Post-train eval** — same test slice, each checkpoint → `acc_after`.
4. `lift = acc_after − acc_before` per cohort.
**Output:** `results/<domain>_summary.jsonl` (signals + lift, per cohort).
**De-noising:** if budget allows (it does — see logistics), run each cohort
2–3× with different seeds and report mean lift. With few cohorts, noise is
the enemy; seed repeats are the cheapest defense.

### Phase 3 — Analysis (OFFLINE, ~1 h)
**Goal:** quantify how well each metric predicts lift, honestly.
- Concatenate both domains' summaries into one dataframe.
- For each candidate metric: fit metric → lift, **leave-one-out CV**, report
  **Spearman rank correlation** (with n≈3–6 cohorts, R² is noise — don't use
  it).
- Compute each metric's **compute cost** (seconds to produce) for the
  frontier.
**Output:** a small table: metric, cross-domain ρ, cost.

### Phase 4 — Dashboard & writeup (OFFLINE, ~2 h)
**Goal:** make the result land in 60 seconds. See `DASHBOARD.md`.
- **Money plot:** predicted vs actual lift, math + code on the same axes,
  y=x line, metric-selector dropdown.
- **Pareto plot:** metric cost (x, log) vs predictive ρ (y) — the brief asks
  for this *by name*.
- Three paragraphs of rationale (Appendix A) + one sentence of honesty about
  small n.
**Output:** the demo.

---

## Parallel execution schedule (who runs what, when)

Two people, two own-$100 A100 boxes, running the same phases on different
domains. The only hard dependency is the **Phase-0 schema sync**; after that
the lanes are independent until they rejoin for analysis. ⚙️ = GPU on,
💻 = laptop/offline.

```
        LANE A — Math (you)                  LANE B — Code (teammate)
        ───────────────────                  ────────────────────────
T0   💻 SYNC ① lock JSON schema (Appendix C) + cohort source list + eval slice
        │  (do this together, 30–45 min — nothing else starts until it's done)
        ▼                                          ▼
T1   💻 build math cohorts                    💻 build code cohorts + write the
        (GSM8K/MATH/synthetic),                  sandbox unit-test verifier
        write slice_cohorts.py                    (the one genuinely new piece)
        + generate synthetic via Claude           + adapt extract_signals.py
        ▼                                          ▼
T2   💻 SYNC ② quick cross-check: both extract_signals.py emit identical
        signal keys on a 5-task dry run (no GPU). Catches schema drift early.
        ▼                                          ▼
T3   ⚙️ BOX UP. Phase 1 signals             ⚙️ BOX UP. Phase 1 signals
        (~1.5h) → push *_signals.jsonl           (~1.5h) → push *_signals.jsonl
        ▼                                          ▼
T4   ⚙️ Phase 2: pre-eval → 5 GRPO          ⚙️ Phase 2: pre-eval → 5 GRPO
        runs in tmux → post-eval                  runs in tmux → post-eval
        (~4h, box stays up the whole time)        (~4h)
        push *_summary.jsonl, BOX DOWN            push *_summary.jsonl, BOX DOWN
        ▼                                          ▼
T5   💻 ───────────── SYNC ③ both summaries in results/ ─────────────
        whoever is free first builds analysis + dashboard (GPU off),
        consumes BOTH domains, produces scatter + Pareto plot
        ▼
T6   💻 writeup together
```

**Within a lane, the golden rule is "one box-up does everything."** Don't
boot the GPU for Phase 1, shut down, boot again for Phase 2 — that wastes
boot time and risk. Boot once at T3, run signals, roll straight into training
and eval at T4, then shut down at the end of T4. The box is up for one
continuous ~5.5h block per person.

**Inside Phase 2, run the 5 GRPO+eval jobs back-to-back in one tmux
session** (a simple bash `for` loop over cohort files). Detach, let it grind,
reattach to check. Don't babysit each run.

**Sync points are the only coupling:**
- **SYNC ① (T0, blocking):** schema + cohort sources + eval slice. If this is
  sloppy, everything downstream misaligns. Spend the full 45 min here.
- **SYNC ② (T2, cheap insurance):** 5-task dry run proving both tracks emit
  identical signal keys. Catches drift before you've spent any GPU money.
- **SYNC ③ (T5, rejoin):** both summaries landed → analysis can run.

**If one lane is faster** (math will be — code's sandbox verifier is extra
work): the fast lane starts the shared analysis script against its own
summary + a stub for the other domain, so it's ready the moment Lane B's
summary lands. No idle waiting.

## Execution logistics

- **GPU discipline.** Phases 1–2 only. Everything else runs on your laptop
  with the box **off**. If you're about to spend >5 min editing code with the
  GPU idle, shut it down. Batch GPU work: boot → pull → run the whole phase
  in `tmux` → push results → stop the box. Full runbook in
  `PRIME_INTELLECT.md`.
- **Budget.** We each have our own **$100** plan → run two A100s fully in
  parallel, no time-sharing. Each track costs ~$15. ~$85 headroom each goes
  to (in priority): seed repeats → more cohort *sources* (extra synthetic
  quality levels, another sibling dataset) → longer training. Past ~6–8
  cohorts you're limited by the regression's point count, not money.
- **Never lose work.** Train inside `tmux`; push `results/` to the repo
  after every phase; let huge checkpoints die with the instance (we only
  need the `lift` number, not the weights). `.gitignore` already blocks
  checkpoints.

---

## Definition of done

A one-pager with:
1. **The scatter** — predicted vs actual lift, both domains, points hugging
   the diagonal.
2. **The Pareto plot** — which metrics break the cost-quality frontier.
3. **The rationale** — for whichever candidate won the bake-off, why its
   mechanism is domain-independent (hence transfers math → code), and an
   honest note on which candidates' stories the data did *not* support.
4. **The honesty line** — "n cohorts per domain; this is a hypothesis, not a
   conclusion." Beats fake p-values.

---
---

# Appendix A — Candidate metric basket & rationale

This is a **bake-off**, not a ladder with a pre-chosen winner. We compute all
of these and let the cross-domain regression decide. Build order is
"sharp but safe": baselines + sampling headroom first (guaranteed
deliverable), gradient coherence as high-ceiling upside.

Ordered cheap → expensive so the Pareto plot has a real spread.

| Tier | Metric | Cost | Why it should predict lift |
|---|---|---|---|
| **Free** | `reasoning_steps`, `qlen`, `num_count` | 0 | Structural difficulty proxies. Expected weak — the "cheap floor" we beat. |
| **Cheap (1 fwd)** | `prompt_ppl` | 1 forward | Novel-but-not-alien tasks. Mid perplexity = headroom without being OOD. |
| **Med (N rollouts)** | `reward_var` | N gens | Baseline. GRPO gradient ∝ within-group reward std; zero var → zero gradient. The mechanistic floor any good metric must clear. |
| **Med (N rollouts)** | `intermediate_frac`, `self_consistency_gap` | N gens (reuse) | Outcome-uncertainty baselines. Goldilocks pass-rate / majority-vs-greedy gap. |
| **★ Med (N rollouts)** | **`sampling_headroom` = pass@N − pass@1** | N gens (reuse) | **Novel candidate, safe anchor.** RL converts *latent* capability (reachable when sampled lucky) into *reliable* capability. Sweet spot = low pass@1, high pass@N. Sharper framing than self-consistency gap. |
| **★ Med (N backward)** | **`gradient_coherence`** (dataset-level) | N backward passes (sketchable → cheaper) | **Novel candidate, high ceiling.** `‖mean task-gradient‖ / mean‖task-gradient‖` across the cohort. Measures whether tasks pull the model the *same* way — inter-task synergy that NO per-task average can see. Connects to influence functions / TracIn / RHO-loss. Random-projection sketching pushes it toward the cheap corner → a candidate to "break the frontier." |
| **○ Med (N rollouts + embed)** | `reward_separability` *(if time)* | N gens + 1 embed | Embed correct vs incorrect rollouts; measure linear separability. RL-as-BC can only reinforce what's representationally coherent. |
| **Expensive (grad probe)** | `learnability` *(stretch)* | 1 SGD step + eval | One-step RHO-loss: does training on this task drop held-out loss? Near-ground-truth but ~as costly as the RL itself — the frontier's expensive corner, useful as an oracle to compare cheap metrics against. |

★ = the two novel candidates we're betting on. ○ = backlog.

**What we report:** the empirical winner across both domains, *and* the
candidates whose first-principles story failed — negative results are part of
the contribution.

## Appendix A2 — Every low-hanging signal (log them all)

Principle: if a signal is free-or-cheap and *might* correlate, log it. They
cost almost nothing once the rollouts are in hand, and the ones that fail
become our "we understand correlation vs. causation" negative results. All of
these get written per-task in `extract_signals.py` and aggregated (mean + std
+ relevant fractions) to the cohort row.

**Tier 0 — Free, model-agnostic (no model, just the text)**
- `qlen_tokens`, `qlen_chars`, `qlen_words` — question length, 3 ways.
- `reasoning_steps` — count of `<<…>>` calc annotations (math) / reference
  solution line count (code).
- `num_count` — numbers in the question; `operator_count` — +−×÷ etc.
- `gold_answer_magnitude` — size of the target number (math).
- `gold_solution_len` — length of the reference solution.
- `type_token_ratio` — lexical diversity of the question.
- `question_entropy` — unigram entropy of the prompt text.
- `has_units` / `has_percent` / `has_fraction` — surface feature flags.

**Tier 1 — Cheap, 1 forward pass (no generation)**
- `prompt_ppl` — perplexity of the question.
- `gold_solution_ppl` — perplexity of the reference solution (teacher-forced).
- `gold_token_surprise` — Σ`(1 − p_model[gold_tok])`, the CE budget RL fights.
- `first_token_entropy` — model's uncertainty at the first answer token.

**Tier 2 — Medium, reuse the N rollouts you already generate**
- `reward_mean` (pass@1-ish), `reward_var`, `reward_std`.
- `pass@1`, `pass@k` for k∈{2,4,N}; `sampling_headroom` = pass@N − pass@1.
- `self_consistency_gap` — majority-vote acc − greedy acc.
- `intermediate_frac` — fraction of tasks with 0 < pass-rate < 1.
- `answer_entropy` — entropy over the *final answers* across rollouts
  (outcome diversity).
- `completion_len_mean` / `_var` — verbosity and its spread.
- `distinct_rollout_frac` — fraction of unique completions (path diversity,
  cheap string-dedup version).
- `format_rate` — fraction obeying the answer format.

**Tier 2.5 — Medium + one embedding pass (local sentence-transformer)**
- `cohort_diversity` — mean pairwise embedding distance within the cohort.
- `redundancy` — fraction of tasks with a near-duplicate neighbor.
- `pca_effective_rank` — participation ratio of the embedding matrix.
- `coverage` — mean similarity of cohort tasks to the eval set (the
  distribution-match control from the training-data section).
- `rollout_diversity` — mean pairwise distance between a task's completions
  (path diversity, embedding version).

**Tier 3 — Novel candidates (the ones we're betting on)**
- `gradient_coherence` — `‖mean task-grad‖ / mean‖task-grad‖` over the cohort.
- `reward_separability` — linear separability of correct vs. incorrect
  rollout embeddings.

Most teams stop at Tier 0–2. Tiers 2.5–3 are where we differentiate. Logging
Tier 0–2 is essentially free, so there's no reason not to — the regression
sorts out which ones matter.

# Appendix B — Idea backlog (only if time)

1. **Embedding diversity / PCA effective-rank** — dataset-level geometry.
   Cheap (one local sentence-transformer pass), model-agnostic. Anchors the
   *cheap* end of the Pareto plot opposite the behavioral metrics. Also gives
   a 2D PCA scatter that visually sells cohort separation. Caveat: diversity
   correlates with difficulty — may not separate at small n; report as a
   limitation.
2. **Gold-trajectory surprise** — see Appendix A; one forward pass, no
   generation.
3. **Synthetic cohort (Claude-generated)** — a data-quality probe: does the
   metric still predict lift on synthetic GSM8K-style problems? Uses
   Anthropic credits, not GPU.
4. **Negative result on purpose** — show raw token length *fails* to predict
   lift. Demonstrates we understand correlation vs causation.

### Compute-budget levers (pull only if we run short)

These are parked, not planned. Decide based on actual credit burn.

- **LoRA / PEFT for GRPO training** *(the real budget SAVER)*. Train only
  low-rank adapters instead of full weights → less memory, faster steps,
  cheaper. TRL supports it via a `peft` `LoraConfig` passed to `GRPOTrainer`.
  Trade-off: lift magnitude may differ from full fine-tuning, so if we go
  LoRA we must use it for **all** cohorts (keep cohort the only variable).
  Promote this first if credits get tight — it directly buys more cohorts or
  longer runs. Cost: *negative* (saves money).
- **Precision portability** *(science add-on, NOT a saver)*. Compute the
  signal basket with a cheap proxy model — 4-bit (`bitsandbytes`, one kwarg)
  or a smaller checkpoint (Qwen2.5-Math-0.5B) — but predict the lift of the
  real **bf16** GRPO run. If cheap-model signals still rank cohorts, we've
  pushed the metric down the *cost* axis without losing predictive power —
  the "break the frontier" result the brief asks for. Each metric becomes a
  curve (bf16 → 4-bit → 0.5B) on the Pareto plot, not a dot. Mechanistic bet:
  outcome signals (sampling headroom) survive quantization; gradient signals
  may degrade (quant noise corrupts gradient direction) — *which* survive is
  itself a finding. Cost: ~1 extra GPU-hr, reuses `signals.py` unchanged.
- **Do NOT** quantize or FP8 the *training* — at 1.5B it adds instability
  with no memory benefit (model is ~3 GB, A100 has ~75 GB free). bf16 stays.

# Appendix C — Shared cohort-summary schema

Both tracks emit this, one row per cohort. The analysis concatenates them.

```json
{
  "domain": "math",            // or "code"
  "cohort": "medium_pass",
  "n_tasks": 256,
  "signals": {
    "sampling_headroom": 0.34,
    "gradient_coherence": 0.41,
    "self_consistency_gap": 0.18,
    "reward_var": 0.21,
    "prompt_ppl": 7.9,
    "reward_mean": 0.45,
    "intermediate_frac": 0.62
  },
  "acc_before": 0.41,
  "acc_after": 0.52,
  "lift": 0.11
}
```

Rules: signal keys identical across domains; `lift = acc_after − acc_before`;
one file per domain (`results/math_summary.jsonl`, `results/code_summary.jsonl`).

# Appendix D — Risks

- **Lift is noisy at 200 steps.** Mitigate with seed repeats; report mean.
- **Reward hacking** on format tokens — weight correctness ≫ format, eyeball
  final completions.
- **vLLM OOM** on a single GPU — fall back to HF `generate`; don't burn hours
  debugging vLLM.
- **Small n** — pre-commit to Spearman + LOO, never R².
- **Schema drift between tracks** — the single biggest coordination risk.
  Lock Appendix C before writing track code.
