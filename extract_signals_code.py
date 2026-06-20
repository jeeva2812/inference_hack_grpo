"""
extract_signals_code.py — Phase 1 signals for the CODE track, via the SHARED
signals.py contract. Code-track analog of extract_signals_math.py, but it emits
the full candidate basket (hero `sampling_headroom` + baselines) in the
Appendix-C schema instead of ad-hoc keys, so plots.py / the regression can
consume math and code identically.

For each task we reuse ONE set of N rollouts to get every Tier-0..2 signal:
  - Tier 0 (free, text): qlen, num_count, type_token_ratio, ...   (signals.text_signals)
  - Tier 1 (1 fwd):      prompt_ppl
  - Tier 2 (N rollouts): reward_mean/var, pass@1, pass@n,
                         sampling_headroom, answer_entropy, intermediate_frac,
                         self_consistency_gap, completion length/diversity

Cross-domain validity: reward = fraction of unit tests passed (continuous). To
reuse signals.py's outcome metrics we set each rollout's "answer" to PASS/FAIL
(did all tests pass?) with gold="PASS", so self_consistency_gap = (sampled
majority passes) - (greedy passes) — the exact analog of the math track's
majority-vs-greedy gap. Reward-based signals depend only on the reward, never on
how it was produced, which is what makes the math<->code comparison valid.

Output:
  cohorts_code/<name>_signals.jsonl  — per-task detail
  results/code_signals.jsonl         — one cohort_summary row per cohort
Run `python merge_summary.py --domain code` afterwards to attach lift from
results/code_eval.jsonl -> results/code_summary.jsonl (what plots.py reads).

Usage:
  python extract_signals_code.py                       # all cohorts_code/*.jsonl
  python extract_signals_code.py --cohort benchmark_slice
  python extract_signals_code.py --max_tasks 20 --n_rollouts 4   # quick smoke
"""

import argparse
import json
import math
from pathlib import Path

from signals import Task, cohort_summary
from test_executor import extract_code_block, run_tests
from eval_code import build_prompt          # shared prompt = same input distribution

COHORT_DIR = Path("cohorts_code")
RESULTS_DIR = Path("results")
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

N_ROLLOUTS = 6
ROLLOUT_TEMP = 0.9
MAX_NEW_TOKENS = 1024


# ── pure signal construction (offline-testable, no torch) ────────────────────

def make_task(prompt_text, codes, rewards, greedy_pass, ppl) -> Task:
    """Assemble a signals.Task from one task's rollouts.

    rewards: continuous fraction-of-tests-passed per rollout.
    answers are PASS/FAIL so self_consistency_gap = majority-pass vs greedy-pass.
    """
    answers = ["PASS" if r >= 1.0 else "FAIL" for r in rewards]
    return Task(
        prompt=prompt_text, completions=codes, rewards=rewards,
        answers=answers, gold="PASS", greedy_correct=bool(greedy_pass),
        prompt_ppl=ppl,
    )


def score_code(code, row, timeout=6):
    rate, _ = run_tests(code, row["test_list"],
                        setup_code=row.get("test_setup_code", "") or "",
                        entry_point=row.get("entry_point"), timeout=timeout)
    return rate


# ── model-backed extraction (torch imported lazily) ──────────────────────────

def _prompt_perplexity(prompt_ids, model, tokenizer):
    import torch
    with torch.no_grad():
        ids = prompt_ids.unsqueeze(0).to(model.device)
        mask = (ids != tokenizer.pad_token_id).long()
        out = model(input_ids=ids, attention_mask=mask, labels=ids.clone())
    return math.exp(out.loss.item())


def _generate(prompt_ids, model, tokenizer, n, temp, greedy=False):
    import torch
    with torch.no_grad():
        ids = prompt_ids.unsqueeze(0).to(model.device)
        mask = (ids != tokenizer.pad_token_id).long()
        kw = dict(attention_mask=mask, max_new_tokens=MAX_NEW_TOKENS,
                  pad_token_id=tokenizer.pad_token_id)
        if greedy:
            out = model.generate(ids, do_sample=False, **kw)
        else:
            out = model.generate(ids, do_sample=True, temperature=temp,
                                 num_return_sequences=n, **kw)
    plen = prompt_ids.shape[0]
    return [tokenizer.decode(seq[plen:], skip_special_tokens=True) for seq in out]


def process_cohort(path, model, tokenizer, max_tasks, n_rollouts):
    rows = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
    if max_tasks:
        rows = rows[:max_tasks]

    tasks, detail = [], []
    for i, row in enumerate(rows):
        prompt_str = build_prompt(row["text"], row["test_list"], tokenizer)
        prompt_ids = tokenizer(prompt_str, return_tensors="pt").input_ids[0]

        ppl = _prompt_perplexity(prompt_ids, model, tokenizer)
        samples = _generate(prompt_ids, model, tokenizer, n_rollouts, ROLLOUT_TEMP)
        greedy = _generate(prompt_ids, model, tokenizer, 1, 0.0, greedy=True)[0]

        codes = [extract_code_block(s) for s in samples]
        rewards = [score_code(c, row) for c in codes]
        greedy_pass = score_code(extract_code_block(greedy), row) >= 1.0

        task = make_task(row["text"], codes, rewards, greedy_pass, ppl)
        tasks.append(task)
        detail.append({"id": row["id"], "rewards": rewards,
                       "greedy_pass": greedy_pass, "prompt_ppl": round(ppl, 4)})
        print(f"  [{i+1}/{len(rows)}] {row['id']}  ppl={ppl:.1f} "
              f"r_mean={sum(rewards)/len(rewards):.2f} greedy={'P' if greedy_pass else 'F'}")

    out_path = path.with_name(path.stem + "_signals.jsonl")
    with out_path.open("w", encoding="utf-8") as f:
        for d in detail:
            f.write(json.dumps(d) + "\n")
    print(f"  -> {out_path}")

    return cohort_summary("code", path.stem, tasks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default=None)
    ap.add_argument("--max_tasks", type=int, default=None)
    ap.add_argument("--n_rollouts", type=int, default=N_ROLLOUTS)
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    if args.cohort:
        files = [COHORT_DIR / f"{args.cohort}.jsonl"]
    else:
        files = sorted(f for f in COHORT_DIR.glob("*.jsonl")
                       if "_signals" not in f.name and f.stem != "eval_slice")

    summaries = []
    for path in files:
        print(f"\n=== {path.name} ===")
        summaries.append(process_cohort(path, model, tokenizer,
                                        args.max_tasks, args.n_rollouts))

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "code_signals.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for s in summaries:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"\nSignals -> {out}  (now run: python merge_summary.py --domain code)")


if __name__ == "__main__":
    main()
