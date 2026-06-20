"""
Extract cheap task-level signals from each cohort file using the base model.

Signals computed per task
--------------------------
prompt_ppl        : perplexity of the question tokens under the base model.
                    Low ppl = in-distribution, familiar. High ppl = unusual.
reward_mean       : mean correctness reward across N high-temp rollouts.
reward_var        : variance of correctness reward — key GRPO learnability signal.
                    Near-0 var (all right or all wrong) = no gradient signal.
                    High var (model is sometimes right) = good training material.
format_rate       : fraction of rollouts that produced <answer>...</answer>.
mean_length       : mean completion token length across rollouts.

Output
------
cohorts/<name>_signals.jsonl — original row + signals dict appended.
cohorts/summary.jsonl        — one row per cohort with aggregate stats.

Usage
-----
python extract_signals_math.py                        # all files in cohorts/
python extract_signals_math.py --cohort short_easy    # single cohort by name
python extract_signals_math.py --max_tasks 50         # cap for quick debugging
"""

import argparse
import json
import math
import re
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen2.5-Math-1.5B"
COHORT_DIR = Path("cohorts")

N_ROLLOUTS = 5
ROLLOUT_TEMP = 0.9
MAX_NEW_TOKENS = 1024   # Qwen-Math CoT is long; 512 truncated every rollout

SYSTEM_PROMPT = (
    "You are a math tutor. Solve the problem step by step. "
    "Put your final numeric answer inside <answer>...</answer>."
)

GOLD_RE = re.compile(r"####\s*(-?[\d,\.]+)")
PRED_RE = re.compile(r"<answer>\s*(-?[\d,\.]+)\s*</answer>")
NUM_RE  = re.compile(r"-?\d+\.?\d*")


# ── reward helpers ──────────────────────────────────────────────────────────

def extract_gold(answer_field: str) -> str:
    m = GOLD_RE.search(answer_field)
    return m.group(1).replace(",", "").strip() if m else ""


def extract_pred(text: str) -> str:
    m = PRED_RE.search(text)
    if m:
        return m.group(1).replace(",", "").strip()
    nums = NUM_RE.findall(text)
    return nums[-1] if nums else ""


def is_correct(pred: str, gold: str) -> bool:
    try:
        return abs(float(pred) - float(gold)) < 1e-4
    except ValueError:
        return False


def has_format(text: str) -> bool:
    return bool(PRED_RE.search(text))


# ── model helpers ────────────────────────────────────────────────────────────

def build_prompt(question: str, tokenizer) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


@torch.no_grad()
def prompt_perplexity(prompt_ids: torch.Tensor, model, tokenizer) -> float:
    """NLL-per-token of the prompt under the model (no generation)."""
    ids = prompt_ids.unsqueeze(0).to(model.device)
    mask = (ids != tokenizer.pad_token_id).long()
    labels = ids.clone()
    out = model(input_ids=ids, attention_mask=mask, labels=labels)
    return math.exp(out.loss.item())


@torch.no_grad()
def rollout_signals(prompt_ids: torch.Tensor, gold: str, model, tokenizer) -> dict:
    """Generate N_ROLLOUTS completions and compute reward stats."""
    prompt_len = prompt_ids.shape[0]
    ids = prompt_ids.unsqueeze(0).to(model.device)
    mask = (ids != tokenizer.pad_token_id).long()

    outputs = model.generate(
        ids,
        attention_mask=mask,
        do_sample=True,
        temperature=ROLLOUT_TEMP,
        max_new_tokens=MAX_NEW_TOKENS,
        num_return_sequences=N_ROLLOUTS,
        pad_token_id=tokenizer.pad_token_id,
    )

    rewards = []
    fmt_hits = 0
    lengths = []

    for seq in outputs:
        completion_ids = seq[prompt_len:]
        text = tokenizer.decode(completion_ids, skip_special_tokens=True)
        pred = extract_pred(text)
        rewards.append(1.0 if is_correct(pred, gold) else 0.0)
        fmt_hits += int(has_format(text))
        lengths.append(len(completion_ids))

    mean_r = sum(rewards) / len(rewards)
    var_r  = sum((r - mean_r) ** 2 for r in rewards) / len(rewards)

    return {
        "reward_mean":  round(mean_r, 4),
        "reward_var":   round(var_r,  6),
        "format_rate":  round(fmt_hits / N_ROLLOUTS, 4),
        "mean_length":  round(sum(lengths) / len(lengths), 1),
        "rewards_raw":  rewards,
    }


# ── main loop ────────────────────────────────────────────────────────────────

def process_cohort(path: Path, model, tokenizer, max_tasks: int | None):
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    if max_tasks:
        rows = rows[:max_tasks]

    out_path = path.with_name(path.stem + "_signals.jsonl")
    results = []

    for i, row in enumerate(rows):
        print(f"  [{i+1}/{len(rows)}] {row['id']}", end=" ", flush=True)

        prompt_str = build_prompt(row["question"], tokenizer)
        prompt_ids = tokenizer(prompt_str, return_tensors="pt").input_ids[0]

        ppl = prompt_perplexity(prompt_ids, model, tokenizer)
        sig = rollout_signals(prompt_ids, row["gold"], model, tokenizer)

        result = {**row, "signals": {"prompt_ppl": round(ppl, 4), **sig}}
        results.append(result)
        print(f"ppl={ppl:.1f}  r_mean={sig['reward_mean']}  r_var={sig['reward_var']}")

    with out_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"  -> {out_path}")

    # aggregate for summary
    sigs = [r["signals"] for r in results]
    return {
        "cohort":            path.stem,
        "n_tasks":           len(results),
        "ppl_mean":          round(sum(s["prompt_ppl"]  for s in sigs) / len(sigs), 4),
        "reward_mean_mean":  round(sum(s["reward_mean"] for s in sigs) / len(sigs), 4),
        "reward_var_mean":   round(sum(s["reward_var"]  for s in sigs) / len(sigs), 6),
        "format_rate_mean":  round(sum(s["format_rate"] for s in sigs) / len(sigs), 4),
        # fraction of tasks where the model is sometimes right but not always
        # (0 < reward_mean < 1) — the "intermediate difficulty" pool
        "intermediate_frac": round(
            sum(1 for s in sigs if 0 < s["reward_mean"] < 1) / len(sigs), 4
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", default=None, help="Name of a single cohort to process")
    parser.add_argument("--max_tasks", type=int, default=None)
    args = parser.parse_args()

    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    # Qwen uses EOS as PAD by default — give it a distinct pad token so
    # attention masks are computed correctly during batched generation/scoring.
    if tokenizer.pad_token_id is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="auto"
    )
    model.resize_token_embeddings(len(tokenizer))
    model.eval()

    if args.cohort:
        files = [COHORT_DIR / f"{args.cohort}.jsonl"]
    else:
        files = sorted(COHORT_DIR.glob("*.jsonl"))
        # Skip: already-processed signal files; this script's own aggregate
        # output (summary.jsonl, which lacks task fields); and eval_slice —
        # the FIXED benchmark for measuring lift, not a training cohort.
        # Running rollouts on it wastes GPU and would pollute the predictor set.
        SKIP_STEMS = {"summary", "eval_slice"}
        files = [f for f in files
                 if "_signals" not in f.name and f.stem not in SKIP_STEMS]

    summaries = []
    for path in files:
        print(f"\n=== {path.name} ===")
        summary = process_cohort(path, model, tokenizer, args.max_tasks)
        summaries.append(summary)
        print(f"  cohort summary: {summary}")

    summary_path = COHORT_DIR / "summary.jsonl"
    with summary_path.open("w", encoding="utf-8") as f:
        for s in summaries:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"\nSummary -> {summary_path}")

    # Also write the per-cohort aggregate to the cross-track contract path
    # (GAMEPLAN Phase 1 output). This is the predictor side of the final
    # signals-vs-lift regression; Phase 2 appends lift to the same rows.
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    contract_path = results_dir / "math_signals.jsonl"
    with contract_path.open("w", encoding="utf-8") as f:
        for s in summaries:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"Contract -> {contract_path}")


if __name__ == "__main__":
    main()
