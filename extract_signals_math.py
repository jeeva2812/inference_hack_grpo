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
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from math_common import build_prompt, extract_pred, has_format, is_correct

MODEL_ID = "Qwen/Qwen2.5-Math-1.5B-Instruct"  # see grpo_math.py: base model unusable zero-shot
COHORT_DIR = Path("cohorts")

N_ROLLOUTS = 5
ROLLOUT_TEMP = 0.9
MAX_NEW_TOKENS = 1024   # Qwen-Math CoT is long; 512 truncated every rollout

# Scoring helpers (extract_pred / is_correct / has_format / build_prompt) are
# imported from math_common — same code path as eval + training.


@torch.no_grad()
def prompt_perplexity(prompt_ids: torch.Tensor, model, tokenizer) -> float:
    """NLL-per-token of the prompt under the model (no generation)."""
    ids = prompt_ids.unsqueeze(0).to(model.device)
    mask = (ids != tokenizer.pad_token_id).long()
    labels = ids.clone()
    out = model(input_ids=ids, attention_mask=mask, labels=labels)
    return math.exp(out.loss.item())


@torch.no_grad()
def rollout_signals_batch(rows: list, model, tokenizer) -> list:
    """Generate N_ROLLOUTS completions for a *batch* of tasks in one call.

    With batch=1 the A100 sits idle (decode is latency-bound), so we pack many
    prompts per generate. Requires tokenizer.padding_side='left' so the prompt
    length is uniform and out[:, prompt_len:] is the clean completion slice.
    Output ordering is batch-major: prompt0's N returns, then prompt1's, ...
    -> reshape to (B, N, gen_len).
    """
    prompts = [build_prompt(r["question"], tokenizer) for r in rows]
    enc = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    prompt_len = enc.input_ids.shape[1]

    outputs = model.generate(
        **enc,
        do_sample=True,
        temperature=ROLLOUT_TEMP,
        max_new_tokens=MAX_NEW_TOKENS,
        num_return_sequences=N_ROLLOUTS,
        pad_token_id=tokenizer.pad_token_id,
    )
    gen = outputs[:, prompt_len:].view(len(rows), N_ROLLOUTS, -1)

    out = []
    for i, row in enumerate(rows):
        rewards, fmt_hits, lengths = [], 0, []
        for n in range(N_ROLLOUTS):
            seq = gen[i, n]
            real = seq[seq != tokenizer.pad_token_id]   # strip right-pad on finished seqs
            text = tokenizer.decode(real, skip_special_tokens=True)
            pred = extract_pred(text)
            rewards.append(1.0 if is_correct(pred, row["gold"]) else 0.0)
            fmt_hits += int(has_format(text))
            lengths.append(int(real.numel()))

        mean_r = sum(rewards) / len(rewards)
        var_r  = sum((r - mean_r) ** 2 for r in rewards) / len(rewards)
        out.append({
            "reward_mean":  round(mean_r, 4),
            "reward_var":   round(var_r,  6),
            "format_rate":  round(fmt_hits / N_ROLLOUTS, 4),
            "mean_length":  round(sum(lengths) / len(lengths), 1),
            "rewards_raw":  rewards,
        })
    return out


# ── main loop ────────────────────────────────────────────────────────────────

def process_cohort(path: Path, model, tokenizer, max_tasks: int | None, batch_size: int):
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    if max_tasks:
        rows = rows[:max_tasks]

    out_path = path.with_name(path.stem + "_signals.jsonl")
    results = []

    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]

        # perplexity is a single cheap forward per task — keep it per-row so the
        # loss-masking stays trivially correct; generation is what we batch.
        ppls = []
        for row in batch:
            prompt_ids = tokenizer(
                build_prompt(row["question"], tokenizer), return_tensors="pt"
            ).input_ids[0]
            ppls.append(prompt_perplexity(prompt_ids, model, tokenizer))

        sigs = rollout_signals_batch(batch, model, tokenizer)

        for row, ppl, sig in zip(batch, ppls, sigs):
            results.append({**row, "signals": {"prompt_ppl": round(ppl, 4), **sig}})

        done = start + len(batch)
        last = results[-1]["signals"]
        print(f"  [{done}/{len(rows)}] ppl={last['prompt_ppl']:.1f}  "
              f"r_mean={last['reward_mean']}  r_var={last['reward_var']}", flush=True)

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
    parser.add_argument("--batch_size", type=int, default=16,
                        help="prompts per generate call; lower if you hit OOM")
    args = parser.parse_args()

    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    # Qwen uses EOS as PAD by default — give it a distinct pad token so
    # attention masks are computed correctly during batched generation/scoring.
    if tokenizer.pad_token_id is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    tokenizer.padding_side = "left"   # uniform prompt len for batched generation
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
        summary = process_cohort(path, model, tokenizer, args.max_tasks, args.batch_size)
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
