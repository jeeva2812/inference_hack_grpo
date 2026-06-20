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
pass_at_1         : greedy-equivalent — fraction correct on first rollout.
pass_at_n         : fraction of tasks where ANY rollout was correct (pass@N).
sampling_headroom : pass_at_n − pass_at_1 — latent capability RL can unlock.
                    High headroom = model knows the answer but can't reliably
                    produce it; sweet spot for GRPO to convert luck into habit.
self_consistency_gap : majority-vote accuracy − pass_at_1 (greedy proxy).
                    Measures how much sampling consensus helps over greedy.
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

from math_common import MODEL_ID, build_prompt, extract_pred, has_format, is_correct

COHORT_DIR = Path("cohorts")

N_ROLLOUTS = 5
ROLLOUT_TEMP = 0.9
MAX_NEW_TOKENS = 512    # answers appear in the first ~200 tokens; 512 is safe headroom

# Scoring helpers (extract_pred / is_correct / has_format / build_prompt) are
# imported from math_common — same code path as eval + training.


@torch.no_grad()
def batch_perplexity(prompts: list[str], model, tokenizer) -> list[float]:
    """Per-sequence NLL under the model for a batch of prompt strings.

    Pads left (matching generation padding_side) so all sequences end at the
    same position. Uses per-token loss masking so padding doesn't inflate ppl.
    """
    enc = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    input_ids = enc.input_ids
    attention_mask = enc.attention_mask

    labels = input_ids.clone()
    labels[attention_mask == 0] = -100   # ignore padding in loss

    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    # out.loss is the mean over all non-masked tokens in the batch — we need
    # per-sequence NLL, so re-compute with reduction="none" via log-softmax.
    logits = out.logits[:, :-1].float()           # (B, L-1, V)
    tgt    = labels[:, 1:].clone()                # (B, L-1)
    mask   = (tgt != -100)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    nll = -log_probs.gather(2, tgt.clamp(min=0).unsqueeze(2)).squeeze(2)  # (B, L-1)
    per_seq_nll = (nll * mask).sum(1) / mask.sum(1).clamp(min=1)
    return [math.exp(v.item()) for v in per_seq_nll]


@torch.no_grad()
def rollout_signals_batch(rows: list, model, tokenizer, prompts: list[str]) -> list:
    """Generate N_ROLLOUTS completions for a *batch* of tasks in one call.

    With batch=1 the A100 sits idle (decode is latency-bound), so we pack many
    prompts per generate. Requires tokenizer.padding_side='left' so the prompt
    length is uniform and out[:, prompt_len:] is the clean completion slice.
    Output ordering is batch-major: prompt0's N returns, then prompt1's, ...
    -> reshape to (B, N, gen_len).
    """
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

    # Batch-decode all (B * N) sequences at once — eliminates the per-token
    # Python loop overhead of calling decode() B*N times individually.
    B, N, L = gen.shape
    flat = gen.reshape(B * N, L)
    # Replace pad tokens with 0 before decode so skip_special_tokens works cleanly.
    flat_clean = flat.clone()
    flat_clean[flat_clean == tokenizer.pad_token_id] = tokenizer.eos_token_id
    texts = tokenizer.batch_decode(flat_clean, skip_special_tokens=True)  # list[B*N]
    # Compute actual lengths (tokens before first pad) for each sequence.
    pad_mask = (flat != tokenizer.pad_token_id)
    raw_lengths = pad_mask.sum(dim=1).tolist()   # list[B*N]

    out = []
    for i, row in enumerate(rows):
        rewards, fmt_hits, lengths = [], 0, []
        for n in range(N_ROLLOUTS):
            text = texts[i * N_ROLLOUTS + n]
            pred = extract_pred(text)
            rewards.append(1.0 if is_correct(pred, row["gold"]) else 0.0)
            fmt_hits += int(has_format(text))
            lengths.append(raw_lengths[i * N_ROLLOUTS + n])

        mean_r   = sum(rewards) / len(rewards)
        var_r    = sum((r - mean_r) ** 2 for r in rewards) / len(rewards)
        pass_at1 = rewards[0]                          # first rollout = greedy proxy
        pass_atn = float(any(r == 1.0 for r in rewards))
        majority = float(sum(rewards) / len(rewards) >= 0.5)
        out.append({
            "reward_mean":           round(mean_r, 4),
            "reward_var":            round(var_r,  6),
            "pass_at_1":             pass_at1,
            "pass_at_n":             pass_atn,
            "sampling_headroom":     round(pass_atn - pass_at1, 4),
            "self_consistency_gap":  round(majority - pass_at1, 4),
            "format_rate":           round(fmt_hits / N_ROLLOUTS, 4),
            "mean_length":           round(sum(lengths) / len(lengths), 1),
            "rewards_raw":           rewards,
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

        prompts = [build_prompt(row["question"], tokenizer) for row in batch]
        ppls = batch_perplexity(prompts, model, tokenizer)
        sigs = rollout_signals_batch(batch, model, tokenizer, prompts)

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
        "reward_var_mean":          round(sum(s["reward_var"]  for s in sigs) / len(sigs), 6),
        "format_rate_mean":         round(sum(s["format_rate"] for s in sigs) / len(sigs), 4),
        "pass_at_1_mean":           round(sum(s["pass_at_1"]  for s in sigs) / len(sigs), 4),
        "pass_at_n_mean":           round(sum(s["pass_at_n"]  for s in sigs) / len(sigs), 4),
        "sampling_headroom_mean":   round(sum(s["sampling_headroom"] for s in sigs) / len(sigs), 4),
        "self_consistency_gap_mean":round(sum(s["self_consistency_gap"] for s in sigs) / len(sigs), 4),
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
