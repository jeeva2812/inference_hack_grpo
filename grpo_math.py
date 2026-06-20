"""
grpo_math.py — GRPO training, MATH track: Qwen2.5-Math-1.5B on GSM8K.
(50-step smoke default; bump max_steps for real cohort runs.)

Experiment tracking: set env vars to stream metrics to Weights & Biases
(survives the box dying, lets the team watch from one dashboard):
    export REPORT_TO=wandb
    export WANDB_PROJECT=grpo-cohorts
    export RUN_NAME=math_medium_pass     # set per cohort run
Defaults to "none" (terminal only) so it never breaks if W&B isn't set up.
"""

import argparse
import json
import os
import re
from pathlib import Path

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from trl import GRPOConfig, GRPOTrainer

# Shared with eval + signals so the training reward scores correctness exactly
# as lift is later measured (single source of truth in math_common).
from math_common import SYSTEM_PROMPT, extract_gold, extract_pred, is_correct

MODEL_ID = "Qwen/Qwen2.5-Math-1.5B"

# Tracking backend: "none" (default) or "wandb". Overridable without code edits.
REPORT_TO = os.environ.get("REPORT_TO", "none")
RUN_NAME = os.environ.get("RUN_NAME", "qwen25-math-1_5b-grpo-smoke")


def correctness_reward(completions, answer, **kwargs):
    # `answer` carries the BARE gold number (build_dataset normalizes it), so the
    # reward scores correctness exactly like extract_signals_math (is_correct vs
    # row["gold"]). Some cohorts (e.g. synthetic_good/Orca) have no '#### N' line
    # in their answer text, so parsing it here would silently zero their reward.
    rewards = []
    for comp, gold in zip(completions, answer):
        text = comp[0]["content"] if isinstance(comp, list) else comp
        rewards.append(1.0 if is_correct(extract_pred(text), gold) else 0.0)
    return rewards


def format_reward(completions, **kwargs):
    # Deliberately looser than signals' has_format: any <answer>...</answer>
    # block (DOTALL) earns shaping reward, even before the content is numeric.
    pat = re.compile(r"<answer>.*?</answer>", re.DOTALL)
    out = []
    for comp in completions:
        text = comp[0]["content"] if isinstance(comp, list) else comp
        out.append(0.2 if pat.search(text) else 0.0)
    return out


def build_dataset(cohort: str | None):
    """Load the training data for one run.

    Phase 2: pass --cohort <name> to train on cohorts/<name>.jsonl. Each
    cohort is the SAME size (256) — only data quality differs — so cohort
    is the only variable behind lift.

    The reward is scored against the BARE gold number: cohort rows already
    carry a clean `gold` field, while vanilla GSM8K rows only have the full
    'answer' text, so we derive gold via extract_gold there. (synthetic_good
    has no '#### N' in its answer text — relying on that would zero its reward.)

    Omit --cohort to fall back to vanilla GSM8K-train (smoke testing only).
    """
    if cohort:
        path = Path("cohorts") / f"{cohort}.jsonl"
        rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
        ds = Dataset.from_list(rows)
    else:
        ds = load_dataset("openai/gsm8k", "main", split="train")

    def fmt(ex):
        gold = ex["gold"] if ex.get("gold") else extract_gold(ex["answer"])
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": ex["question"]},
            ],
            "answer": gold,
        }

    return ds.map(fmt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default=None,
                    help="cohort name in cohorts/<name>.jsonl; omit for vanilla GSM8K (smoke)")
    ap.add_argument("--max_steps", type=int, default=50,
                    help="bump to ~200-300 for real Phase-2 cohort runs")
    ap.add_argument("--seed", type=int, default=0,
                    help="vary across repeat runs of the same cohort to de-noise lift")
    args = ap.parse_args()

    # Run name: env RUN_NAME wins (per-team dashboard), else derive from cohort+seed.
    # outputs/<run_name> is exactly what eval_math.py --model points at afterwards.
    default_run = (f"math_{args.cohort}_seed{args.seed}"
                   if args.cohort else "qwen25-math-1_5b-grpo-smoke")
    run_name = os.environ.get("RUN_NAME", default_run)
    print(f"=== GRPO run: {run_name} "
          f"(cohort={args.cohort or 'gsm8k-train'}, steps={args.max_steps}, seed={args.seed}) ===")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
    )

    train_ds = build_dataset(args.cohort)

    # NOTE: config is IDENTICAL across cohorts on purpose — only the data
    # (and --seed for repeats) may change, so cohort is the only variable
    # explaining differences in lift.
    config = GRPOConfig(
        output_dir=f"outputs/{run_name}",
        run_name=run_name,
        seed=args.seed,
        learning_rate=1e-6,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        num_generations=4,
        max_completion_length=512,
        max_steps=args.max_steps,
        logging_steps=1,
        save_steps=args.max_steps,
        bf16=True,
        gradient_checkpointing=True,
        report_to=REPORT_TO,    # "none" or "wandb" via env var
        # vLLM rollout (toggle off if it fails on your node)
        use_vllm=False,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[correctness_reward, format_reward],
        args=config,
        train_dataset=train_ds,
    )
    trainer.train()


if __name__ == "__main__":
    main()
