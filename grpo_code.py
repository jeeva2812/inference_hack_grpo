"""
grpo_code.py — GRPO training, CODE track: Qwen2.5-1.5B-Instruct on a cohort.
Code-track analog of grpo_math.py.

Trains on ONE cohort file (cohorts_code/<cohort>.jsonl) so cohort is the only
variable. Reward = fraction of the cohort task's unit tests the model's code
passes (continuous), computed by test_executor in a sandboxed subprocess. The
prompt is shared with eval_code.py so signals/training/eval all see the same
input distribution.

Run one cohort:
    RUN_NAME=code_benchmark_slice python grpo_code.py --cohort benchmark_slice

Loop all cohorts (Phase 2, in tmux):
    for c in benchmark_slice harder_sibling synthetic_good synthetic_degraded random_control; do
        RUN_NAME=code_$c python grpo_code.py --cohort $c
        python eval_code.py --model outputs/code_$c --label $c
    done

Knobs via env (so the runbook never edits code):
    REPORT_TO=wandb WANDB_PROJECT=grpo-cohorts   # experiment tracking
    MAX_STEPS=250 NUM_GENERATIONS=8              # game-plan Phase-2 values
    MAX_COMPLETION_LENGTH=1024                   # generation budget
"""

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer

from test_executor import compute_test_reward
from eval_code import SYSTEM_PROMPT, build_user_content  # shared prompt = same distribution

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
COHORT_DIR = Path("cohorts_code")

REPORT_TO = os.environ.get("REPORT_TO", "none")
# Phase-2 defaults follow the game plan (num_generations=8, 200-300 steps);
# override down for a quick smoke test, e.g. MAX_STEPS=20.
MAX_STEPS = int(os.environ.get("MAX_STEPS", "250"))
NUM_GENERATIONS = int(os.environ.get("NUM_GENERATIONS", "8"))
# Generation/context budget. TRL 1.6.0's GRPOConfig only exposes the completion
# side (no max_prompt_length); prompts are kept full.
MAX_COMPLETION_LENGTH = int(os.environ.get("MAX_COMPLETION_LENGTH", "1024"))


def correctness_reward(completions, test_list, test_setup_code=None, **kwargs):
    """Reward = fraction of the task's unit tests passed (continuous)."""
    return compute_test_reward(
        completions, test_list, setup_code=test_setup_code, timeout=6
    )


def code_format_reward(completions, **kwargs):
    """Small bonus for emitting a fenced code block (kept << correctness)."""
    import re
    pat = re.compile(r"```(?:python)?\s*\n.*?\n```", re.DOTALL)
    out = []
    for comp in completions:
        text = comp[0]["content"] if isinstance(comp, list) else comp
        out.append(0.1 if pat.search(text) else 0.0)
    return out


def build_dataset(cohort: str) -> Dataset:
    path = COHORT_DIR / f"{cohort}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run `python slice_cohorts_code.py` first."
        )
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    ds = Dataset.from_list(rows)

    def fmt(ex):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_content(ex["text"], ex["test_list"])},
            ],
            "test_list": ex["test_list"],
            "test_setup_code": ex.get("test_setup_code", "") or "",
        }

    # keep only what the trainer + reward funcs need (drops reference, etc.)
    return ds.map(fmt, remove_columns=[c for c in ds.column_names
                                       if c not in ("test_list", "test_setup_code")])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default=os.environ.get("COHORT", "benchmark_slice"),
                    help="cohort file stem under cohorts_code/")
    args = ap.parse_args()

    run_name = os.environ.get("RUN_NAME", f"code_{args.cohort}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="auto"
    )

    train_ds = build_dataset(args.cohort)
    print(f"cohort={args.cohort}  n_tasks={len(train_ds)}  "
          f"steps={MAX_STEPS}  num_generations={NUM_GENERATIONS}  "
          f"completion_len={MAX_COMPLETION_LENGTH}")

    config = GRPOConfig(
        output_dir=f"outputs/{run_name}",
        run_name=run_name,
        learning_rate=1e-6,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,        # 4*2=8 divisible by num_generations=8
        num_generations=NUM_GENERATIONS,
        max_completion_length=MAX_COMPLETION_LENGTH,
        max_steps=MAX_STEPS,
        logging_steps=1,
        save_steps=MAX_STEPS,                 # one checkpoint at the end -> eval_code
        bf16=True,
        gradient_checkpointing=True,
        report_to=REPORT_TO,
        use_vllm=False,                       # flip on once a node has headroom
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[correctness_reward, code_format_reward],
        args=config,
        train_dataset=train_ds,
    )
    trainer.train()
    trainer.save_model(config.output_dir)
    print(f"saved -> {config.output_dir}")


if __name__ == "__main__":
    main()
