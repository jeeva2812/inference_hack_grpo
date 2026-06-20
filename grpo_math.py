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

import os
import re
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from trl import GRPOConfig, GRPOTrainer

MODEL_ID = "Qwen/Qwen2.5-Math-1.5B"

# Tracking backend: "none" (default) or "wandb". Overridable without code edits.
REPORT_TO = os.environ.get("REPORT_TO", "none")
RUN_NAME = os.environ.get("RUN_NAME", "qwen25-math-1_5b-grpo-smoke")

SYSTEM_PROMPT = (
    "You are a math tutor. Solve the problem step by step. "
    "Put your final numeric answer inside <answer>...</answer>."
)


def extract_gold(answer_field: str) -> str:
    # GSM8K answers end with '#### <number>'
    m = re.search(r"####\s*(-?[\d,\.]+)", answer_field)
    return m.group(1).replace(",", "").strip() if m else ""


def extract_pred(text: str) -> str:
    m = re.search(r"<answer>\s*(-?[\d,\.]+)\s*</answer>", text)
    if m:
        return m.group(1).replace(",", "").strip()
    # fallback: last number in the text
    nums = re.findall(r"-?\d+\.?\d*", text)
    return nums[-1] if nums else ""


def to_float(s: str):
    try:
        return float(s)
    except ValueError:
        return None


def correctness_reward(completions, answer, **kwargs):
    rewards = []
    for comp, gold in zip(completions, answer):
        text = comp[0]["content"] if isinstance(comp, list) else comp
        pred_f = to_float(extract_pred(text))
        gold_f = to_float(extract_gold(gold))
        rewards.append(1.0 if pred_f is not None and gold_f is not None
                       and abs(pred_f - gold_f) < 1e-4 else 0.0)
    return rewards


def format_reward(completions, **kwargs):
    pat = re.compile(r"<answer>.*?</answer>", re.DOTALL)
    out = []
    for comp in completions:
        text = comp[0]["content"] if isinstance(comp, list) else comp
        out.append(0.2 if pat.search(text) else 0.0)
    return out


def build_dataset():
    ds = load_dataset("openai/gsm8k", "main", split="train")

    def fmt(ex):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": ex["question"]},
            ],
            "answer": ex["answer"],
        }

    return ds.map(fmt)


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
    )

    train_ds = build_dataset()

    config = GRPOConfig(
        output_dir=f"outputs/{RUN_NAME}",
        run_name=RUN_NAME,
        learning_rate=1e-6,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        num_generations=4,
        max_completion_length=512,
        max_steps=50,
        logging_steps=1,
        save_steps=50,
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
