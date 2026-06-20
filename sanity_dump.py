"""sanity_dump.py — see what the model ACTUALLY generates during a GRPO rollout.

Diagnoses the flat-reward problem: are completions truncated? do they emit
<answer> tags or \\boxed{}? is the math right? do groups have reward variance?

Usage:
    python sanity_dump.py --cohort benchmark_slice --prompts 4 --gen 4
"""

import argparse
import re

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from grpo_math import MODEL_ID, build_dataset
from math_common import extract_pred, is_correct

BOXED_RE = re.compile(r"\\boxed\{")
ANSWER_RE = re.compile(r"<answer>.*?</answer>", re.DOTALL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="benchmark_slice")
    ap.add_argument("--prompts", type=int, default=4, help="distinct prompts to inspect")
    ap.add_argument("--gen", type=int, default=4, help="completions per prompt (GRPO group size)")
    ap.add_argument("--max_new", type=int, default=1024)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="auto")
    model.eval()

    has_template = tok.chat_template is not None
    print(f"\n=== chat_template present on tokenizer: {has_template} ===\n")

    ds = build_dataset(args.cohort)

    for i in range(args.prompts):
        row = ds[i]
        gold = row["answer"]
        prompt_text = tok.apply_chat_template(
            row["prompt"], tokenize=False, add_generation_prompt=True
        )
        enc = tok(prompt_text, return_tensors="pt").to(model.device)

        print("=" * 100)
        print(f"PROMPT {i} | gold={gold!r}")
        print("-" * 100)
        print(prompt_text.strip()[:600])
        print("-" * 100)

        group_rewards = []
        for g in range(args.gen):
            with torch.no_grad():
                out = model.generate(
                    **enc,
                    max_new_tokens=args.max_new,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.95,
                    pad_token_id=tok.eos_token_id,
                )
            gen_ids = out[0][enc["input_ids"].shape[1]:]
            text = tok.decode(gen_ids, skip_special_tokens=True)
            n_new = gen_ids.shape[0]
            truncated = n_new >= args.max_new  # hit the cap → likely no final answer
            pred = extract_pred(text)
            correct = is_correct(pred, gold)
            group_rewards.append(1.0 if correct else 0.0)

            print(
                f"  gen{g}: len={n_new:4d} trunc={truncated!s:5} "
                f"answer_tag={bool(ANSWER_RE.search(text))!s:5} "
                f"boxed={bool(BOXED_RE.search(text))!s:5} "
                f"pred={pred!r:>10} correct={correct}"
            )
            print(f"        tail: ...{text[-220:].strip()!r}")

        std = torch.tensor(group_rewards).std().item() if len(group_rewards) > 1 else 0.0
        print(f"  >> group correctness rewards={group_rewards} std={std:.3f} "
              f"{'(ZERO-STD: contributes NO gradient)' if std == 0 else ''}")
        print()


if __name__ == "__main__":
    main()
