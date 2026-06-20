"""
eval.py — the dependent variable. Accuracy of a model on a FIXED benchmark
slice, deterministic (greedy). Run before and after each GRPO cohort run;
lift = acc_after - acc_before.

Design rules that make lift comparable across cohorts:
  - SAME test slice every time (first --n of the test split, deterministic).
  - GREEDY decoding (do_sample=False) — no sampling noise in the measurement.
  - identical prompt + answer-extraction as training (grpo_baseline.py).

Each run appends one record to results/<domain>_eval.jsonl:
  {"label": "base" | "<cohort>", "model": <path>, "n": 200, "accuracy": 0.41}
If a "base" record already exists, the run also prints lift vs base.

Usage:
  # baseline (before any training)
  python eval.py --model Qwen/Qwen2.5-Math-1.5B --label base --n 200

  # after training a cohort (point --model at the checkpoint dir)
  python eval.py --model outputs/math_medium_pass --label medium_pass --n 200

  # quick smoke test (tiny slice)
  python eval.py --n 8
"""

import argparse
import json
import re
from pathlib import Path

# NOTE: torch / datasets / transformers are imported lazily inside the
# functions that need a GPU, so the pure extraction helpers (and the test
# suite) can import this module without the heavy deps installed.

DEFAULT_MODEL = "Qwen/Qwen2.5-Math-1.5B"
RESULTS_DIR = Path("results")

SYSTEM_PROMPT = (
    "You are a math tutor. Solve the problem step by step. "
    "Put your final numeric answer inside <answer>...</answer>."
)

GOLD_RE = re.compile(r"####\s*(-?[\d,\.]+)")
PRED_RE = re.compile(r"<answer>\s*(-?[\d,\.]+)\s*</answer>")
NUM_RE = re.compile(r"-?\d+\.?\d*")


# ── extraction (identical semantics to training/signals) ─────────────────────

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


def build_prompt(question: str, tokenizer) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# ── evaluation ───────────────────────────────────────────────────────────────

def evaluate(model, tokenizer, examples, batch_size, max_new_tokens, verbose):
    import torch
    correct = 0
    with torch.no_grad():
        for i in range(0, len(examples), batch_size):
            batch = examples[i:i + batch_size]
            prompts = [build_prompt(ex["question"], tokenizer) for ex in batch]
            enc = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

            out = model.generate(
                **enc,
                do_sample=False,              # greedy = deterministic
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )
            gen = out[:, enc.input_ids.shape[1]:]  # left-pad -> uniform prompt len
            for j, ex in enumerate(batch):
                text = tokenizer.decode(gen[j], skip_special_tokens=True)
                ok = is_correct(extract_pred(text), extract_gold(ex["answer"]))
                correct += int(ok)
            if verbose:
                done = min(i + batch_size, len(examples))
                print(f"  {done}/{len(examples)}  running acc={correct/done:.3f}")
    return correct / len(examples)


def load_eval_records(path: Path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL, help="HF id or checkpoint dir")
    ap.add_argument("--label", default="base", help="'base' or cohort name")
    ap.add_argument("--domain", default="math")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=200, help="fixed test-slice size")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"Loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"            # required for batched generation
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    ds = load_dataset("openai/gsm8k", "main", split=args.split)
    examples = [ds[i] for i in range(min(args.n, len(ds)))]  # FIXED first-n slice
    print(f"Evaluating on {len(examples)} {args.split} tasks (greedy)...")

    acc = evaluate(model, tokenizer, examples, args.batch_size,
                   args.max_new_tokens, verbose=not args.quiet)
    print(f"\n{args.label}: accuracy = {acc:.4f}")

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"{args.domain}_eval.jsonl"
    records = load_eval_records(out_path)
    records = [r for r in records if r["label"] != args.label]   # overwrite same label
    record = {"label": args.label, "model": args.model, "n": len(examples),
              "accuracy": round(acc, 4)}
    records.append(record)
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"-> {out_path}")

    # report lift vs base if available
    base = next((r for r in records if r["label"] == "base"), None)
    if base and args.label != "base":
        print(f"lift vs base: {acc - base['accuracy']:+.4f} "
              f"({base['accuracy']:.4f} -> {acc:.4f})")


if __name__ == "__main__":
    main()
