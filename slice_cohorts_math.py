"""
Partition GSM8K train split into cohorts and dump each as JSONL.

Heuristic v1 (model-agnostic, cheap):
  - reasoning_steps:  number of '<<...>>' calc annotations in the gold solution
    (GSM8K's gold solutions contain one per arithmetic step). This is a clean
    proxy for difficulty.
  - question_length:  whitespace token count of the question.
  - num_count:        how many numbers appear in the question.

Cohorts produced (size-matched so cohort is the only variable):
  short_easy      : few steps, short question
  short_hard      : many steps, short question
  long_easy       : few steps, long question
  long_hard       : many steps, long question
  random_baseline : uniform random draw from full pool

Each cohort is written to cohorts/<name>.jsonl with one task per line:
  {"id": str, "question": str, "answer": str, "gold": str,
   "steps": int, "qlen": int, "nums": int}

We will swap in model-DEPENDENT heuristics (pass-rate, reward variance,
learnability) once the baseline run works on Prime Intellect.
"""

import json
import random
import re
from pathlib import Path

from datasets import load_dataset

OUT_DIR = Path("cohorts")
COHORT_SIZE = 256          # tasks per cohort — keep small for $100 budget
SEED = 0

STEPS_RE = re.compile(r"<<[^>]+>>")
NUM_RE = re.compile(r"-?\d+\.?\d*")
GOLD_RE = re.compile(r"####\s*(-?[\d,\.]+)")


def featurize(ex, idx):
    q = ex["question"]
    a = ex["answer"]
    gold_m = GOLD_RE.search(a)
    return {
        "id": str(idx),
        "question": q,
        "answer": a,
        "gold": gold_m.group(1).replace(",", "").strip() if gold_m else "",
        "steps": len(STEPS_RE.findall(a)),
        "qlen": len(q.split()),
        "nums": len(NUM_RE.findall(q)),
    }


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  wrote {len(rows):4d} -> {path}")


def main():
    rng = random.Random(SEED)
    ds = load_dataset("openai/gsm8k", "main", split="train")
    rows = [featurize(ex, i) for i, ex in enumerate(ds)]
    print(f"loaded {len(rows)} train tasks")

    steps_sorted = sorted(r["steps"] for r in rows)
    qlen_sorted = sorted(r["qlen"] for r in rows)
    # medians as split thresholds (robust, no tuning)
    step_thr = steps_sorted[len(steps_sorted) // 2]
    qlen_thr = qlen_sorted[len(qlen_sorted) // 2]
    print(f"thresholds: steps_median={step_thr}  qlen_median={qlen_thr}")

    buckets = {
        "short_easy": [r for r in rows if r["steps"] <= step_thr and r["qlen"] <= qlen_thr],
        "short_hard": [r for r in rows if r["steps"] >  step_thr and r["qlen"] <= qlen_thr],
        "long_easy":  [r for r in rows if r["steps"] <= step_thr and r["qlen"] >  qlen_thr],
        "long_hard":  [r for r in rows if r["steps"] >  step_thr and r["qlen"] >  qlen_thr],
    }
    for name, pool in buckets.items():
        print(f"  {name:11s} pool={len(pool)}")
        if len(pool) < COHORT_SIZE:
            print(f"    !! pool smaller than COHORT_SIZE={COHORT_SIZE}, taking all")
        rng.shuffle(pool)
        write_jsonl(OUT_DIR / f"{name}.jsonl", pool[:COHORT_SIZE])

    pool = list(rows)
    rng.shuffle(pool)
    write_jsonl(OUT_DIR / "random_baseline.jsonl", pool[:COHORT_SIZE])

    print("done.")


if __name__ == "__main__":
    main()
