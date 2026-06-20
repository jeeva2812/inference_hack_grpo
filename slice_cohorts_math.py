"""
slice_cohorts_math.py — Phase 0: source-diverse, size-matched math cohorts.

5 cohorts × COHORT_SIZE tasks, spanning the quality spectrum:
  benchmark_slice    GSM8K train   — in-distribution, high quality (the baseline)
  harder_sibling     MATH/GSM-hard — harder, distribution-shifted
  synthetic_good     Orca Math     — synthetic but verified, variable quality
  synthetic_degraded GSM8K + wrong_answer corruption — bad supervision signal
  random_control     GSM8K Q/A mismatched — near-zero reward floor (control)

Plus a fixed eval slice (GSM8K test, 256 tasks) used before and after every run.

All rows share one schema:
  {id, question, answer, gold, steps, qlen, nums, source, [corruption]}

How a cohort is built
─────────────────────
Each source dataset is loaded, converted to the shared schema via a featurizer,
shuffled with a fixed random seed, then truncated to COHORT_SIZE. The degraded
and control cohorts delegate to make_synthetic_math.generate() so the corruption
logic lives in one place. Every cohort is written to cohorts/<name>.jsonl.

Usage:
  python slice_cohorts_math.py           # build all 5 cohorts + eval slice
  python slice_cohorts_math.py --dry-run # print sizes only, no files written
"""

import argparse
import json
import random
import re
from pathlib import Path

from math_common import GOLD_RE, NUM_RE, STEPS_RE, extract_gold

OUT_DIR = Path("cohorts")
COHORT_SIZE = 256
EVAL_SIZE = 256
SEED = 42

BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")   # MATH-competition gold, slice-only


# ── utilities ─────────────────────────────────────────────────────────────────

def write_jsonl(path: Path, rows: list, dry_run: bool = False):
    if dry_run:
        print(f"  [dry-run] {len(rows):4d} rows -> {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  wrote {len(rows):4d} -> {path}")


def shuffle_sample(pool: list, n: int, rng: random.Random) -> list:
    pool = list(pool)
    rng.shuffle(pool)
    return pool[:n]


def is_numeric(s: str) -> bool:
    try:
        float(s.replace(",", "").strip())
        return True
    except (ValueError, AttributeError):
        return False


# ── per-source featurizers ────────────────────────────────────────────────────

def _base(id_, q, a, gold, source, corruption=None):
    row = {
        "id": id_,
        "question": q,
        "answer": a,
        "gold": gold,
        "steps": len(STEPS_RE.findall(a)),
        "qlen": len(q.split()),
        "nums": len(NUM_RE.findall(q)),
        "source": source,
    }
    if corruption is not None:
        row["corruption"] = corruption
    return row


def featurize_gsm8k(ex, idx, source="gsm8k_train"):
    gold = extract_gold(ex["answer"])
    return _base(f"{source}_{idx}", ex["question"], ex["answer"], gold, source)


def featurize_math(ex, idx):
    """MATH competition dataset (lighteval/MATH or hendrycks/competition_math)."""
    q = ex.get("problem", ex.get("question", ""))
    a = ex.get("solution", ex.get("answer", ""))
    gold_field = ex.get("answer", "")
    m = BOXED_RE.search(gold_field) or BOXED_RE.search(a)
    gold = m.group(1).strip() if m else gold_field.strip()
    return _base(f"math_{idx}", q, a, gold, "math_competition")


def featurize_orca(ex, idx):
    """Orca Math: question + answer (solution) fields."""
    q = ex.get("question", "")
    a = ex.get("answer", "")
    m = GOLD_RE.search(a)
    if m:
        gold = m.group(1).replace(",", "").strip()
    else:
        nums = NUM_RE.findall(a)
        gold = nums[-1] if nums else ""
    return _base(f"orca_{idx}", q, a, gold, "orca_math")


# ── cohort builders ───────────────────────────────────────────────────────────

def build_benchmark_slice(rng: random.Random, n: int) -> list:
    print("benchmark_slice  — loading GSM8K train ...")
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="train")
    pool = [featurize_gsm8k(ex, i) for i, ex in enumerate(ds)]
    rows = shuffle_sample(pool, n, rng)
    print(f"  pool={len(pool)}  sampled={len(rows)}")
    return rows


def build_harder_sibling(rng: random.Random, n: int) -> list:
    print("harder_sibling   — loading MATH competition dataset ...")
    from datasets import load_dataset
    pool = []

    for ds_id, kwargs in [
        ("lighteval/MATH", {"name": "all"}),
        ("EleutherAI/hendrycks_math", {"name": "all"}),
        ("math-ai/MATH", {}),
    ]:
        try:
            ds = load_dataset(ds_id, split="train", **kwargs)
            pool = [featurize_math(ex, i) for i, ex in enumerate(ds)]
            pool = [r for r in pool if is_numeric(r["gold"])]
            print(f"  loaded {ds_id}: {len(pool)} numeric-answer problems")
            break
        except Exception as e:
            print(f"  {ds_id} failed: {e}")

    if not pool:
        print("  falling back to hard-difficulty GSM8K slice (top 30% by step count)")
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split="train")
        all_rows = [featurize_gsm8k(ex, i, "gsm8k_hard") for i, ex in enumerate(ds)]
        step_thr = sorted(r["steps"] for r in all_rows)[int(len(all_rows) * 0.70)]
        pool = [r for r in all_rows if r["steps"] > step_thr]
        for r in pool:
            r["source"] = "gsm8k_hard"

    rows = shuffle_sample(pool, n, rng)
    print(f"  sampled={len(rows)}")
    return rows


def build_synthetic_good(rng: random.Random, n: int) -> list:
    print("synthetic_good   — loading Orca Math ...")
    from datasets import load_dataset
    pool = []
    try:
        ds = load_dataset("microsoft/orca-math-word-problems-200k", split="train")
        pool = [featurize_orca(ex, i) for i, ex in enumerate(ds)]
        pool = [r for r in pool if r["gold"]]
        print(f"  loaded orca-math: {len(pool)} rows with gold answers")
    except Exception as e:
        print(f"  Orca Math failed: {e}")
        print("  falling back to programmatic multi-step word problems")
        pool = _gen_synthetic_good(n * 4, rng)

    rows = shuffle_sample(pool, n, rng)
    print(f"  sampled={len(rows)}")
    return rows


def _gen_synthetic_good(n: int, rng: random.Random) -> list:
    """Multi-step arithmetic word problems, programmatically generated."""
    templates = [
        ("apples",    "basket",    "vendor"),
        ("books",     "shelf",     "library"),
        ("coins",     "jar",       "bank"),
        ("students",  "classroom", "school"),
        ("kilometers","trip",      "driver"),
    ]
    rows = []
    for i in range(n):
        obj, loc, agent = templates[i % len(templates)]
        a, b, c = rng.randint(5, 50), rng.randint(2, 20), rng.randint(1, 15)
        step1 = a * b
        step2 = step1 + c
        q = (
            f"There are {a} {obj} in each {loc}. "
            f"After collecting from {b} {agent}s, {c} more {obj} were added. "
            f"How many {obj} are there in total?"
        )
        ans = (
            f"First: {a} × {b} = <<{a}*{b}={step1}>>{step1} {obj}.\n"
            f"Then add {c}: <<{step1}+{c}={step2}>>{step2} {obj} total.\n"
            f"#### {step2}"
        )
        rows.append(_base(f"synth_good_{i}", q, ans, str(step2),
                          "synthetic_programmatic"))
    return rows


def build_synthetic_degraded(rng: random.Random, n: int) -> list:
    print("synthetic_degraded — GSM8K + wrong_answer corruption ...")
    from make_synthetic_math import generate
    rows = generate("wrong_answer", n, seed=rng.randint(0, 99999))
    for r in rows:
        r.setdefault("source", "gsm8k_corrupted")
    print(f"  generated={len(rows)}")
    return rows


def build_random_control(rng: random.Random, n: int) -> list:
    print("random_control   — GSM8K Q/A mismatched ...")
    from make_synthetic_math import generate
    rows = generate("mismatched", n, seed=rng.randint(0, 99999))
    for r in rows:
        r.setdefault("source", "gsm8k_mismatched")
    print(f"  generated={len(rows)}")
    return rows


def build_eval_slice(n: int) -> list:
    print("eval_slice       — loading GSM8K test ...")
    from datasets import load_dataset
    rng = random.Random(SEED)
    ds = load_dataset("openai/gsm8k", "main", split="test")
    pool = [featurize_gsm8k(ex, i, "gsm8k_test") for i, ex in enumerate(ds)]
    rng.shuffle(pool)
    rows = pool[:n]
    print(f"  pool={len(pool)}  sampled={len(rows)}")
    return rows


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true",
                   help="Print sizes only, write no files")
    p.add_argument("--cohort-size", type=int, default=COHORT_SIZE)
    p.add_argument("--eval-size", type=int, default=EVAL_SIZE)
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args()

    rng = random.Random(args.seed)
    n = args.cohort_size

    cohorts = {
        "benchmark_slice":    build_benchmark_slice(rng, n),
        "harder_sibling":     build_harder_sibling(rng, n),
        "synthetic_good":     build_synthetic_good(rng, n),
        "synthetic_degraded": build_synthetic_degraded(rng, n),
        "random_control":     build_random_control(rng, n),
    }

    print()
    for name, rows in cohorts.items():
        write_jsonl(OUT_DIR / f"{name}.jsonl", rows, dry_run=args.dry_run)

    eval_rows = build_eval_slice(args.eval_size)
    write_jsonl(OUT_DIR / "eval_slice.jsonl", eval_rows, dry_run=args.dry_run)

    print(f"\nPhase 0 complete — {OUT_DIR}/")
    print(f"{'cohort':22s}  tasks  source")
    print("-" * 55)
    for name, rows in cohorts.items():
        src = rows[0]["source"] if rows else "?"
        print(f"  {name:20s}  {len(rows):4d}   {src}")
    print(f"  {'eval_slice':20s}  {len(eval_rows):4d}   gsm8k_test")


if __name__ == "__main__":
    main()
