"""
slice_cohorts_code.py — Phase 0: source-diverse, size-matched CODE cohorts.
Code-track analog of slice_cohorts_math.py.

5 cohorts x COHORT_SIZE tasks, spanning the quality spectrum:
  benchmark_slice    MBPP train       — in-distribution, high quality (baseline)
  harder_sibling     HumanEval        — harder, distribution-shifted
  synthetic_good     programmatic     — clean synthetic tasks, verified solvable
  synthetic_degraded MBPP + buggy_tests — broken supervision (low-lift anchor)
  random_control     MBPP mismatched  — incoherent supervision (control)

Plus a fixed eval slice (MBPP test, first EVAL_SIZE) used before and after every
run. The eval slice matches what eval_code.py evaluates (first-n of MBPP test),
and is NEVER used as a training cohort.

All rows share one schema (see make_synthetic_code.featurize):
  {id, text, test_list, test_setup_code, reference, source, n_tests, qlen,
   sol_lines, [corruption], [entry_point]}

The "good" cohorts (benchmark/harder/synthetic_good) are VALIDATED at build
time: a task is kept only if its reference solution passes its own test_list.
That guarantees those cohorts are genuinely solvable (so any low lift is about
learnability, not broken data). Degraded/control cohorts are intentionally NOT
validated — being broken is the point.

Usage:
  python slice_cohorts_code.py            # build all 5 cohorts + eval slice
  python slice_cohorts_code.py --dry-run  # print sizes only, no files written
"""

import argparse
import json
import random
import re
from pathlib import Path

import make_synthetic_code as ms
from test_executor import run_tests

OUT_DIR = Path("cohorts_code")
COHORT_SIZE = 256
EVAL_SIZE = 200
SEED = 42
MBPP_ID = "google-research-datasets/mbpp"

ASSERT_RE = re.compile(r"^\s*assert\s")


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


def reference_passes(row) -> bool:
    """True if the row's reference solution passes its own test_list."""
    if not row.get("reference") or not row.get("test_list"):
        return False
    rate, _ = run_tests(row["reference"], row["test_list"],
                        setup_code=row.get("test_setup_code", ""),
                        entry_point=row.get("entry_point"), timeout=6)
    return rate >= 1.0


def sample_validated(pool, n, rng, validate=True):
    """Shuffle pool, keep up to n rows (passing validation if requested)."""
    pool = list(pool)
    rng.shuffle(pool)
    kept, checked, dropped = [], 0, 0
    for row in pool:
        if len(kept) >= n:
            break
        if validate:
            checked += 1
            if not reference_passes(row):
                dropped += 1
                continue
        kept.append(row)
    if validate:
        print(f"  validated {checked} (dropped {dropped} unsolvable)")
    return kept


# ── cohort builders ───────────────────────────────────────────────────────────

def build_benchmark_slice(rng, n, validate=True):
    print("benchmark_slice  — loading MBPP train ...")
    from datasets import load_dataset
    ds = load_dataset(MBPP_ID, "full", split="train")
    pool = [ms.featurize(f"mbpp_{ex['task_id']}", ex["text"], ex["test_list"],
                         ex["code"], "mbpp_train",
                         setup=ex.get("test_setup_code", "")) for ex in ds]
    rows = sample_validated(pool, n, rng, validate)
    print(f"  pool={len(pool)}  kept={len(rows)}")
    return rows


def _humaneval_tests(check_src: str, entry_point: str) -> list:
    """Convert a HumanEval check() body into standalone asserts on entry_point."""
    tests = []
    for line in check_src.splitlines():
        if ASSERT_RE.match(line):
            tests.append(line.strip().replace("candidate", entry_point))
    return tests


def build_harder_sibling(rng, n, validate=True):
    print("harder_sibling   — loading HumanEval ...")
    from datasets import load_dataset
    pool = []
    try:
        try:
            ds = load_dataset("openai/openai_humaneval", split="test")
        except Exception:                       # older datasets versions
            ds = load_dataset("openai_humaneval", split="test")
        native_fallbacks = 0
        for ex in ds:
            ep = ex["entry_point"]
            reference = ex["prompt"] + ex["canonical_solution"]
            tests = _humaneval_tests(ex["test"], ep)
            row = ms.featurize(f"he_{ex['task_id']}", ex["prompt"], tests or [""],
                               reference, "humaneval", entry_point=ep)
            # If splitting check() into independent asserts loses logic (loops /
            # helper vars / multi-line asserts), the canonical solution fails its
            # own tests. Don't drop the task — keep HumanEval's native check() as a
            # single composite test unit (binary all-or-nothing, the standard
            # HumanEval metric). Lossless: all 164 tasks are retained.
            if not tests or not reference_passes(row):
                row["test_list"] = [ex["test"] + f"\ncheck({ep})"]
                row["n_tests"] = 1
                native_fallbacks += 1
            pool.append(row)
        print(f"  loaded HumanEval: {len(pool)} tasks "
              f"({native_fallbacks} via native check() fallback)")
    except Exception as e:  # noqa: BLE001
        print(f"  HumanEval failed: {e}")

    if not pool:
        print("  falling back to hard MBPP slice (most tests / longest solution)")
        from datasets import load_dataset
        ds = load_dataset(MBPP_ID, "full", split="train")
        allrows = [ms.featurize(f"mbpp_{ex['task_id']}", ex["text"], ex["test_list"],
                                ex["code"], "mbpp_hard",
                                setup=ex.get("test_setup_code", "")) for ex in ds]
        allrows.sort(key=lambda r: (r["sol_lines"], r["n_tests"]), reverse=True)
        pool = allrows[: n * 3]

    rows = sample_validated(pool, n, rng, validate)
    print(f"  pool={len(pool)}  kept={len(rows)}")
    return rows


# clean programmatic tasks — token-free "synthetic_good" source
def _synthetic_good_templates(rng):
    a = rng.randint(2, 40)
    lst = [rng.randint(1, 20) for _ in range(rng.randint(3, 6))]
    s = "".join(rng.choice("abcde") for _ in range(rng.randint(3, 7)))
    templates = [
        ("Write a function `sum_list(xs)` that returns the sum of a list of integers.",
         [f"assert sum_list({lst}) == {sum(lst)}", "assert sum_list([]) == 0"],
         "def sum_list(xs):\n    return sum(xs)"),
        ("Write a function `count_evens(xs)` that returns how many even numbers a list contains.",
         [f"assert count_evens({lst}) == {sum(1 for x in lst if x % 2 == 0)}",
          "assert count_evens([]) == 0"],
         "def count_evens(xs):\n    return sum(1 for x in xs if x % 2 == 0)"),
        ("Write a function `reverse_str(s)` that returns the reversed string.",
         [f"assert reverse_str({s!r}) == {s[::-1]!r}", "assert reverse_str('') == ''"],
         "def reverse_str(s):\n    return s[::-1]"),
        (f"Write a function `add_n(x)` that returns x plus {a}.",
         [f"assert add_n(0) == {a}", f"assert add_n(10) == {a + 10}"],
         f"def add_n(x):\n    return x + {a}"),
        ("Write a function `max_list(xs)` that returns the largest element of a non-empty list.",
         [f"assert max_list({lst}) == {max(lst)}"],
         "def max_list(xs):\n    return max(xs)"),
    ]
    return templates


def build_synthetic_good(rng, n, validate=True):
    print("synthetic_good   — programmatic clean tasks ...")
    pool = []
    i = 0
    while len(pool) < n * 2:
        for text, tests, ref in _synthetic_good_templates(rng):
            pool.append(ms.featurize(f"synth_good_{i}", text, tests, ref,
                                     "synthetic_programmatic", corruption=None))
            i += 1
    rows = sample_validated(pool, n, rng, validate)
    print(f"  pool={len(pool)}  kept={len(rows)}")
    return rows


def build_synthetic_degraded(rng, n):
    print("synthetic_degraded — MBPP + buggy_tests corruption ...")
    rows = ms.generate("buggy_tests", n, seed=rng.randint(0, 99999))
    print(f"  generated={len(rows)}")
    return rows


def build_random_control(rng, n):
    print("random_control   — MBPP text/test mismatched ...")
    rows = ms.generate("mismatched", n, seed=rng.randint(0, 99999))
    print(f"  generated={len(rows)}")
    return rows


def build_eval_slice(n):
    print("eval_slice       — loading MBPP test (first-n, matches eval_code) ...")
    from datasets import load_dataset
    ds = load_dataset(MBPP_ID, "full", split="test")
    rows = [ms.featurize(f"mbpp_{ds[i]['task_id']}", ds[i]["text"], ds[i]["test_list"],
                         ds[i]["code"], "mbpp_test",
                         setup=ds[i].get("test_setup_code", ""))
            for i in range(min(n, len(ds)))]
    print(f"  sampled={len(rows)}")
    return rows


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true", help="print sizes only")
    p.add_argument("--cohort-size", type=int, default=COHORT_SIZE)
    p.add_argument("--eval-size", type=int, default=EVAL_SIZE)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--no-validate", action="store_true",
                   help="skip reference-passes-tests validation (faster, less safe)")
    args = p.parse_args()

    rng = random.Random(args.seed)
    n = args.cohort_size
    validate = not args.no_validate

    cohorts = {
        "benchmark_slice":    build_benchmark_slice(rng, n, validate),
        "harder_sibling":     build_harder_sibling(rng, n, validate),
        "synthetic_good":     build_synthetic_good(rng, n, validate),
        "synthetic_degraded": build_synthetic_degraded(rng, n),
        "random_control":     build_random_control(rng, n),
    }

    print()
    for name, rows in cohorts.items():
        write_jsonl(OUT_DIR / f"{name}.jsonl", rows, dry_run=args.dry_run)

    eval_rows = build_eval_slice(args.eval_size)
    write_jsonl(OUT_DIR / "eval_slice.jsonl", eval_rows, dry_run=args.dry_run)

    print(f"\nPhase 0 (code) complete — {OUT_DIR}/")
    print(f"{'cohort':22s}  tasks  source")
    print("-" * 55)
    for name, rows in cohorts.items():
        src = rows[0]["source"] if rows else "?"
        print(f"  {name:20s}  {len(rows):4d}   {src}")
    print(f"  {'eval_slice':20s}  {len(eval_rows):4d}   mbpp_test")


if __name__ == "__main__":
    main()
