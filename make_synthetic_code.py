"""
make_synthetic_code.py — synthetic / degraded CODE cohorts, ZERO TOKENS.
Code-track analog of make_synthetic_math.py.

KEY DIFFERENCE from the math track: for code the training reward is computed by
running the model's code against the cohort's `test_list`. So the supervision we
corrupt is the TESTS, not a gold answer.

Corruption modes (each keeps the task TEXT real, breaks the supervision):
  buggy_tests : mutate the expected value of each assert (numeric -> +1, else
                flip `==` to `!=`). Reward now points away from correct behavior
                — the analog of math `wrong_answer`. Anchors the low-lift end.
  mismatched  : pair task_i's text with task_j's test_list. Incoherent
                supervision — the control's control (analog of math `mismatched`).
  trivial     : auto-generated trivial functions ("return a+b") with correct
                tests. A near-zero-difficulty floor cohort.

Row schema (shared across all code cohorts; consumed by slice_cohorts_code.py /
extract_signals_code.py / grpo_code.py / eval_code.py unchanged):
  {id, text, test_list, test_setup_code, reference, source, n_tests, qlen,
   sol_lines, [corruption], [entry_point]}

Usage:
  python make_synthetic_code.py --mode buggy_tests --out synthetic_degraded --n 256
  python make_synthetic_code.py --mode mismatched  --out random_control     --n 256
  python make_synthetic_code.py --mode trivial     --out trivial_floor      --n 256
"""

import argparse
import json
import random
import re
from pathlib import Path

OUT_DIR = Path("cohorts_code")          # separate dir so we never clobber math cohorts
MBPP_ID = "google-research-datasets/mbpp"
INT_RHS_RE = re.compile(r"==\s*(-?\d+)\s*$")


def featurize(id_, text, test_list, reference, source,
              setup="", corruption=None, entry_point=None):
    row = {
        "id": str(id_),
        "text": text,
        "test_list": list(test_list),
        "test_setup_code": setup or "",
        "reference": reference or "",
        "source": source,
        "n_tests": len(test_list),
        "qlen": len(text.split()),
        "sol_lines": len((reference or "").strip().split("\n")) if reference else 0,
    }
    if corruption is not None:
        row["corruption"] = corruption
    if entry_point is not None:
        row["entry_point"] = entry_point
    return row


# ── corruption helpers ───────────────────────────────────────────────────────

def corrupt_assert(a: str) -> str:
    """Make one assert demand WRONG behavior. Numeric RHS -> +1; else == -> !=."""
    s = a.rstrip()
    m = INT_RHS_RE.search(s)
    if m:
        wrong = int(m.group(1)) + 1
        return s[:m.start()] + f"== {wrong}"
    if "==" in s:
        return s.replace("==", "!=", 1)
    return s            # nothing safely corruptible; leave as-is


def corrupt_tests(test_list) -> list:
    return [corrupt_assert(t) for t in test_list]


def make_trivial(idx, rng):
    a, b = rng.randint(2, 99), rng.randint(2, 99)
    text = ("Write a function `trivial_add(a, b)` that returns the sum of its "
            "two integer arguments.")
    test_list = [
        f"assert trivial_add({a}, {b}) == {a + b}",
        "assert trivial_add(0, 0) == 0",
        f"assert trivial_add(1, {b}) == {1 + b}",
    ]
    reference = "def trivial_add(a, b):\n    return a + b"
    return text, test_list, reference


# ── generate (no file written) — importable by slice_cohorts_code.py ─────────

def _load_mbpp(n, rng, split="train"):
    from datasets import load_dataset
    ds = load_dataset(MBPP_ID, "full", split=split)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    return ds, idxs[:n]


def generate(mode: str, n: int, seed: int) -> list:
    rng = random.Random(seed)
    rows = []

    if mode == "trivial":
        for i in range(n):
            text, tests, ref = make_trivial(i, rng)
            rows.append(featurize(f"trivial_{i}", text, tests, ref,
                                  "synthetic_trivial", corruption="trivial"))
        return rows

    ds, idxs = _load_mbpp(n, rng)

    if mode == "mismatched":
        shifted = idxs[1:] + idxs[:1]
        for ti, tj in zip(idxs, shifted):
            rows.append(featurize(
                f"mbpp_{ds[ti]['task_id']}", ds[ti]["text"], ds[tj]["test_list"],
                ds[ti]["code"], "mbpp_mismatched",
                setup=ds[tj].get("test_setup_code", ""), corruption="mismatched"))
    elif mode == "buggy_tests":
        for i in idxs:
            rows.append(featurize(
                f"mbpp_{ds[i]['task_id']}", ds[i]["text"],
                corrupt_tests(ds[i]["test_list"]), ds[i]["code"], "mbpp_buggy_tests",
                setup=ds[i].get("test_setup_code", ""), corruption="buggy_tests"))
    else:
        raise ValueError(f"unknown mode: {mode}")

    return rows


def build(mode: str, out: str, n: int, seed: int):
    rows = generate(mode, n, seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{out}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows ({mode}) -> {path}")
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True,
                   choices=["buggy_tests", "mismatched", "trivial"])
    p.add_argument("--out", required=True, help="cohort name (file stem)")
    p.add_argument("--n", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    build(args.mode, args.out, args.n, args.seed)


if __name__ == "__main__":
    main()
