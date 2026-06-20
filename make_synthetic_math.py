"""
make_synthetic_math.py — build synthetic / degraded cohorts from GSM8K, ZERO TOKENS.

Why programmatic instead of Claude: for the "low-quality data" cohorts we want
to control *exactly how* the data is broken, so the "does our metric flag bad
data?" story is crisp. Deterministic perturbation does that for free and
reproducibly. Save Anthropic credits for analysis/writeup.

Corruption modes (each keeps the QUESTION real, breaks the supervision):
  wrong_answer   : replace the gold #### number with a plausible wrong one.
                   Training reward then points the model at wrong answers.
  shuffled_steps : shuffle the reasoning sentences in the solution. Answer
                   stays correct but the chain-of-thought is incoherent.
  mismatched     : pair question_i with the solution+answer of question_j.
                   Total supervision noise — the control's control.
  trivial        : auto-generated trivial arithmetic ("What is a+b?"). A
                   near-zero-difficulty floor cohort.

Output: cohorts/<name>.jsonl, same schema as slice_cohorts_math.py plus a
`corruption` field, so extract_signals_math.py / training consume it unchanged.

Usage:
  python make_synthetic_math.py --mode wrong_answer   --out synthetic_degraded --n 256
  python make_synthetic_math.py --mode mismatched     --out random_control     --n 256
  python make_synthetic_math.py --mode trivial        --out trivial_floor      --n 256
"""

import argparse
import json
import random
import re
from pathlib import Path

OUT_DIR = Path("cohorts")
GOLD_RE = re.compile(r"####\s*(-?[\d,\.]+)")
STEPS_RE = re.compile(r"<<[^>]+>>")
NUM_RE = re.compile(r"-?\d+\.?\d*")


def gold_of(answer: str) -> str:
    m = GOLD_RE.search(answer)
    return m.group(1).replace(",", "").strip() if m else ""


def featurize(ex_id, question, answer, corruption):
    return {
        "id": str(ex_id),
        "question": question,
        "answer": answer,
        "gold": gold_of(answer),
        "steps": len(STEPS_RE.findall(answer)),
        "qlen": len(question.split()),
        "nums": len(NUM_RE.findall(question)),
        "corruption": corruption,
    }


# ── corruption modes ─────────────────────────────────────────────────────────

def corrupt_wrong_answer(question, answer, rng):
    gold = gold_of(answer)
    try:
        g = float(gold)
    except ValueError:
        return question, answer
    # plausible wrong: perturb by a non-zero delta, keep integer-ness
    delta = rng.choice([-3, -2, -1, 1, 2, 3, 10, -10]) or 1
    wrong = int(g + delta) if g == int(g) else round(g + delta, 2)
    new_answer = GOLD_RE.sub(f"#### {wrong}", answer)
    return question, new_answer


def corrupt_shuffled_steps(question, answer, rng):
    # split off the gold line, shuffle the preceding sentences
    parts = answer.split("####")
    body = parts[0].strip()
    tail = "####" + parts[1] if len(parts) > 1 else ""
    sents = [s.strip() for s in re.split(r"(?<=[.\n])", body) if s.strip()]
    rng.shuffle(sents)
    return question, " ".join(sents) + "\n" + tail


def make_trivial(idx, rng):
    a, b = rng.randint(2, 99), rng.randint(2, 99)
    q = f"What is {a} plus {b}?"
    ans = f"{a} + {b} = <<{a}+{b}={a+b}>>{a+b}\n#### {a+b}"
    return q, ans


# ── builders ─────────────────────────────────────────────────────────────────

def generate(mode: str, n: int, seed: int) -> list:
    """Return n rows without writing any file. Importable by slice_cohorts_math.py."""
    rng = random.Random(seed)
    rows = []

    if mode == "trivial":
        for i in range(n):
            q, a = make_trivial(i, rng)
            rows.append(featurize(i, q, a, mode))
    else:
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split="train")
        idxs = list(range(len(ds)))
        rng.shuffle(idxs)
        idxs = idxs[:n]

        if mode == "mismatched":
            shifted = idxs[1:] + idxs[:1]
            for qi, ai in zip(idxs, shifted):
                rows.append(featurize(qi, ds[qi]["question"], ds[ai]["answer"], mode))
        else:
            fn = {"wrong_answer": corrupt_wrong_answer,
                  "shuffled_steps": corrupt_shuffled_steps}[mode]
            for i in idxs:
                q, a = fn(ds[i]["question"], ds[i]["answer"], rng)
                rows.append(featurize(i, q, a, mode))

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
                   choices=["wrong_answer", "shuffled_steps", "mismatched", "trivial"])
    p.add_argument("--out", required=True, help="cohort name (file stem)")
    p.add_argument("--n", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    build(args.mode, args.out, args.n, args.seed)


if __name__ == "__main__":
    main()
