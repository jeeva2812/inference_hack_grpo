"""
merge_summary.py — join Phase-1 signals with Phase-2 eval into the final
per-cohort summary that plots.py / the regression consume. Domain-agnostic:
the same script closes both the math and code pipelines.

Inputs  (results/):
  <domain>_signals.jsonl  — one cohort_summary row per cohort {domain,cohort,n_tasks,signals}
  <domain>_eval.jsonl     — {label, accuracy} rows; label "base" is acc_before,
                            each cohort name is its acc_after
Output:
  <domain>_summary.jsonl  — signals row + acc_before/acc_after/lift (Appendix C)

Usage:
  python merge_summary.py --domain code
  python merge_summary.py --domain math
"""

import argparse
import json
from pathlib import Path

RESULTS_DIR = Path("results")


def load_jsonl(path: Path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def merge(domain: str):
    signals = load_jsonl(RESULTS_DIR / f"{domain}_signals.jsonl")
    evals = load_jsonl(RESULTS_DIR / f"{domain}_eval.jsonl")
    acc = {r["label"]: r["accuracy"] for r in evals}

    base = acc.get("base")
    if base is None:
        print(f"!! no 'base' record in {domain}_eval.jsonl — run the pre-train "
              f"eval first (python eval_{domain}.py --label base). Writing signals "
              f"with null lift.")

    out = []
    for row in signals:
        cohort = row["cohort"]
        after = acc.get(cohort)
        rec = dict(row)
        rec["acc_before"] = base
        rec["acc_after"] = after
        rec["lift"] = (after - base) if (after is not None and base is not None) else None
        out.append(rec)
        status = "ok" if rec["lift"] is not None else "MISSING eval"
        lift_s = f"{rec['lift']:+.4f}" if rec["lift"] is not None else "n/a"
        print(f"  {cohort:22s} lift={lift_s}  ({status})")

    out_path = RESULTS_DIR / f"{domain}_summary.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"-> {out_path}  ({len(out)} cohorts)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="code")
    args = ap.parse_args()
    merge(args.domain)


if __name__ == "__main__":
    main()
