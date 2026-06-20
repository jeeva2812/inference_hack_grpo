"""
smoke_test_code.py — minimal INFERENCE check for the code track (needs a GPU).

Unlike test_executor.py / test_pipeline.py (which are pure-logic, no model), this
actually LOADS Qwen2.5-Coder, generates on a few real cohort tasks, and runs the
verifier on the output. It exercises the exact path that eval_code / grpo_code /
extract_signals_code use: build_prompt -> generate -> extract_code_block ->
run_tests. ~1 min on an A100.

Run this on the box FIRST — if it prints rewards and "INFERENCE OK", the model +
prompt + extraction + verifier are all wired, so the full Phase 1/2 run won't die
30 minutes in on a dumb bug.

Usage:
  python smoke_test_code.py                          # 3 tasks from benchmark_slice
  python smoke_test_code.py --cohort harder_sibling --n 5
  python smoke_test_code.py --sample                 # also do a sampled rollout
"""

import argparse
import json
from pathlib import Path

from test_executor import extract_code_block, run_tests
from eval_code import build_prompt          # same prompt as eval/train/signals

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
COHORT_DIR = Path("cohorts_code")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="benchmark_slice")
    ap.add_argument("--n", type=int, default=3, help="number of tasks to try")
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--sample", action="store_true",
                    help="also generate one sampled rollout (mirrors extract_signals)")
    args = ap.parse_args()

    path = COHORT_DIR / f"{args.cohort}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run slice_cohorts_code.py first.")
    rows = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()][:args.n]

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"loading {MODEL_ID} ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    print(f"loaded. device={model.device}  cohort={args.cohort}  tasks={len(rows)}\n")

    def gen(prompt, do_sample):
        enc = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **enc, do_sample=do_sample,
                temperature=0.9 if do_sample else None,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tok.pad_token_id,
            )
        return tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True)

    def score(text, r):
        rate, err = run_tests(extract_code_block(text), r["test_list"],
                              setup_code=r.get("test_setup_code", "") or "",
                              entry_point=r.get("entry_point"))
        return rate, err

    solved = 0
    for i, r in enumerate(rows):
        prompt = build_prompt(r["text"], r["test_list"], tok)
        greedy = gen(prompt, do_sample=False)
        rate, err = score(greedy, r)
        solved += rate >= 1.0
        print(f"[{i+1}/{len(rows)}] {r['id']}  greedy reward(pass-rate)={rate:.2f}")
        print(f"  task : {r['text'][:75].strip()}")
        code = extract_code_block(greedy)
        for line in code.splitlines()[:6]:
            print(f"     | {line}")
        if err:
            print(f"  test error: {err[:160]}")
        if args.sample:
            srate, _ = score(gen(prompt, do_sample=True), r)
            print(f"  sampled reward={srate:.2f}")
        print()

    print(f"INFERENCE OK — model loaded, generated, and the verifier ran.")
    print(f"greedy solved {solved}/{len(rows)} (base model solving some is expected; "
          f"GRPO should raise it).")


if __name__ == "__main__":
    main()
