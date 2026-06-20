"""math_common.py — single source of truth for math-track answer scoring.

GSM8K gold/prediction extraction and the correctness test MUST be byte-for-byte
identical across signal extraction, GRPO training, and eval — otherwise `lift`
is measured against a different notion of correctness than training optimized,
silently corrupting the experiment. eval_math.py, extract_signals_math.py, and
grpo_math.py all import from here so the three can never drift.

Pure stdlib (only `re`) — safe to import without torch/datasets, so the test
suite and offline tooling can use these helpers GPU-free.
"""

import re

SYSTEM_PROMPT = (
    "You are a math tutor. Solve the problem step by step. "
    "Put your final numeric answer inside <answer>...</answer>."
)

GOLD_RE = re.compile(r"####\s*(-?[\d,\.]+)")           # GSM8K gold line: '#### N'
PRED_RE = re.compile(r"<answer>\s*(-?[\d,\.]+)\s*</answer>")
BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")          # Qwen-Math native final answer
NUM_RE = re.compile(r"-?\d+\.?\d*")
STEPS_RE = re.compile(r"<<[^>]+>>")                    # GSM8K calc annotations <<a*b=c>>


def extract_gold(answer_field: str) -> str:
    m = GOLD_RE.search(answer_field)
    return m.group(1).replace(",", "").strip() if m else ""


def extract_pred(text: str) -> str:
    # 1) the format we explicitly ask for: <answer>N</answer>
    m = PRED_RE.search(text)
    if m:
        return m.group(1).replace(",", "").strip()
    # 2) \boxed{N} — Qwen2.5-Math's NATIVE final-answer format. The model emits
    #    this even though the prompt asks for <answer> tags. Take the LAST box
    #    (the model's final answer after its CoT) and pull the number out of it;
    #    the box may carry "$", units, or LaTeX around the number.
    for box in reversed(BOXED_RE.findall(text)):
        nums = NUM_RE.findall(box.replace(",", ""))
        if nums:
            return nums[-1]
    # 3) last resort: last number anywhere. Unreliable for base models that
    #    ramble past the answer until max_new_tokens — that's why 1 and 2 exist.
    nums = NUM_RE.findall(text.replace(",", ""))
    return nums[-1] if nums else ""


def is_correct(pred: str, gold: str) -> bool:
    try:
        return abs(float(pred) - float(gold)) < 1e-4
    except ValueError:
        return False


def has_format(text: str) -> bool:
    """Strict: did the model emit a *numeric* <answer>...</answer> tag?"""
    return bool(PRED_RE.search(text))


def build_prompt(question: str, tokenizer) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
