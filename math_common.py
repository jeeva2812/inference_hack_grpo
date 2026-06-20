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
# Require >=1 digit inside the tag so an echoed literal "<answer>...</answer>"
# template (which the model sometimes parrots) is NOT matched as a prediction.
PRED_RE = re.compile(r"<answer>\s*(-?[\d.,]*\d[\d.,]*)\s*</answer>")
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)  # lenient, for extract
BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")          # Qwen-Math native final answer
NUM_RE = re.compile(r"-?\d+\.?\d*")
STEPS_RE = re.compile(r"<<[^>]+>>")                    # GSM8K calc annotations <<a*b=c>>


def extract_gold(answer_field: str) -> str:
    m = GOLD_RE.search(answer_field)
    return m.group(1).replace(",", "").strip() if m else ""


def _last_number(s: str) -> str:
    nums = NUM_RE.findall(s.replace(",", ""))
    return nums[-1] if nums else ""


def extract_pred(text: str) -> str:
    # 1) <answer>...</answer> — the format we ask for. Use the LAST such block and
    #    pull the number out of it (content may carry "$"/units/LaTeX). Empty or
    #    non-numeric blocks (e.g. an echoed "<answer>...</answer>" template) skip.
    for content in reversed(ANSWER_RE.findall(text)):
        n = _last_number(content)
        if n:
            return n
    # 2) \boxed{N} — Qwen2.5-Math's native final-answer format. Take the last box.
    for box in reversed(BOXED_RE.findall(text)):
        n = _last_number(box)
        if n:
            return n
    # 3) last resort: last number anywhere in the text.
    return _last_number(text)


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
