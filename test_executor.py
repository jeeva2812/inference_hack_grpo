"""
Safe-ish test executor for code-generation tasks (CODE track verifier).

Extracts Python from a model generation, runs it against unit tests in a fresh
subprocess with a timeout, and returns the FRACTION of tests passed (continuous
reward, per the game plan's "reward = fraction of tests passed").

Two test formats are supported so the same executor serves every cohort source:

  1. MBPP-style  — `tests` is a list[str] of independent `assert` statements
                   (optionally with `setup_code`). Each assert is run in its own
                   try/except so the reward is a true fraction passed/total.
  2. HumanEval   — `tests` is a str module defining `def check(candidate): ...`;
                   pass `entry_point` (the function name) and we append the
                   `check(entry_point)` call the dataset omits. Scored binary.

CRITICAL: the previous version never CALLED `check`, so every completion scored
1.0. Always run this module's self-test (`python test_executor.py`) after edits.

Isolation note: this runs model-written code via `subprocess` + timeout only —
no filesystem/network sandbox. Fine on a disposable GPU box; do not run
untrusted cohorts on a machine you care about.
"""

import re
import subprocess
import sys
from typing import List, Optional, Sequence, Tuple, Union

RESULT_RE = re.compile(r"__RESULT__\s+(\d+)\s+(\d+)")
CODE_FENCE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)


def extract_code_block(text: str) -> str:
    """Pull Python out of a model generation.

    Prefer a fenced ```python ... ``` block; fall back to the raw text if the
    model emitted bare code. Returns "" only for empty input.
    """
    if not text:
        return ""
    m = CODE_FENCE_RE.search(text)
    if m and m.group(1).strip():
        return m.group(1).strip("\n")
    return text


def _indent(src: str, n: int = 4) -> str:
    pad = " " * n
    return "\n".join(pad + line if line.strip() else line
                     for line in src.splitlines())


def _build_harness(code: str,
                   tests: Union[str, Sequence[str]],
                   setup_code: str,
                   entry_point: Optional[str]) -> Tuple[str, int]:
    """Assemble a self-reporting program. Prints `__RESULT__ <passed> <total>`."""
    parts = [setup_code, code, ""]

    if isinstance(tests, str):
        # HumanEval-style module that defines check(candidate) but never calls it.
        parts.append(tests)
        call = f"check({entry_point})" if entry_point else "check(globals().get('candidate'))"
        parts.append("__passed__ = 0")
        parts.append("try:")
        parts.append(f"    {call}")
        parts.append("    __passed__ = 1")
        parts.append("except Exception:")
        parts.append("    pass")
        parts.append('print("__RESULT__", __passed__, 1)')
        total = 1
    else:
        # MBPP-style list of independent assert statements.
        total = len(tests)
        parts.append("__passed__ = 0")
        for t in tests:
            parts.append("try:")
            parts.append(_indent(t, 4))
            parts.append("    __passed__ += 1")
            parts.append("except Exception:")
            parts.append("    pass")
        parts.append(f'print("__RESULT__", __passed__, {total})')

    return "\n".join(parts), total


def run_tests(
    code: str,
    tests: Union[str, Sequence[str]],
    setup_code: str = "",
    entry_point: Optional[str] = None,
    timeout: int = 6,
) -> Tuple[float, str]:
    """Execute `code` against `tests`. Returns (pass_rate in [0,1], error_msg)."""
    if not code or not str(code).strip():
        return 0.0, "empty code"
    if isinstance(tests, str) and not tests.strip():
        return 0.0, "empty tests"
    if not isinstance(tests, str) and len(tests) == 0:
        return 0.0, "no tests"

    program, total = _build_harness(code, tests, setup_code, entry_point)
    try:
        result = subprocess.run(
            [sys.executable, "-c", program],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return 0.0, f"timeout after {timeout}s"
    except Exception as e:  # noqa: BLE001
        return 0.0, str(e)

    m = RESULT_RE.search(result.stdout or "")
    if not m:
        # harness never reached its print -> candidate failed to import/compile
        err = (result.stderr or result.stdout or "no __RESULT__ marker").strip()
        return 0.0, err[-500:]
    passed, reported_total = int(m.group(1)), int(m.group(2))
    denom = reported_total or total or 1
    return passed / denom, ""


def compute_test_reward(
    completions: Sequence,
    tests: Sequence[Union[str, Sequence[str]]],
    setup_code: Optional[Sequence[str]] = None,
    entry_points: Optional[Sequence[Optional[str]]] = None,
    timeout: int = 6,
) -> List[float]:
    """Vectorized reward for a batch of completions (used as a GRPO reward fn).

    `completions` may be raw strings or TRL chat-format ([{"content": ...}]).
    `tests[i]` pairs with `completions[i]`; `setup_code`/`entry_points` optional.
    """
    n = len(completions)
    setup_code = setup_code or [""] * n
    entry_points = entry_points or [None] * n
    rewards: List[float] = []
    for comp, t, su, ep in zip(completions, tests, setup_code, entry_points):
        text = comp[0]["content"] if isinstance(comp, list) else comp
        code = extract_code_block(text)
        rate, _ = run_tests(code, t, setup_code=su or "", entry_point=ep, timeout=timeout)
        rewards.append(rate)
    return rewards


# ── self-test (no model, no GPU) — run after every edit ──────────────────────
if __name__ == "__main__":
    checks = []

    # MBPP-style: fractional credit
    code_ok = "def add(a, b):\n    return a + b"
    tests = ["assert add(1, 2) == 3", "assert add(-1, 1) == 0", "assert add(2, 2) == 5"]
    rate, _ = run_tests(code_ok, tests)
    checks.append(("mbpp fractional 2/3", abs(rate - 2/3) < 1e-9))

    rate, _ = run_tests(code_ok, ["assert add(1, 2) == 3", "assert add(0, 0) == 0"])
    checks.append(("mbpp all pass", rate == 1.0))

    # wrong code must NOT score 1.0 (the bug we are fixing)
    wrong = "def add(a, b):\n    return 0"
    rate, _ = run_tests(wrong, ["assert add(1, 2) == 3"])
    checks.append(("wrong code scores 0", rate == 0.0))

    # HumanEval-style: check() must actually be called
    he_test = "def check(candidate):\n    assert candidate('abc', 'cba') == True\n    assert candidate('abc', 'abd') == False"
    he_code = "def same_chars(a, b):\n    return set(a) == set(b)"
    rate, _ = run_tests(he_code, he_test, entry_point="same_chars")
    checks.append(("humaneval correct -> 1.0", rate == 1.0))
    rate, _ = run_tests("def same_chars(a, b):\n    return True", he_test, entry_point="same_chars")
    checks.append(("humaneval wrong -> 0.0", rate == 0.0))

    # syntax error / timeout / empty
    rate, _ = run_tests("def f(:\n  pass", ["assert f() == 1"])
    checks.append(("syntax error -> 0", rate == 0.0))
    rate, _ = run_tests("def loop():\n    while True: pass", ["assert loop() == 1"], timeout=2)
    checks.append(("timeout -> 0", rate == 0.0))
    rate, _ = run_tests("", ["assert True"])
    checks.append(("empty code -> 0", rate == 0.0))

    # setup code is available to the tests
    rate, _ = run_tests("def g(x):\n    return x * 2", ["assert g(BASE) == 20"],
                        setup_code="BASE = 10")
    checks.append(("setup_code visible", rate == 1.0))

    passed = sum(ok for _, ok in checks)
    for name, ok in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    print(f"\n{passed}/{len(checks)} passed")
    sys.exit(0 if passed == len(checks) else 1)
