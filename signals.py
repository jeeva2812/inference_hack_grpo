"""
signals.py — domain-agnostic cohort signal definitions (PROTOTYPE).

This module is the SHARED CONTRACT between the math and code tracks. It does
NOT touch a GPU, a model, or a dataset. It operates only on primitives that
each track collects in its own way:

    Per task, the track must produce:
      prompt        : str                     the question / problem
      completions   : list[str]               N sampled rollouts
      rewards       : list[float]             reward per rollout
                                              (binary for math, continuous for code)
      answers       : list[str | None]        extracted final answer per rollout
      gold          : str                      ground-truth answer key
      greedy_correct: bool                     pass@1 from greedy decoding
      embedding     : np.ndarray | None        task embedding (optional, tier 2.5)
      task_grad     : np.ndarray | None        GRPO pseudo-gradient (optional, tier 3)

Both tracks import the SAME functions here, so a signal means the same thing
in both domains. That is what makes the cross-domain comparison valid.

Run `python signals.py` to see every signal computed on fake data — no GPU,
no model, instant. Use it to sanity-check the math before spending compute.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Container for one task's collected primitives
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Task:
    prompt: str
    completions: list[str]
    rewards: list[float]
    answers: list[str | None]
    gold: str
    greedy_correct: bool
    prompt_ppl: float | None = None          # tier 1, needs 1 forward pass
    embedding: np.ndarray | None = None       # tier 2.5, needs embedder
    task_grad: np.ndarray | None = None       # tier 3, needs 1 backward pass

    @property
    def n(self) -> int:
        return len(self.completions)

    @property
    def n_correct(self) -> int:
        # treat reward > 0.5 as "correct" for pass@k accounting
        return sum(1 for r in self.rewards if r > 0.5)


# ─────────────────────────────────────────────────────────────────────────────
# Tier 0 — free, model-agnostic (text only)
# ─────────────────────────────────────────────────────────────────────────────

_NUM_RE = re.compile(r"-?\d+\.?\d*")
_OP_RE = re.compile(r"[\+\-\*/×÷=%]")


def text_signals(prompt: str) -> dict[str, float]:
    words = prompt.split()
    toks = set(words)
    return {
        "qlen_words": len(words),
        "qlen_chars": len(prompt),
        "num_count": len(_NUM_RE.findall(prompt)),
        "operator_count": len(_OP_RE.findall(prompt)),
        "type_token_ratio": len(toks) / max(len(words), 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Tier 2 — reward-based (reuse the N rollouts already generated)
# ─────────────────────────────────────────────────────────────────────────────

def _pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator (Codex paper). Prob >=1 of k samples correct."""
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def reward_signals(task: Task) -> dict[str, float]:
    r = np.asarray(task.rewards, dtype=float)
    n, c = task.n, task.n_correct

    # outcome diversity: entropy over the distinct final answers
    counts: dict[str, int] = {}
    for a in task.answers:
        key = a if a is not None else "<none>"
        counts[key] = counts.get(key, 0) + 1
    probs = np.array(list(counts.values()), dtype=float) / n
    answer_entropy = float(-(probs * np.log(probs + 1e-12)).sum())

    lengths = np.array([len(c.split()) for c in task.completions], dtype=float)

    return {
        "reward_mean": float(r.mean()),
        "reward_var": float(r.var()),
        "pass_at_1": c / n,                      # single-sample accuracy
        "pass_at_n": 1.0 if c >= 1 else 0.0,     # solvable within N tries
        # ── HERO CANDIDATE #1: sampling headroom ──
        # latent capability (any of N correct) minus reliable capability (1 sample).
        # high = model CAN do it but doesn't reliably -> RL has room to sharpen.
        "sampling_headroom": (1.0 if c >= 1 else 0.0) - (c / n),
        "answer_entropy": answer_entropy,
        "distinct_frac": len(set(task.completions)) / n,
        "completion_len_mean": float(lengths.mean()),
        "completion_len_var": float(lengths.var()),
    }


def self_consistency_correct(task: Task) -> bool:
    """Majority vote over extracted answers == gold?  (baseline metric input)"""
    counts: dict[str, int] = {}
    for a in task.answers:
        if a is None:
            continue
        counts[a] = counts.get(a, 0) + 1
    if not counts:
        return False
    winner = max(counts, key=counts.get)
    return winner == task.gold


# ─────────────────────────────────────────────────────────────────────────────
# Cohort-level aggregation of per-task signals
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_tasks(tasks: list[Task]) -> dict[str, float]:
    """Mean every per-task signal across the cohort, plus a few cohort-only ones."""
    per_task: list[dict[str, float]] = []
    for t in tasks:
        d = {}
        d.update(text_signals(t.prompt))
        d.update(reward_signals(t))
        if t.prompt_ppl is not None:
            d["prompt_ppl"] = t.prompt_ppl
        per_task.append(d)

    keys = per_task[0].keys()
    agg = {f"{k}_mean": float(np.mean([d[k] for d in per_task])) for k in keys}

    # intermediate_frac: fraction of tasks the model sometimes-but-not-always solves
    agg["intermediate_frac"] = float(
        np.mean([0 < t.n_correct < t.n for t in tasks])
    )
    # self-consistency gap: majority-vote acc minus greedy acc (cohort level)
    maj = np.mean([self_consistency_correct(t) for t in tasks])
    greedy = np.mean([t.greedy_correct for t in tasks])
    agg["self_consistency_gap"] = float(maj - greedy)
    return agg


# ─────────────────────────────────────────────────────────────────────────────
# Tier 2.5 — embedding-based cohort geometry (needs a local embedder)
# ─────────────────────────────────────────────────────────────────────────────

def _normalize(M: np.ndarray) -> np.ndarray:
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)


def embedding_signals(cohort_emb: np.ndarray) -> dict[str, float]:
    """cohort_emb: (n_tasks, dim).  Diversity / redundancy / effective rank."""
    E = _normalize(cohort_emb)
    sims = E @ E.T
    n = len(E)
    off = sims[~np.eye(n, dtype=bool)]
    diversity = float(1.0 - off.mean())                 # mean pairwise distance
    redundancy = float((off > 0.9).mean())              # near-duplicate fraction

    # PCA effective rank (participation ratio of eigenvalues of the cov)
    cov = np.cov(cohort_emb.T)
    eig = np.clip(np.linalg.eigvalsh(cov), 0, None)
    eff_rank = float((eig.sum() ** 2) / ((eig ** 2).sum() + 1e-12))

    return {
        "cohort_diversity": diversity,
        "redundancy": redundancy,
        "pca_effective_rank": eff_rank,
    }


def benchmark_coverage(cohort_emb: np.ndarray, benchmark_emb: np.ndarray) -> float:
    """
    YOUR IDEA: how close is the TRAINING cohort to the EVAL benchmark?

    For each cohort task, similarity to its nearest benchmark task; averaged.
    High = cohort looks like the test set (transfer easy, but maybe low headroom).
    Doubles as a CONTROL in the regression: lets us ask whether a metric
    predicts lift *beyond* mere distribution match.
    """
    C = _normalize(cohort_emb)
    B = _normalize(benchmark_emb)
    sims = C @ B.T                       # (n_cohort, n_benchmark)
    return float(sims.max(axis=1).mean())


# ─────────────────────────────────────────────────────────────────────────────
# Tier 3 — gradient coherence (the high-ceiling bet)
# ─────────────────────────────────────────────────────────────────────────────

def gradient_coherence(task_grads: list[np.ndarray]) -> float:
    """
    HERO CANDIDATE #2.  Do the cohort's tasks pull the model the same way?

        coherence = ||mean(grad)|| / mean(||grad||)   in [0, 1]

    ~1 : all gradients aligned -> tasks reinforce -> big, decisive update -> high lift.
    ~0 : gradients cancel -> tug-of-war -> model barely moves -> low lift.

    Captures inter-task SYNERGY that no per-task average can see.
    """
    G = np.stack(task_grads)                       # (n_tasks, dim)
    mean_norm = np.linalg.norm(G, axis=1).mean()
    norm_of_mean = np.linalg.norm(G.mean(axis=0))
    return float(norm_of_mean / (mean_norm + 1e-12))


def gradient_coherence_sketched(task_grads: list[np.ndarray], proj_dim: int = 256,
                                seed: int = 0) -> float:
    """
    Same metric, cheaper: random-project each gradient to proj_dim first
    (Johnson–Lindenstrauss preserves norms/inner-products in expectation).
    This is the trick that pushes gradient coherence toward the CHEAP corner
    of the Pareto frontier — i.e. our shot at 'breaking the frontier'.
    """
    G = np.stack(task_grads)
    rng = np.random.default_rng(seed)
    R = rng.standard_normal((G.shape[1], proj_dim)) / math.sqrt(proj_dim)
    Gp = G @ R
    mean_norm = np.linalg.norm(Gp, axis=1).mean()
    norm_of_mean = np.linalg.norm(Gp.mean(axis=0))
    return float(norm_of_mean / (mean_norm + 1e-12))


# ─────────────────────────────────────────────────────────────────────────────
# Build the full cohort summary row (matches Appendix C schema)
# ─────────────────────────────────────────────────────────────────────────────

def cohort_summary(domain: str, cohort: str, tasks: list[Task],
                   cohort_emb: np.ndarray | None = None,
                   benchmark_emb: np.ndarray | None = None,
                   task_grads: list[np.ndarray] | None = None) -> dict:
    signals = aggregate_tasks(tasks)
    if cohort_emb is not None:
        signals.update(embedding_signals(cohort_emb))
        if benchmark_emb is not None:
            signals["benchmark_coverage"] = benchmark_coverage(cohort_emb, benchmark_emb)
    if task_grads is not None:
        signals["gradient_coherence"] = gradient_coherence(task_grads)
        signals["gradient_coherence_sketched"] = gradient_coherence_sketched(task_grads)
    return {
        "domain": domain,
        "cohort": cohort,
        "n_tasks": len(tasks),
        "signals": signals,
        # acc_before / acc_after / lift filled in by the training+eval step
    }


# ─────────────────────────────────────────────────────────────────────────────
# Fake-data demo — run offline to verify the math (no GPU, no model)
# ─────────────────────────────────────────────────────────────────────────────

def _fake_cohort(name: str, pass_prob: float, n_tasks=20, n_roll=8, seed=0):
    rng = np.random.default_rng(seed)
    tasks = []
    for _ in range(n_tasks):
        rewards = (rng.random(n_roll) < pass_prob).astype(float).tolist()
        gold = "42"
        answers = [gold if r > 0.5 else str(rng.integers(0, 100)) for r in rewards]
        comps = [f"reasoning... answer is {a}" for a in answers]
        tasks.append(Task(
            prompt="What is 6 times 7?", completions=comps, rewards=rewards,
            answers=answers, gold=gold, greedy_correct=rewards[0] > 0.5,
            prompt_ppl=float(rng.uniform(5, 15)),
        ))
    emb = rng.standard_normal((n_tasks, 64))
    grads = [rng.standard_normal(1000) for _ in range(n_tasks)]
    return name, tasks, emb, grads


if __name__ == "__main__":
    bench_emb = np.random.default_rng(99).standard_normal((50, 64))

    print("Demo: 3 fake cohorts at different pass-rates\n" + "=" * 60)
    for name, prob, seed in [("easy", 0.85, 1), ("medium", 0.45, 2), ("hard", 0.1, 3)]:
        cname, tasks, emb, grads = _fake_cohort(name, prob, seed=seed)
        s = cohort_summary("demo", cname, tasks, emb, bench_emb, grads)["signals"]
        print(f"\n[{cname}]")
        for k in ["pass_at_1_mean", "pass_at_n_mean", "sampling_headroom_mean",
                  "reward_var_mean", "intermediate_frac", "self_consistency_gap",
                  "cohort_diversity", "benchmark_coverage",
                  "gradient_coherence", "gradient_coherence_sketched"]:
            print(f"  {k:30s} {s[k]:.4f}")
