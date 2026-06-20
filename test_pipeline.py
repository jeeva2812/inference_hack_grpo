"""
test_pipeline.py — fast offline sanity checks. NO GPU, NO model, NO network.

Two ways to run:
    python test_pipeline.py        # plain — prints PASS/FAIL per check
    pytest test_pipeline.py        # if you have pytest

Covers the pure logic that's easy to get subtly wrong: signal math, the
synthetic corruptions, and the plotting stats.
"""

import numpy as np

import signals as sg
import plots as pl


# ── signals.py ───────────────────────────────────────────────────────────────

def _cohort(pass_prob, seed):
    _, tasks, emb, grads = sg._fake_cohort("c", pass_prob, seed=seed)
    return tasks, emb, grads


def test_pass_at_n_ge_pass_at_1():
    tasks, _, _ = _cohort(0.4, 1)
    for t in tasks:
        s = sg.reward_signals(t)
        assert s["pass_at_n"] >= s["pass_at_1"] - 1e-9
        assert s["sampling_headroom"] >= -1e-9   # never negative


def test_headroom_peaks_at_medium():
    # medium pass-rate should have more headroom than near-mastered easy
    easy = sg.aggregate_tasks(_cohort(0.9, 2)[0])["sampling_headroom_mean"]
    med = sg.aggregate_tasks(_cohort(0.45, 3)[0])["sampling_headroom_mean"]
    assert med > easy


def test_reward_var_zero_when_all_same():
    tasks, _, _ = _cohort(1.0, 4)          # everyone always correct
    assert sg.aggregate_tasks(tasks)["reward_var_mean"] < 1e-9


def test_gradient_coherence_bounds_and_alignment():
    # identical gradients -> coherence == 1; opposing -> ~0
    g = np.ones(100)
    assert abs(sg.gradient_coherence([g, g, g]) - 1.0) < 1e-6
    assert sg.gradient_coherence([g, -g]) < 1e-6
    # sketched approximates full on aligned-ish data
    rng = np.random.default_rng(0)
    grads = [rng.standard_normal(2000) + 5 for _ in range(20)]  # shared mean shift
    full = sg.gradient_coherence(grads)
    sk = sg.gradient_coherence_sketched(grads, proj_dim=256)
    assert abs(full - sk) < 0.15


def test_benchmark_coverage_self_is_high():
    _, emb, _ = _cohort(0.5, 5)
    # coverage of a set against itself should be ~1 (each task's nearest is itself)
    assert sg.benchmark_coverage(emb, emb) > 0.99


# ── make_synthetic.py ────────────────────────────────────────────────────────

def test_synthetic_trivial_is_self_consistent():
    import make_synthetic as ms
    import random
    rng = random.Random(0)
    q, a = ms.make_trivial(0, rng)
    # the gold number must equal the actual sum in the question text
    nums = [int(n) for n in ms.NUM_RE.findall(q)]
    assert ms.gold_of(a) == str(sum(nums))


def test_synthetic_wrong_answer_changes_gold():
    import make_synthetic as ms
    import random
    rng = random.Random(1)
    ans = "He had 5 apples. <<2+3=5>>5\n#### 5"
    _, new = ms.corrupt_wrong_answer("q", ans, rng)
    assert ms.gold_of(new) != "5"            # gold was corrupted
    assert "####" in new                      # structure preserved


# ── plots.py ─────────────────────────────────────────────────────────────────

def test_spearman_monotonic():
    assert abs(pl.spearman([1, 2, 3, 4], [2, 4, 6, 8]) - 1.0) < 1e-9
    assert abs(pl.spearman([1, 2, 3, 4], [8, 6, 4, 2]) + 1.0) < 1e-9


def test_loo_predict_recovers_linear_signal():
    x = np.linspace(0, 1, 8)
    y = 0.3 * x + 0.05                        # perfectly linear, no noise
    preds = pl.loo_predict(x, y)
    assert np.max(np.abs(preds - y)) < 1e-6


def test_fake_summaries_have_signal():
    rows = pl.fake_summaries()
    lifts = [r["lift"] for r in rows]
    head = [r["signals"]["sampling_headroom_mean"] for r in rows]
    # by construction lift depends on headroom -> positive rank correlation
    assert pl.spearman(head, lifts) > 0.5


# ── plain-python runner ──────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except Exception as e:  # noqa: BLE001
            print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} passed")
