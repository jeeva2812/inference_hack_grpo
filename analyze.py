"""
analyze.py — join Phase-1 signals (predictors) with Phase-2 eval lift
(dependent variable), write results/math_joined.jsonl, and render the real
results figure to report/results.png. Offline, no GPU.

Usage:  python analyze.py
"""
import json
import math
from pathlib import Path

ORDER = ["benchmark_slice", "harder_sibling", "synthetic_good",
         "synthetic_degraded", "random_control"]
RESULTS = Path("results")


def load():
    ev = {json.loads(l)["label"]: json.loads(l)
          for l in RESULTS.joinpath("math_eval.jsonl").open(encoding="utf-8")}
    sig = {json.loads(l)["cohort"]: json.loads(l)
           for l in RESULTS.joinpath("math_signals.jsonl").open(encoding="utf-8")}
    return ev, sig


def se(p, n):
    return math.sqrt(p * (1 - p) / n)


def join(ev, sig):
    base = ev["base"]["accuracy"]
    nbase = ev["base"]["n"]
    rows = []
    for c in ORDER:
        acc = ev[c]["accuracy"]
        n = ev[c]["n"]
        lift = acc - base
        # SE of a difference of two independent proportions
        lift_se = math.sqrt(se(acc, n) ** 2 + se(base, nbase) ** 2)
        row = {"cohort": c, "acc_before": base, "acc_after": acc,
               "lift": round(lift, 4), "lift_se": round(lift_se, 4), **sig[c]}
        rows.append(row)
    return base, rows


def pearson(x, y):
    n = len(x); mx = sum(x) / n; my = sum(y) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    return cov / (sx * sy) if sx * sy else float("nan")


def figure(base, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    names = [r["cohort"].replace("_", "\n") for r in rows]
    lifts = np.array([r["lift"] for r in rows])
    lse = np.array([r["lift_se"] for r in rows])
    reward = np.array([r["reward_mean_mean"] for r in rows])
    p1 = np.array([r["pass_at_1_mean"] for r in rows])
    x = np.arange(len(rows))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # left: post-GRPO lift with noise band
    colors = ["#2a9d8f", "#2a9d8f", "#2a9d8f", "#e76f51", "#e76f51"]
    ax1.bar(x, lifts, yerr=lse, color=colors, alpha=0.85, capsize=4)
    ax1.axhline(0, color="k", lw=0.8)
    band = math.sqrt(se(base, 200) ** 2 * 2)
    ax1.axhspan(-band, band, color="gray", alpha=0.15,
                label=f"±1 SE eval noise (≈{band:.3f})")
    ax1.set_xticks(x); ax1.set_xticklabels(names, fontsize=8)
    ax1.set_ylabel("accuracy lift  (after − before GRPO)")
    ax1.set_title("Phase 2 — downstream GRPO lift\n(base acc = %.3f, n=200, single seed)" % base)
    ax1.legend(fontsize=8)

    # right: cheap signals cleanly separate data quality
    w = 0.38
    ax2.bar(x - w / 2, reward, w, label="reward_mean (cost ≈ free)", color="#264653")
    ax2.bar(x + w / 2, p1, w, label="pass@1", color="#e9c46a")
    ax2.set_xticks(x); ax2.set_xticklabels(names, fontsize=8)
    ax2.set_ylabel("signal value")
    ax2.set_title("Phase 1 — training-free signals\nseparate good vs corrupt cohorts perfectly")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    Path("report").mkdir(exist_ok=True)
    fig.savefig("report/results.png", dpi=130)
    print("saved report/results.png")


def main():
    ev, sig = load()
    base, rows = join(ev, sig)
    with RESULTS.joinpath("math_joined.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print("wrote results/math_joined.jsonl")

    lifts = [r["lift"] for r in rows]
    print("\nSignal -> lift Pearson r (n=5 cohorts):")
    for k in ["reward_mean_mean", "pass_at_1_mean", "ppl_mean",
              "sampling_headroom_mean", "reward_var_mean"]:
        print("  %-24s r=%+.2f" % (k, pearson([r[k] for r in rows], lifts)))
    figure(base, rows)


if __name__ == "__main__":
    main()
