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


SIGNALS = ["reward_mean_mean", "reward_var_mean", "pass_at_1_mean",
           "pass_at_n_mean", "sampling_headroom_mean", "ppl_mean",
           "intermediate_frac", "format_rate_mean"]


def figure(base, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    lifts = np.array([r["lift"] for r in rows])
    lse = np.array([r["lift_se"] for r in rows])
    short = [r["cohort"].replace("_", " ") for r in rows]

    # correlation of each signal with lift
    corrs = {s: pearson([r[s] for r in rows], list(lifts)) for s in SIGNALS}
    # money-plot axis: the theory-motivated signal (room RL can close), NOT the
    # max-|r| signal — at n=5 the top |r| (format_rate) is a spurious artifact.
    best = "sampling_headroom_mean"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

    # LEFT: the money plot — best signal vs actual lift, with a fit line
    xv = np.array([r[best] for r in rows])
    ax1.errorbar(xv, lifts, yerr=lse, fmt="o", ms=9, capsize=4,
                 color="#264653", ecolor="gray", zorder=3)
    # least-squares fit line
    slope, intc = np.polyfit(xv, lifts, 1)
    xs = np.linspace(xv.min(), xv.max(), 50)
    ax1.plot(xs, slope * xs + intc, "--", color="#e76f51",
             label=f"linear fit  (Pearson r = {corrs[best]:+.2f})")
    for x, y, lab in zip(xv, lifts, short):
        ax1.annotate(lab, (x, y), fontsize=7.5, xytext=(6, 5),
                     textcoords="offset points")
    band = math.sqrt(se(base, rows and 200) ** 2 * 2)
    ax1.axhspan(-band, band, color="gray", alpha=0.12,
                label=f"±1 SE eval noise (≈{band:.3f})")
    ax1.axhline(0, color="k", lw=0.6)
    ax1.set_xlabel(f"{best.replace('_mean', '')}  (training-free signal)")
    ax1.set_ylabel("actual GRPO lift  (after − before)")
    ax1.set_title("Does the best signal predict lift?\n(base acc=%.2f, n=5 cohorts, single seed)" % base)
    ax1.legend(fontsize=8, loc="upper left")

    # RIGHT: |Pearson r| with lift, per signal — the correlation ranking
    items = sorted(corrs.items(), key=lambda kv: abs(kv[1]))
    labels = [k.replace("_mean", "").replace("_", " ") for k, _ in items]
    vals = [abs(v) for _, v in items]
    signs = ["+" if v >= 0 else "−" for _, v in items]
    y = np.arange(len(items))
    ax2.barh(y, vals, color="#2a9d8f", alpha=0.85)
    for yi, (v, s) in enumerate(zip(vals, signs)):
        ax2.text(v + 0.01, yi, f"{s}{v:.2f}", va="center", fontsize=8)
    ax2.set_yticks(y); ax2.set_yticklabels(labels, fontsize=8)
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("|Pearson r| with lift")
    ax2.set_title("Which signal best predicts lift?\n(⚠ n=5 — not statistically significant)")

    fig.tight_layout()
    Path("report").mkdir(exist_ok=True)
    fig.savefig("report/results.png", dpi=130)
    print("saved report/results.png  (best signal: %s, r=%+.2f)" % (best, corrs[best]))


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
