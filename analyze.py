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

# All signals available in math_signals.jsonl, with human-readable labels
# and a rough compute-cost tier for the Pareto framing.
SIGNAL_META = {
    "reward_mean_mean":          {"label": "reward mean",          "tier": "Med (N rollouts)",  "cost_rank": 3},
    "reward_var_mean":           {"label": "reward variance",      "tier": "Med (N rollouts)",  "cost_rank": 3},
    "pass_at_1_mean":            {"label": "pass@1",               "tier": "Med (N rollouts)",  "cost_rank": 3},
    "pass_at_n_mean":            {"label": "pass@N",               "tier": "Med (N rollouts)",  "cost_rank": 3},
    "sampling_headroom_mean":    {"label": "sampling headroom\n(pass@N-pass@1)", "tier": "Med (N rollouts)", "cost_rank": 3},
    "self_consistency_gap_mean": {"label": "self-consistency gap", "tier": "Med (N rollouts)",  "cost_rank": 3},
    "format_rate_mean":          {"label": "format rate",          "tier": "Free (reuse)",      "cost_rank": 1},
    "intermediate_frac":         {"label": "intermediate frac",    "tier": "Free (reuse)",      "cost_rank": 1},
    "ppl_mean":                  {"label": "prompt perplexity",    "tier": "Cheap (1 fwd)",     "cost_rank": 2},
}

SIGNALS = list(SIGNAL_META.keys())


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
    import matplotlib.gridspec as gridspec
    import numpy as np

    lifts = np.array([r["lift"] for r in rows])
    lse   = np.array([r["lift_se"] for r in rows])
    short = [r["cohort"].replace("_", "\n") for r in rows]
    cohort_colors = {
        "benchmark_slice":   "#264653",
        "harder_sibling":    "#2a9d8f",
        "synthetic_good":    "#e9c46a",
        "synthetic_degraded":"#f4a261",
        "random_control":    "#e76f51",
    }
    point_colors = [cohort_colors[r["cohort"]] for r in rows]

    # correlation of each signal with lift
    corrs = {s: pearson([r[s] for r in rows], list(lifts)) for s in SIGNALS}

    # ── Layout: 3 panels ──────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 12))
    gs  = gridspec.GridSpec(2, 2, figure=fig,
                            left=0.07, right=0.97, top=0.93, bottom=0.08,
                            hspace=0.45, wspace=0.35)
    ax1 = fig.add_subplot(gs[0, 0])   # money plot
    ax2 = fig.add_subplot(gs[0, 1])   # full correlation bar chart
    ax3 = fig.add_subplot(gs[1, :])   # signal heatmap across cohorts

    # ── Panel 1: money plot — sampling_headroom vs lift ──────────────────────
    best = "sampling_headroom_mean"
    xv   = np.array([r[best] for r in rows])
    ax1.errorbar(xv, lifts, yerr=lse, fmt="o", ms=10, capsize=5,
                 color="none", ecolor="gray", zorder=2,
                 markerfacecolor="none", markeredgecolor="gray")
    for x, y, c, lab in zip(xv, lifts, point_colors, short):
        ax1.scatter(x, y, s=120, color=c, zorder=3)
        ax1.annotate(lab, (x, y), fontsize=7.5, xytext=(6, 4),
                     textcoords="offset points")

    slope, intc = np.polyfit(xv, lifts, 1)
    xs = np.linspace(xv.min() - 0.02, xv.max() + 0.02, 50)
    ax1.plot(xs, slope * xs + intc, "--", color="#e76f51", lw=1.5,
             label=f"fit  (r = {corrs[best]:+.2f})")

    band = math.sqrt(se(base, 200) ** 2 * 2)
    ax1.axhspan(-band, band, color="gray", alpha=0.12,
                label=f"±1 SE noise  ≈ ±{band:.3f}")
    ax1.axhline(0, color="k", lw=0.6)
    ax1.set_xlabel("sampling headroom  (pass@N - pass@1)", fontsize=9)
    ax1.set_ylabel("GRPO lift  (acc after - acc before)", fontsize=9)
    ax1.set_title("Money plot: training-free signal vs actual lift\n"
                  f"(base={base:.2f}, n=5 cohorts, single seed)", fontsize=9)
    ax1.legend(fontsize=7.5, loc="upper left")

    # ── Panel 2: all-signal correlation bar chart ─────────────────────────────
    # Sort descending so positive bars go top→right (intuitive reading direction).
    # format_rate lands at the very bottom with a leftward (negative) bar.
    items  = sorted(corrs.items(), key=lambda kv: kv[1], reverse=True)  # descending
    labels = [SIGNAL_META[k]["label"].replace("\n", " ") for k, _ in items]
    vals   = [v for _, v in items]
    keys   = [k for k, _ in items]

    # Color: teal for positive correlation, red-orange for negative
    bar_colors = ["#2a9d8f" if v >= 0 else "#e76f51" for v in vals]
    y = np.arange(len(items))
    ax2.barh(y, vals, color=bar_colors, alpha=0.88, height=0.6)

    # Annotate with value — place inside bar for readability
    for yi, (v, k) in enumerate(zip(vals, keys)):
        # label sits just inside the far end of the bar
        xpos = v - 0.03 if v >= 0 else v + 0.03
        ha   = "right"  if v >= 0 else "left"
        color = "white" if abs(v) > 0.25 else "black"
        ax2.text(xpos, yi, f"{v:+.2f}", va="center", ha=ha,
                 fontsize=8.5, color=color, fontweight="bold")

    # Draw a dashed separator between positive and negative signals
    neg_count = sum(1 for v in vals if v < 0)
    if neg_count:
        sep_y = len(vals) - neg_count - 0.5   # y-coord between last+ and first-
        ax2.axhline(sep_y, color="#888", lw=1, linestyle="--", alpha=0.6)
        ax2.annotate("SPURIOUS (n=5)\n— no mechanism",
                     xy=(vals[-1] / 2, len(vals) - neg_count - 0.25),
                     fontsize=7, color="#c0392b", ha="center", style="italic")

    ax2.set_yticks(y)
    ax2.set_yticklabels(labels, fontsize=8.5)
    ax2.set_xlim(-1.1, 1.1)
    ax2.axvline(0, color="k", lw=1.0)
    # shade negative region
    ax2.axvspan(-1.1, 0, color="#fde8e8", alpha=0.25, zorder=0)
    ax2.axvspan(0, 1.1,  color="#e8f8f5", alpha=0.25, zorder=0)
    ax2.set_xlabel("Pearson r  (← negative correlation    |    positive correlation →)",
                   fontsize=8.5)
    ax2.set_title("All signals ranked by Pearson r with lift\n"
                  "⚠ n=5 cohorts — not statistically significant", fontsize=9)

    # ── Panel 3: signal heatmap across cohorts ────────────────────────────────
    # Normalise each signal to [0,1] across cohorts for visual comparison
    cohort_labels  = [r["cohort"].replace("_", " ") for r in rows]
    signal_labels  = [SIGNAL_META[s]["label"].replace("\n", " ") for s in SIGNALS]
    mat = np.array([[r[s] for s in SIGNALS] for r in rows])   # (5, 9)
    col_min = mat.min(axis=0, keepdims=True)
    col_max = mat.max(axis=0, keepdims=True)
    mat_norm = (mat - col_min) / np.where(col_max - col_min > 0, col_max - col_min, 1)

    im = ax3.imshow(mat_norm.T, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
    ax3.set_xticks(range(len(rows)))
    ax3.set_xticklabels(cohort_labels, fontsize=9)
    ax3.set_yticks(range(len(SIGNALS)))
    ax3.set_yticklabels(signal_labels, fontsize=8.5)
    ax3.set_title("Signal heatmap across cohorts  (each row normalised to [0,1])",
                  fontsize=9)

    # Annotate with raw values
    for ci, row in enumerate(rows):
        for si, s in enumerate(SIGNALS):
            v = row[s]
            txt = f"{v:.2f}" if abs(v) < 10 else f"{v:.1f}"
            ax3.text(ci, si, txt, ha="center", va="center",
                     fontsize=7, color="black" if mat_norm[ci, si] < 0.6 else "white")

    # Highlight lift bar below x-axis
    lift_norm = (lifts - lifts.min()) / max(lifts.max() - lifts.min(), 1e-6)
    for ci, (ln, lv) in enumerate(zip(lift_norm, lifts)):
        ax3.add_patch(plt.Rectangle((ci - 0.5, len(SIGNALS) - 0.5),
                                    1, 0.6,
                                    color=plt.cm.RdYlGn(ln), alpha=0.8, zorder=5))
        ax3.text(ci, len(SIGNALS) - 0.15, f"lift\n{lv:+.3f}",
                 ha="center", va="center", fontsize=6.5, fontweight="bold", zorder=6)
    ax3.set_ylim(-0.5, len(SIGNALS) + 0.15)

    plt.colorbar(im, ax=ax3, fraction=0.015, pad=0.01,
                 label="normalised signal value")

    fig.suptitle("GRPO lift predictors — Qwen2.5-1.5B-Instruct on GSM8K  (5 cohorts × 256 tasks)",
                 fontsize=12, fontweight="bold")

    Path("report").mkdir(exist_ok=True)
    fig.savefig("report/results.png", dpi=130, bbox_inches="tight")
    print(f"saved report/results.png  (best signal: {best}, r={corrs[best]:+.2f})")


def main():
    ev, sig = load()
    base, rows = join(ev, sig)
    with RESULTS.joinpath("math_joined.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print("wrote results/math_joined.jsonl")

    lifts = [r["lift"] for r in rows]
    corrs = {s: pearson([r[s] for r in rows], lifts) for s in SIGNALS}

    print("\nSignal -> lift Pearson r (n=5 cohorts, all signals):")
    for k in sorted(corrs, key=lambda k: corrs[k], reverse=True):
        label = SIGNAL_META[k]["label"].replace("\n", " ")
        tier  = SIGNAL_META[k]["tier"]
        print(f"  {label:<38s}  r={corrs[k]:+.2f}  [{tier}]")

    figure(base, rows)


if __name__ == "__main__":
    main()
