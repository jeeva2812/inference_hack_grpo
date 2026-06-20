"""
plots.py — the two deliverable figures, built to run OFFLINE against fake data.

  1. money plot   : predicted vs actual lift (leave-one-out fit), both domains,
                    y=x reference line. Points hugging the diagonal = metric works.
  2. pareto plot  : metric compute-cost (x, log) vs predictive power (Spearman ρ).
                    Answers the brief's cost-quality-frontier question directly.

No scipy dependency — Spearman + LOO linear fit implemented in numpy so this
runs anywhere. Reads results/*_summary.jsonl (Appendix C schema). Run with no
results present and it generates fake summaries so you can see the plots now.

Usage:
  python plots.py                       # uses results/ if present, else fake
  python plots.py --metric reward_var   # money plot driven by a chosen metric
"""

import argparse
import json
from pathlib import Path

import numpy as np

RESULTS_DIR = Path("results")

# rough seconds-to-compute per metric, for the Pareto x-axis (fill with real
# timings once measured; these are placeholders that give the plot its shape)
METRIC_COST = {
    "qlen_words_mean": 0.001,
    "prompt_ppl_mean": 0.5,
    "reward_var_mean": 8.0,
    "intermediate_frac": 8.0,
    "self_consistency_gap": 8.0,
    "sampling_headroom_mean": 8.0,
    "gradient_coherence": 40.0,
    "gradient_coherence_sketched": 12.0,
}


# ── numpy-only stats ─────────────────────────────────────────────────────────

def _rank(x):
    order = np.argsort(np.argsort(x))
    return order.astype(float)


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    rx, ry = _rank(x), _rank(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def loo_predict(metric_vals, lifts):
    """Leave-one-out linear-fit prediction of lift from a single metric."""
    x, y = np.asarray(metric_vals, float), np.asarray(lifts, float)
    preds = np.empty_like(y)
    for i in range(len(x)):
        mask = np.arange(len(x)) != i
        if np.std(x[mask]) == 0:
            preds[i] = y[mask].mean()
            continue
        slope, intercept = np.polyfit(x[mask], y[mask], 1)
        preds[i] = slope * x[i] + intercept
    return preds


# ── data loading ─────────────────────────────────────────────────────────────

def load_summaries():
    rows = []
    for f in sorted(RESULTS_DIR.glob("*_summary.jsonl")):
        with f.open(encoding="utf-8") as fh:
            rows.extend(json.loads(line) for line in fh if line.strip())
    return rows


def fake_summaries(seed=0):
    """Synthetic cohorts where lift truly depends on sampling_headroom + noise."""
    rng = np.random.default_rng(seed)
    rows = []
    for domain in ("math", "code"):
        for c in range(6):
            headroom = rng.uniform(0.05, 0.6)
            lift = 0.25 * headroom + rng.normal(0, 0.01)   # real signal + noise
            rows.append({
                "domain": domain, "cohort": f"{domain}_c{c}", "n_tasks": 256,
                "signals": {
                    "sampling_headroom_mean": headroom,
                    "reward_var_mean": headroom * 0.4 + rng.normal(0, 0.05),
                    "prompt_ppl_mean": rng.uniform(5, 15),
                    "qlen_words_mean": rng.uniform(30, 80),
                    "self_consistency_gap": headroom * 0.5 + rng.normal(0, 0.05),
                    "intermediate_frac": rng.uniform(0.3, 0.9),
                    "gradient_coherence": headroom * 0.6 + rng.normal(0, 0.08),
                    "gradient_coherence_sketched": headroom * 0.6 + rng.normal(0, 0.1),
                },
                "lift": float(lift),
            })
    return rows


# ── plots ────────────────────────────────────────────────────────────────────

def money_plot(rows, metric, ax):
    lifts = np.array([r["lift"] for r in rows])
    vals = np.array([r["signals"][metric] for r in rows])
    preds = loo_predict(vals, lifts)

    for domain, marker in (("math", "o"), ("code", "^")):
        idx = [i for i, r in enumerate(rows) if r["domain"] == domain]
        if idx:
            ax.scatter(preds[idx], lifts[idx], marker=marker, s=70,
                       label=domain, alpha=0.8)
    lo = min(lifts.min(), preds.min())
    hi = max(lifts.max(), preds.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5, label="perfect (y=x)")
    rho = spearman(preds, lifts)
    ax.set_xlabel(f"predicted lift  (LOO fit on {metric})")
    ax.set_ylabel("actual lift")
    ax.set_title(f"Predicted vs actual lift — ρ={rho:.2f}")
    ax.legend(fontsize=8)


def pareto_plot(rows, ax):
    lifts = np.array([r["lift"] for r in rows])
    xs, ys, names = [], [], []
    for metric, cost in METRIC_COST.items():
        if not all(metric in r["signals"] for r in rows):
            continue
        vals = np.array([r["signals"][metric] for r in rows])
        rho = spearman(vals, lifts)
        if np.isnan(rho):
            continue
        xs.append(cost)
        ys.append(abs(rho))
        names.append(metric)
    ax.scatter(xs, ys, s=70)
    for x, y, n in zip(xs, ys, names):
        ax.annotate(n.replace("_mean", ""), (x, y), fontsize=7,
                    xytext=(4, 4), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("compute cost per cohort (s, log)")
    ax.set_ylabel("predictive power  |Spearman ρ|")
    ax.set_title("Cost–quality Pareto frontier")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="sampling_headroom_mean")
    ap.add_argument("--out", default="report")
    args = ap.parse_args()

    rows = load_summaries()
    if not rows:
        print("No results/*_summary.jsonl found — using FAKE data for layout.")
        rows = fake_summaries()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    money_plot(rows, args.metric, a1)
    pareto_plot(rows, a2)
    fig.tight_layout()

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    png = out / "figures.png"
    fig.savefig(png, dpi=130)
    print(f"saved {png}")


if __name__ == "__main__":
    main()
