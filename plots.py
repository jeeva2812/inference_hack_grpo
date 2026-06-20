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


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def all_signal_keys(rows):
    """Every numeric signal present (and non-null) for *all* given rows."""
    keys = set()
    for r in rows:
        keys.update(r.get("signals", {}).keys())
    usable = []
    for k in sorted(keys):
        vals = [r.get("signals", {}).get(k) for r in rows]
        if all(isinstance(v, (int, float)) for v in vals):
            usable.append(k)
    return usable


def correlations(rows):
    """Pearson + Spearman of every signal vs lift, sorted by |Spearman|."""
    lifts = np.array([r["lift"] for r in rows], float)
    out = []
    for k in all_signal_keys(rows):
        vals = np.array([r["signals"][k] for r in rows], float)
        out.append({
            "signal": k,
            "pearson": pearson(vals, lifts),
            "spearman": spearman(vals, lifts),
            "n": len(rows),
        })
    out.sort(key=lambda d: (-abs(d["spearman"]) if not np.isnan(d["spearman"]) else 1))
    return out


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


def signal_grid_plot(rows, fig):
    """One scatter (signal vs lift) per signal, ρ/r annotated, on a shared fig."""
    corrs = correlations(rows)            # already sorted by |Spearman|
    lifts = np.array([r["lift"] for r in rows], float)
    n = len(corrs)
    if n == 0:
        fig.text(0.5, 0.5, "no numeric signals to plot", ha="center")
        return
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    for i, c in enumerate(corrs):
        ax = fig.add_subplot(nrows, ncols, i + 1)
        vals = np.array([r["signals"][c["signal"]] for r in rows], float)
        for domain, marker in (("math", "o"), ("code", "^")):
            idx = [j for j, r in enumerate(rows) if r["domain"] == domain]
            if idx:
                ax.scatter(vals[idx], lifts[idx], marker=marker, s=45, alpha=0.8)
        # least-squares guide line when we have enough spread
        if len(vals) >= 2 and np.std(vals) > 0:
            slope, intercept = np.polyfit(vals, lifts, 1)
            xs = np.array([vals.min(), vals.max()])
            ax.plot(xs, slope * xs + intercept, "k--", lw=0.8, alpha=0.5)
        ax.set_title(f"{c['signal'].replace('_mean', '')}\n"
                     f"ρ={c['spearman']:.2f}  r={c['pearson']:.2f}", fontsize=8)
        ax.set_xlabel("signal", fontsize=7)
        ax.set_ylabel("lift", fontsize=7)
        ax.tick_params(labelsize=6)


def correlation_bar_plot(rows, ax):
    corrs = [c for c in correlations(rows) if not np.isnan(c["spearman"])]
    if not corrs:
        ax.text(0.5, 0.5, "need >=3 cohorts for correlation", ha="center")
        return
    names = [c["signal"].replace("_mean", "") for c in corrs]
    ys = np.arange(len(names))
    ax.barh(ys, [c["spearman"] for c in corrs], alpha=0.8)
    ax.set_yticks(ys)
    ax.set_yticklabels(names, fontsize=7)
    ax.invert_yaxis()
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Spearman ρ  (signal vs lift)")
    ax.set_title("Per-signal correlation with lift")


def write_correlations(rows, out_dir, domain_tag):
    """Persist the correlation table as CSV + JSONL alongside the figures."""
    corrs = correlations(rows)
    csv_path = out_dir / f"correlations{domain_tag}.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("signal,pearson,spearman,n\n")
        for c in corrs:
            f.write(f"{c['signal']},{c['pearson']:.4f},{c['spearman']:.4f},{c['n']}\n")
    json_path = out_dir / f"correlations{domain_tag}.jsonl"
    with json_path.open("w", encoding="utf-8") as f:
        for c in corrs:
            f.write(json.dumps(c) + "\n")
    return corrs, csv_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="sampling_headroom_mean")
    ap.add_argument("--out", default="report")
    ap.add_argument("--tag", default="",
                    help="suffix for output filenames (e.g. a cohort name) so "
                         "incremental snapshots don't overwrite each other")
    args = ap.parse_args()

    rows = load_summaries()
    if not rows:
        print("No results/*_summary.jsonl found — using FAKE data for layout.")
        rows = fake_summaries()
    # only cohorts with a measured lift can be correlated/plotted
    rows = [r for r in rows if r.get("lift") is not None]
    if not rows:
        print("No cohorts with a measured lift yet (need base eval + >=1 cohort eval).")
        return

    import os
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")  # writable cache dir
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""

    # 1) deliverable figures: money + pareto
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    money_plot(rows, args.metric, a1)
    pareto_plot(rows, a2)
    fig.tight_layout()
    fig.savefig(out / f"figures{tag}.png", dpi=130)
    plt.close(fig)

    # 2) every signal vs lift, in one grid
    gfig = plt.figure(figsize=(15, 3.2 * ((len(all_signal_keys(rows)) + 3) // 4) or 1))
    signal_grid_plot(rows, gfig)
    gfig.suptitle(f"Signal quantities vs accuracy lift  (n={len(rows)} cohorts)",
                  fontsize=11)
    gfig.tight_layout(rect=(0, 0, 1, 0.97))
    gfig.savefig(out / f"signals_vs_lift{tag}.png", dpi=130)
    plt.close(gfig)

    # 3) correlation bar + persisted table
    cfig, cax = plt.subplots(figsize=(7, max(3, 0.35 * len(all_signal_keys(rows)))))
    correlation_bar_plot(rows, cax)
    cfig.tight_layout()
    cfig.savefig(out / f"correlations{tag}.png", dpi=130)
    plt.close(cfig)

    corrs, csv_path = write_correlations(rows, out, tag)
    print(f"saved figures{tag}.png, signals_vs_lift{tag}.png, "
          f"correlations{tag}.png -> {out}/")
    print(f"wrote {csv_path}")
    top = [c for c in corrs if not np.isnan(c["spearman"])][:3]
    if top:
        print("top |ρ| signals: " +
              ", ".join(f"{c['signal']}={c['spearman']:+.2f}" for c in top))


if __name__ == "__main__":
    main()
