"""
make_report.py — assemble the deliverable into ONE self-contained HTML file
that prints straight to PDF (open in a browser → Ctrl/Cmd-P → "Save as PDF").

It reads the live pipeline outputs and embeds everything (figures as base64) so
the single file is portable — no server, no external assets:

  results/<domain>_summary.jsonl   per-cohort signals + acc_before/after/lift
  results/<domain>_eval.jsonl      raw accuracies (base + each cohort)
  report/figures.png               money + Pareto plots          (from plots.py)
  report/signals_vs_lift.png       every signal vs lift          (from plots.py)
  report/correlations.png          per-signal ρ bar              (from plots.py)

Correlations are recomputed here from the summaries (Pearson + Spearman) so the
tables are always consistent with the plotted data.

Usage:
  python plots.py            # (re)generate the figures first
  python make_report.py      # -> report/report.html
  python make_report.py --domain code --out report/report.html
"""

import argparse
import base64
import html
import json
from pathlib import Path

import numpy as np

from plots import load_summaries, correlations, all_signal_keys

RESULTS_DIR = Path("results")
REPORT_DIR = Path("report")


# ── signal glossary: what each quantity means + why it might predict lift ─────
# Keyed by the per-task signal name; the cohort columns carry a "_mean" suffix.
SIGNAL_DOCS = {
    "qlen_words": ("Tier 0", "len(prompt.split())",
                   "Question length in words. Structural difficulty proxy — the "
                   "cheap floor a real signal must beat."),
    "qlen_chars": ("Tier 0", "len(prompt)",
                   "Question length in characters. Same difficulty-proxy role as qlen_words."),
    "num_count": ("Tier 0", "count of numeric tokens",
                  "How many numbers the task mentions — a surface complexity proxy."),
    "operator_count": ("Tier 0", "count of + − × ÷ = % tokens",
                       "Arithmetic/operator density — surface complexity proxy."),
    "type_token_ratio": ("Tier 0", "unique_words / total_words",
                         "Lexical diversity of the prompt. Higher = less repetitive phrasing."),
    "prompt_ppl": ("Tier 1", "exp(mean token CE of the prompt)",
                   "Prompt perplexity (1 forward pass). Mid perplexity = novel-but-"
                   "not-alien: headroom to learn without being out-of-distribution."),
    "reward_mean": ("Tier 2", "mean reward over N rollouts",
                    "Average pass-rate. ~pass@1. Very high (saturated) or very low "
                    "leaves little for RL to move."),
    "reward_var": ("Tier 2", "Var(reward) over N rollouts",
                   "Baseline with a mechanism: the GRPO advantage ∝ within-group "
                   "reward std. Zero variance ⇒ zero gradient. The floor any signal must clear."),
    "pass_at_1": ("Tier 2", "fraction of rollouts that fully pass",
                  "Single-sample accuracy. The reliable capability the model already has."),
    "pass_at_n": ("Tier 2", "1 if any of N rollouts passes",
                  "Solvable-within-N. The latent capability reachable when sampling lucky."),
    "sampling_headroom": ("Tier 2 ★", "pass@N − pass@1",
                          "HERO #1. Latent minus reliable capability. High = the model "
                          "CAN do it but not reliably ⇒ RL has room to sharpen. Our safe anchor."),
    "answer_entropy": ("Tier 2", "entropy over distinct final answers",
                       "Outcome diversity across rollouts. Captures how undecided the model is."),
    "distinct_frac": ("Tier 2", "unique completions / N",
                      "Path diversity (cheap string-dedup). Low = mode-collapsed rollouts."),
    "completion_len_mean": ("Tier 2", "mean rollout length (words)",
                            "Verbosity. Often tracks difficulty / reasoning depth."),
    "completion_len_var": ("Tier 2", "Var(rollout length)",
                           "Spread of verbosity — instability in how the model approaches the task."),
    "intermediate_frac": ("Tier 2", "frac of tasks with 0 < pass-rate < 1",
                          "Goldilocks fraction: tasks the model sometimes-but-not-always "
                          "solves — exactly where RL gradient is non-zero."),
    "self_consistency_gap": ("Tier 2", "majority-vote acc − greedy acc",
                             "Does sampling+voting beat greedy? Positive = latent capability "
                             "greedy decoding misses; the majority-vs-greedy analog across domains."),
}


def doc_for(key: str):
    """Look up a glossary entry, tolerating the cohort '_mean' suffix."""
    base = key[:-5] if key.endswith("_mean") else key
    return SIGNAL_DOCS.get(base) or SIGNAL_DOCS.get(key) or ("", "", "")


# ── small html helpers ────────────────────────────────────────────────────────

def esc(x):
    return html.escape(str(x))


def fmt(x, nd=4):
    if x is None:
        return "—"
    if isinstance(x, float) and np.isnan(x):
        return "nan"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def img_tag(path: Path, alt: str):
    if not path.exists():
        return (f'<p class="missing">[missing {esc(path.name)} — run '
                f'<code>python plots.py</code> first]</p>')
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f'<img alt="{esc(alt)}" src="data:image/png;base64,{b64}"/>'


def load_jsonl(path: Path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


# ── section builders ──────────────────────────────────────────────────────────

def accuracy_table(domain, evals, rows):
    base = next((r["accuracy"] for r in evals if r["label"] == "base"), None)
    body = []
    body.append("<tr><th>Cohort</th><th>acc_before</th><th>acc_after</th>"
                "<th>lift = after − before</th><th>n_tasks</th></tr>")
    # order cohorts by lift desc
    srt = sorted(rows, key=lambda r: (r.get("lift") is None, -(r.get("lift") or 0)))
    for r in srt:
        lift = r.get("lift")
        cls = "pos" if (lift or 0) > 0 else ("neg" if (lift or 0) < 0 else "")
        body.append(
            f"<tr><td class='mono'>{esc(r['cohort'])}</td>"
            f"<td>{fmt(r.get('acc_before'))}</td>"
            f"<td>{fmt(r.get('acc_after'))}</td>"
            f"<td class='{cls}'>{('+' if (lift or 0) >= 0 else '')}{fmt(lift)}</td>"
            f"<td>{esc(r.get('n_tasks', '—'))}</td></tr>")
    return (f"<p>Baseline (no GRPO), greedy on the fixed MBPP test slice: "
            f"<b>acc_before = {fmt(base)}</b>. Each cohort trains one identical "
            f"GRPO run, then is re-evaluated on the same slice.</p>"
            f"<table>{''.join(body)}</table>")


def correlation_table(corrs):
    body = ["<tr><th>Signal</th><th>Tier</th><th>Pearson r</th>"
            "<th>Spearman ρ</th><th>What it measures</th></tr>"]
    for c in corrs:
        tier, _, meaning = doc_for(c["signal"])
        body.append(
            f"<tr><td class='mono'>{esc(c['signal'])}</td>"
            f"<td>{esc(tier)}</td>"
            f"<td>{fmt(c['pearson'], 3)}</td>"
            f"<td><b>{fmt(c['spearman'], 3)}</b></td>"
            f"<td class='small'>{esc(meaning)}</td></tr>")
    return f"<table>{''.join(body)}</table>"


def glossary_table(keys):
    body = ["<tr><th>Signal</th><th>Tier</th><th>Definition</th><th>Why it might predict lift</th></tr>"]
    # group: keep the order of appearance but stable
    for k in keys:
        tier, formula, meaning = doc_for(k)
        if not meaning:
            continue
        body.append(
            f"<tr><td class='mono'>{esc(k)}</td><td>{esc(tier)}</td>"
            f"<td class='small mono'>{esc(formula)}</td>"
            f"<td class='small'>{esc(meaning)}</td></tr>")
    return f"<table>{''.join(body)}</table>"


def insights(rows, corrs):
    lifts = [r["lift"] for r in rows if r.get("lift") is not None]
    pts = []
    if lifts:
        spread = max(lifts) - min(lifts)
        pts.append(
            f"Across {len(lifts)} cohorts the measured lift ranges from "
            f"{fmt(min(lifts))} to {fmt(max(lifts))} (spread {fmt(spread)}). "
            + ("This is a <b>narrow</b> spread — at the current step budget the "
               "dependent variable barely moves, so any correlation below is a "
               "weak, small-n signal rather than a conclusion."
               if spread < 0.05 else
               "There is real spread in lift for a metric to predict."))
    ranked = [c for c in corrs if not np.isnan(c["spearman"])]
    if ranked:
        top = ranked[0]
        tname = top["signal"]
        pts.append(
            f"Highest-ranked predictor by |Spearman ρ|: <b>{esc(tname)}</b> "
            f"(ρ = {fmt(top['spearman'], 3)}, r = {fmt(top['pearson'], 3)}).")
        # where does the hero candidate land?
        hero = next((c for c in ranked if c["signal"] == "sampling_headroom_mean"), None)
        if hero:
            pts.append(
                f"Our pre-registered anchor <b>sampling_headroom</b> lands at "
                f"ρ = {fmt(hero['spearman'], 3)}. "
                + ("It is not the leader here — an honest negative for the small-n, "
                   "low-spread regime." if abs(hero["spearman"]) < abs(top["spearman"]) - 1e-9
                   else "It leads the basket, consistent with the hypothesis."))
    pts.append(
        "With n≈5 cohorts per domain, rank correlation (Spearman) and "
        "leave-one-out fits are the only honest statistics — R² and p-values "
        "are noise at this scale, and we do not report them.")
    return "<ul>" + "".join(f"<li>{p}</li>" for p in pts) + "</ul>"


def build_html(domain, rows, evals, corrs, keys):
    css = """
    :root { --ink:#1a1a1a; --muted:#666; --line:#d9d9d9; --accent:#2b5797;
            --pos:#1a7f37; --neg:#b42318; }
    * { box-sizing: border-box; }
    body { font: 14px/1.55 -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
           color: var(--ink); max-width: 900px; margin: 0 auto; padding: 32px; }
    h1 { font-size: 26px; margin: 0 0 4px; }
    h2 { font-size: 19px; margin: 30px 0 8px; padding-bottom: 4px;
         border-bottom: 2px solid var(--accent); color: var(--accent); }
    h3 { font-size: 15px; margin: 18px 0 6px; }
    .sub { color: var(--muted); margin: 0 0 18px; }
    table { border-collapse: collapse; width: 100%; margin: 10px 0 16px; font-size: 12.5px; }
    th, td { border: 1px solid var(--line); padding: 5px 8px; text-align: left; vertical-align: top; }
    th { background: #f4f6fa; font-weight: 600; }
    td.mono, .mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
    .small { font-size: 11.5px; color: #333; }
    .pos { color: var(--pos); font-weight: 600; }
    .neg { color: var(--neg); font-weight: 600; }
    img { max-width: 100%; height: auto; border: 1px solid var(--line); border-radius: 4px;
          margin: 6px 0 4px; }
    figcaption { font-size: 11.5px; color: var(--muted); margin-bottom: 14px; }
    code { background: #f0f2f5; padding: 1px 4px; border-radius: 3px;
           font-family: ui-monospace, monospace; font-size: 12px; }
    .callout { background: #f7f9fc; border-left: 4px solid var(--accent);
               padding: 10px 14px; margin: 12px 0; border-radius: 0 4px 4px 0; }
    .honest { background: #fff7ed; border-left: 4px solid #c2750a; }
    ul { margin: 8px 0 14px; padding-left: 22px; page-break-inside: avoid; }
    li { margin: 4px 0; }
    @media print {
      body { padding: 0; max-width: none; }
      h2 { page-break-after: avoid; }
      table, figure { page-break-inside: avoid; }
      .page-break { page-break-before: always; }
    }
    """

    cohort_table = """
    <table>
      <tr><th>Cohort source</th><th>What it is (code track)</th><th>Expected quality</th></tr>
      <tr><td class="mono">benchmark_slice</td><td>MBPP train slice (in-distribution)</td><td>high</td></tr>
      <tr><td class="mono">harder_sibling</td><td>Harder/shifted sibling tasks</td><td>medium-high, shifted</td></tr>
      <tr><td class="mono">synthetic_good</td><td>Generated + tests pass</td><td>variable</td></tr>
      <tr><td class="mono">synthetic_degraded</td><td>Buggy tests / wrong refs (deliberate)</td><td>deliberately low</td></tr>
      <tr><td class="mono">random_control</td><td>Off-task / trivial snippets</td><td>near-zero (control)</td></tr>
    </table>"""

    parts = []
    parts.append(f"<h1>Which cheap, training-free signal predicts GRPO lift?</h1>")
    parts.append(f'<p class="sub">Code track — Qwen2.5-1.5B-Instruct on MBPP. '
                 f'Auto-generated report ({len(rows)} cohorts).</p>')

    # 1. Problem
    parts.append("<h2>1. Problem statement</h2>")
    parts.append(
        "<p>Reinforcement learning (GRPO) on a language model helps a lot on some "
        "training datasets and barely at all on others — but actually running GRPO "
        "to find out is expensive. <b>The question: which cheap metric, computable "
        "<i>before</i> any training, best predicts how much a model will improve "
        "(its <i>lift</i>) after GRPO on a given dataset?</b></p>")
    parts.append('<div class="callout"><b>The bet — cross-domain.</b> A metric that '
                 "predicts lift on <i>both</i> math (GSM8K) and code (MBPP), with two "
                 "different models, is a property of learnable data — not a quirk of one "
                 "benchmark. We do not pre-pick a winner; we compute a basket of candidate "
                 "signals and let the correlation with measured lift crown one.</div>")

    # 2. Design
    parts.append("<h2>2. Experimental design</h2>")
    parts.append(
        "<p><b>The eval benchmark is FIXED; the training cohort is the only VARIABLE.</b> "
        "Lift = <code>acc_after − acc_before</code> is always measured on the same held-out "
        "MBPP test slice, greedy/deterministic. What changes between runs is only the data "
        "we GRPO on. Cohorts are drawn from deliberately different sources so cohort "
        "<i>quality</i> — and therefore lift — has spread to predict:</p>")
    parts.append(cohort_table)
    parts.append("<p>Each cohort is size-matched, so source/quality is the only variable. "
                 "The degraded and control cohorts anchor the low-lift end: a good signal "
                 "should correctly flag <i>don't train on this</i>.</p>")

    # 3. Signals glossary
    parts.append("<h2>3. What the signals mean</h2>")
    parts.append("<p>Every signal is computed from one shared set of N sampled rollouts per "
                 "task (plus one greedy rollout), so the whole basket is near-free once the "
                 "rollouts exist. Tiers order them cheap → expensive for the cost–quality "
                 "frontier. ★ marks the pre-registered hero candidate.</p>")
    parts.append(glossary_table(keys))

    # 4. Methodology
    parts.append("<h2>4. Methodology</h2>")
    parts.append(
        "<ul>"
        "<li><b>Model:</b> Qwen2.5-1.5B-Instruct. <b>Reward:</b> fraction of the task's "
        "unit tests passed, executed in a sandbox (continuous).</li>"
        "<li><b>Phase 1 — signals:</b> 6 sampled rollouts (temp 0.9) + 1 greedy per task, "
        "scored on unit tests; aggregated to one row per cohort. Computed on the base model, "
        "before any training.</li>"
        "<li><b>Phase 2 — lift:</b> base eval (acc_before) → one identical GRPO run per "
        "cohort (250 steps, num_generations 8, max completion length 1024) → re-eval each "
        "checkpoint (acc_after) on the same slice.</li>"
        "<li><b>Analysis:</b> per-signal Pearson + Spearman vs lift; leave-one-out linear "
        "fit for the money plot. Small n ⇒ Spearman only, never R².</li>"
        "</ul>")

    # 5. Results
    parts.append('<h2 class="page-break">5. Results</h2>')
    parts.append("<h3>5.1 Accuracy and lift per cohort</h3>")
    parts.append(accuracy_table(domain, evals, rows))

    parts.append("<h3>5.2 Every signal vs lift</h3>")
    parts.append("<figure>" + img_tag(REPORT_DIR / "signals_vs_lift.png",
                 "signals vs lift") +
                 "<figcaption>Each panel: one signal (x) vs measured lift (y), per-cohort "
                 "points, least-squares guide line, Spearman ρ and Pearson r annotated.</figcaption></figure>")

    parts.append("<h3>5.3 Per-signal correlation with lift</h3>")
    parts.append(correlation_table(corrs))
    parts.append("<figure>" + img_tag(REPORT_DIR / "correlations.png",
                 "correlation bar") +
                 "<figcaption>Signals ranked by Spearman ρ against lift.</figcaption></figure>")

    parts.append("<h3>5.4 Deliverable figures</h3>")
    parts.append("<figure>" + img_tag(REPORT_DIR / "figures.png",
                 "money and pareto plots") +
                 "<figcaption>Left: predicted vs actual lift (leave-one-out fit) — points on "
                 "the y=x line mean the signal predicts. Right: cost–quality Pareto — metric "
                 "compute cost (log) vs predictive |ρ|.</figcaption></figure>")

    # 6. Insights
    parts.append("<h2>6. Insights</h2>")
    parts.append(insights(rows, corrs))

    # 7. Honesty
    parts.append("<h2>7. Limitations &amp; honesty</h2>")
    parts.append('<div class="callout honest">'
                 f"<b>n = {len(rows)} cohorts (code track only here).</b> This is a "
                 "hypothesis-generating result, not a conclusion. At the current step budget "
                 "the lift spread is small, which compresses every correlation toward noise. "
                 "The cross-domain claim only lands once the math track's summary is merged "
                 "and lift spread widens (more steps / more cohort sources / seed repeats).</div>")

    return (f"<!doctype html><html><head><meta charset='utf-8'>"
            f"<title>GRPO lift report — {esc(domain)}</title>"
            f"<style>{css}</style></head><body>{''.join(parts)}</body></html>")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="code")
    ap.add_argument("--out", default="report/report.html")
    args = ap.parse_args()

    rows = [r for r in load_summaries() if r.get("domain") == args.domain]
    if not rows:
        print(f"No results/{args.domain}_summary.jsonl rows — run phase2 + "
              f"merge_summary.py first.")
        return
    with_lift = [r for r in rows if r.get("lift") is not None]

    evals = load_jsonl(RESULTS_DIR / f"{args.domain}_eval.jsonl")
    corrs = correlations(with_lift) if with_lift else []
    keys = all_signal_keys(rows)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_html(args.domain, rows, evals, corrs, keys), encoding="utf-8")
    print(f"-> {out}  ({len(rows)} cohorts, {len(with_lift)} with lift)")
    print("Open it in a browser and Ctrl/Cmd-P → 'Save as PDF'.")


if __name__ == "__main__":
    main()
