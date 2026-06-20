# Dashboard ideas

The dashboard is how we *sell* the result in the demo. Judges look at it for
60 seconds — it has to land the story without us talking. Keep it dead simple.

## Recommended stack

**Streamlit.** One `pip install streamlit`, pure Python, no JS, hot-reload.
A single `dashboard.py` reads the cohort summary files and renders. We can
build it locally against fake data (GPU OFF) and it Just Works once real
numbers land.

```
pip install streamlit plotly pandas
streamlit run dashboard.py
```

Alternative if we want zero-infra sharing: dump the same plots to a single
static `report.html` with Plotly's `fig.write_html()`. No server needed,
email-able, survives the demo wifi dying. Honestly do BOTH — Streamlit to
explore, static HTML as the backup artifact.

## What it reads

Both tracks emit the shared schema (see GAMEPLAN). Point the dashboard at:
```
results/math_summary.jsonl
results/code_summary.jsonl
```
Concatenate into one dataframe with a `domain` column. Everything keys off
that.

## The four panels (in priority order)

### 1. The money plot — predicted vs actual lift
Scatter. X = metric-predicted lift, Y = actual measured lift. One point per
cohort. Color by `domain` (math vs code), shape by cohort. Draw the y=x line.
Points hugging the diagonal = the metric works. **This is the slide that
wins or loses.** Put it top-left, biggest.

Add a dropdown to switch which metric drives the X axis
(`self_consistency_gap`, `reward_var`, `prompt_ppl`) so we can show the hero
beating the baselines live.

### 2. The Pareto frontier
Scatter. X = compute-cost (seconds to compute the metric, log scale),
Y = predictive power (Spearman ρ vs lift). One point per metric. The brief
asked for this *by name* — most teams skip it. Annotate each point with the
metric name. The story: cheap geometric metrics sit low-left, behavioral
metrics sit upper-right, and we show which ones "break the frontier."

### 3. Per-cohort bar chart
Grouped bars: each cohort's `acc_before` and `acc_after`, lift as the gap.
Sorted by lift. Lets a judge see at a glance which cohorts actually moved.
Split math/code into two rows.

### 4. Cohort geometry (if we do embeddings/PCA)
2D PCA scatter of all tasks, colored by cohort. Shows whether cohorts are
genuinely distinct or overlapping. Sells the data-quality story visually —
especially if `synthetic_claude` lands in its own cluster.

## Nice-to-haves (only if time)
- A single big-number header: "Best metric: self-consistency gap, ρ=0.NN
  across both domains." The one-line takeaway.
- Hover tooltips showing example tasks from each cohort.
- A toggle to show/hide the code domain so we can present math-only first,
  then reveal the cross-domain transfer as the "and it generalizes" beat.

## Build order
1. Stub `dashboard.py` now (GPU off) against a hand-written fake
   `results/*.jsonl` with made-up numbers. Get all 4 panels rendering.
2. When real summaries land, just drop them in `results/`. Zero code change.
3. `fig.write_html()` each panel into a `report/` dir as the static backup.

Keeping the dashboard reading from files (not from live training) means it's
completely decoupled from the GPU work — build it while training runs.
