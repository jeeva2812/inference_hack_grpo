# Prime Intellect runbook

Practical pointers for getting results off the rented box without burning
credits or losing work. Adapt as the actual UI differs.

## Provisioning
- Pick the **cheapest GPU that fits**: 1×A100-80GB is plenty for a 1.5B
  model + GRPO. Don't grab an H100 unless A100 is unavailable — you're
  paying per hour.
- Choose a base image with **CUDA + PyTorch preinstalled** if offered; saves
  10+ min of `pip` each boot.
- Note the **hourly rate** before you click. A100 ≈ $1.5–2/hr. Budget mental
  math: $100 / $2 = 50 GPU-hours total across the whole team. We need ~6.

## First-boot setup (do once, fast)
```bash
git clone https://github.com/jeeva2812/inference_hack_grpo.git
cd inference_hack_grpo
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# vLLM is heavy; if it stalls the install, comment it out and use HF generate.
huggingface-cli login        # paste a token -> avoids the rate-limit warning
wandb login                  # paste API key from wandb.ai/authorize (see below)
```

## Experiment tracking with Weights & Biases
The box is ephemeral, so don't rely on terminal scrollback. W&B streams every
training metric to a cloud dashboard that survives the box dying and lets both
teammates watch the same runs (from a phone, even).

One-time, on the box:
```bash
wandb login                              # paste your API key
```
Then enable it per run via env vars (no code edits — see grpo_baseline.py):
```bash
export REPORT_TO=wandb
export WANDB_PROJECT=grpo-cohorts
export RUN_NAME=math_medium_pass         # change per cohort so runs are labelled
python grpo_baseline.py
```
Leaving `REPORT_TO` unset = terminal-only (safe default, never breaks).

## Never lose a long run — use tmux
The cardinal rule: **never run training in a raw SSH shell.** If your laptop
sleeps or wifi drops, the SSH session dies and takes the job with it.

```bash
tmux new -s train          # start a named session
# ... launch python grpo_baseline.py inside it ...
# detach with: Ctrl-b then d   (job keeps running)
tmux attach -t train       # reattach later, even after reconnecting SSH
```

Run `claude` inside tmux too, so your whole working context survives a drop.

## Don't pay for an idle GPU
- The box bills wall-clock, not utilization. An idle box with you editing
  code still costs money.
- **Shut it down** between GPU batches (see GPU discipline in GAMEPLAN).
- Before shutdown, **pull results off the box** — instances are usually
  ephemeral; storage may be wiped on stop.

## Getting results OFF the box (the part people forget)
Three options, easiest first:

### A. Push to the repo (best for small files)
Logs, summary JSONLs, plots, configs — commit and push from the box.
```bash
git add results/ *.jsonl
git commit -m "math track results"
git push
```
Then pull them on your laptop / the dashboard reads from the repo. Add a
`.gitignore` so you don't accidentally commit multi-GB checkpoints:
```
outputs/
*.safetensors
*.bin
__pycache__/
venv/
```

### B. scp for medium files (plots, sample completions)
From your **laptop**:
```bash
scp -P <port> root@<host>:~/inference_hack_grpo/results/*.png ./local_results/
```

### C. Keep checkpoints on the box, only export numbers
Model checkpoints are huge and we don't need them after eval. Run
train -> eval -> write `lift` to a JSONL **on the box**, and only export the
tiny summary. The checkpoint can die with the instance.

## Suggested workflow per GPU session
1. Boot box, `tmux attach` (or new).
2. `git pull` to get latest scripts.
3. Run the queued batch (extraction OR training+eval) — all of it, back to
   back. Don't boot a GPU for one script.
4. Write results to `results/*.jsonl`.
5. `git add results/ && git commit && git push`.
6. Detach tmux, **stop the instance.**
7. Build/iterate dashboard + analysis locally with GPU off.

## Cost sanity check
**Each teammate has their own $100 plan** — so the budgets are independent,
not shared. Run two A100s in parallel, one per person.

Per-person budget (your own track only):
| Activity | GPU-hrs | $ at $2/hr |
|---|---|---|
| Signal extraction (your track) | ~1.5 | $3 |
| 3 training runs + eval | ~3 | $6 |
| Retries / debugging buffer | ~3 | $6 |
| **Total per person** | **~7.5** | **~$15** |

That leaves ~$85 each — plenty for v2 cohorts (`long_context`,
`synthetic_claude`), seed repeats for noise control, or longer training.
The only real risk is **leaving a box running overnight**, not the runs
themselves. Set a phone reminder to confirm the instance is stopped before
you sleep.
