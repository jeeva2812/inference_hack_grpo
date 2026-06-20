#!/usr/bin/env bash
# setup.sh — one-shot environment setup for a fresh GPU box (CODE track).
# Captures the fixes found while bringing the code track up on Prime Intellect:
#   - python3-venv may be missing on the base image (venv silently fails)
#   - scripts call `python` (not `python3`) -> need it on PATH
#   - base image shipped jinja2 3.0.3, but apply_chat_template needs >=3.1.0
#   - vllm is skipped on purpose (use_vllm=False everywhere; heavy/slow install)
#
# Usage (run from the repo root):
#   bash setup.sh
#   # then, for cloud logging of training curves:
#   wandb login            # paste key from wandb.ai/authorize
#   export REPORT_TO=wandb WANDB_PROJECT=grpo-cohorts
#   bash run_code_track.sh check   # ... inference / grpo1 / phase1 / phase2
set -euo pipefail

PKGS="torch transformers trl datasets accelerate numpy matplotlib wandb"

echo "== 1/4: python venv =="
if python3 -m venv --help >/dev/null 2>&1 && python3 -c "import ensurepip" >/dev/null 2>&1; then
  python3 -m venv venv
  # shellcheck disable=SC1091
  source venv/bin/activate
  echo "   venv created and activated -> $(which python)"
else
  echo "   python3-venv/ensurepip unavailable; installing system-wide instead."
  echo "   (to use a venv: sudo apt-get install -y python3.10-venv, then re-run)"
  # scripts call `python`; make sure it resolves to python3
  if ! command -v python >/dev/null 2>&1; then
    ln -sf "$(command -v python3)" /usr/local/bin/python
    echo "   symlinked python -> $(command -v python3)"
  fi
fi

echo "== 2/4: pip deps (no vllm) =="
python -m pip install -U pip wheel
python -m pip install $PKGS
# apply_chat_template needs jinja2>=3.1.0 (base image often ships 3.0.3)
python -m pip install "jinja2>=3.1.0"

echo "== 3/4: sanity check =="
python - <<'PY'
import torch, transformers, trl, datasets, jinja2
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "|", (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU"))
print("transformers", transformers.__version__, "| trl", trl.__version__,
      "| datasets", datasets.__version__, "| jinja2", jinja2.__version__)
assert tuple(int(x) for x in jinja2.__version__.split(".")[:2]) >= (3, 1), "jinja2 too old"
PY

echo "== 4/4: done =="
cat <<'MSG'
Environment ready. Installed in ./venv: torch, transformers, trl, datasets,
accelerate, numpy, matplotlib, wandb (jinja2>=3.1.0). vllm intentionally skipped.

!! IMPORTANT — ACTIVATE THE VENV IN EVERY NEW SHELL FIRST !!
  source venv/bin/activate
  # The scripts call `python` (not `python3`). `python` ONLY exists inside the
  # venv, so if you skip this you'll get: "python: command not found".
  # Verify with:  which python   ->  .../inference_hack_grpo/venv/bin/python

Next (run these AFTER `source venv/bin/activate`):
  wandb login                                   # cloud curves (optional)
  export REPORT_TO=wandb WANDB_PROJECT=grpo-cohorts
  tmux new -s code                              # survive SSH drops
  bash run_code_track.sh check                  # offline tests (no GPU)
  bash run_code_track.sh inference              # model load + generate + verify
  bash run_code_track.sh grpo1                  # dummy: 15-step GRPO + eval
  bash run_code_track.sh phase1                 # base eval + signals (all cohorts)
  bash run_code_track.sh phase2                 # train+eval per cohort -> summary

Run phase1 + phase2 back-to-back in the BACKGROUND (survives SSH drops):
  # -d = detached; phase2 only starts if phase1 succeeds (&&); output -> run_phase12.log
  tmux new -s code -d 'source venv/bin/activate 2>/dev/null; \
  export REPORT_TO=wandb WANDB_PROJECT=grpo-cohorts &&  bash run_code_track.sh phase1 && bash run_code_track.sh phase2 |& tee run_phase12.log'
  tmux attach -t code                           # watch live (Ctrl-b d to detach again)
  tail -f run_phase12.log                        # ...or just follow the log
  tmux ls                                        # list sessions / check it's still running

If you installed system-wide (no venv), this shell already has `python`.
If you used a venv, remember to `source venv/bin/activate` in new shells.
MSG
