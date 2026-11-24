#!/usr/bin/env bash
# Host-agnostic remote training: Colab, Kaggle, or any SSH GPU box.
# Required environment: DATASET_REPO, OUTPUT_REPO, HF_TOKEN.
# Optional: REPO_URL, WORK_DIR, MODEL, plus any finetune.py flags after the script.
set -euo pipefail

: "${DATASET_REPO:?set DATASET_REPO to the private Hugging Face dataset id}"
: "${OUTPUT_REPO:?set OUTPUT_REPO to the Hugging Face model id}"
: "${HF_TOKEN:?set HF_TOKEN; never hard-code it}"

REPO_URL=${REPO_URL:-https://github.com/shaheeranser/retone.git}
WORK_DIR=${WORK_DIR:-$HOME/retone-remote}
MODEL=${MODEL:-}

if [ -d "$WORK_DIR/.git" ]; then
  git -C "$WORK_DIR" pull --ff-only
else
  git clone --depth 1 "$REPO_URL" "$WORK_DIR"
fi

cd "$WORK_DIR"
# The host normally provides a CUDA torch build; let Unsloth install without
# replacing it. If pip insists on a different torch, install Unsloth first.
python -m pip install -e ".[train]"

model_args=()
if [ -n "$MODEL" ]; then
  model_args=(--model "$MODEL")
fi

exec python -m retone.train.finetune \
  --dataset-repo "$DATASET_REPO" \
  --output-repo "$OUTPUT_REPO" \
  "${model_args[@]}" \
  "$@"
