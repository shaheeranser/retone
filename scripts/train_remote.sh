#!/usr/bin/env bash
# Host-agnostic remote training: Colab, Kaggle, or any SSH GPU box.
# Reads .env from the current directory or the repo root if present.
# Required environment: DATASET_REPO, OUTPUT_REPO, HF_TOKEN.
# Optional: REPO_URL, WORK_DIR, MODEL, plus any finetune.py flags after the script.
set -euo pipefail

REPO_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
load_env() {
  if [ -f "$1" ]; then
    set -a
    # shellcheck disable=SC1090
    . "$1"
    set +a
  fi
}

load_env "$PWD/.env"
load_env "$REPO_ROOT/.env"

REPO_URL=${REPO_URL:-https://github.com/shaheeranser/retone.git}
WORK_DIR=${WORK_DIR:-$HOME/retone-remote}
MODEL=${MODEL:-}

if [ -d "$WORK_DIR/.git" ]; then
  git -C "$WORK_DIR" pull --ff-only
else
  git clone --depth 1 "$REPO_URL" "$WORK_DIR"
fi

cd "$WORK_DIR"
load_env "$WORK_DIR/.env"
load_env "$PWD/.env"

: "${DATASET_REPO:?set DATASET_REPO to the private Hugging Face dataset id}"
: "${OUTPUT_REPO:?set OUTPUT_REPO to the Hugging Face model id}"
: "${HF_TOKEN:?set HF_TOKEN; never hard-code it}"

case "$HF_TOKEN" in
  hf_replace_me) echo "replace the placeholder HF_TOKEN in .env" >&2; exit 2 ;;
esac
case "$DATASET_REPO" in
  your-username/*) echo "replace the placeholder DATASET_REPO in .env" >&2; exit 2 ;;
esac
case "$OUTPUT_REPO" in
  your-username/*) echo "replace the placeholder OUTPUT_REPO in .env" >&2; exit 2 ;;
esac

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
