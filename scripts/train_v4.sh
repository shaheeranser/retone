#!/usr/bin/env bash
# v4 pilot-then-full run on one 24 GB GPU (ModelScope A10). Needs HF_TOKEN in the environment.
#   bash scripts/train_v4.sh pilot   # 400 steps, ~minutes: read the loss and the `collapsed` flag
#   bash scripts/train_v4.sh full    # only if the pilot loss clearly drops below ~0.1
# Why a pilot: v3 sat at ~0.17 loss (the label prior) for a whole epoch, so check cheaply first.
set -euo pipefail
STEPS=$([ "${1:-pilot}" = full ] && echo 3000 || echo 400)
pip install -e ".[train]"
python -m retone.train.finetune \
  --dataset-repo "${DATASET_REPO:-shaheeranser/retone-dataset-v4}" \
  --output-repo "${OUTPUT_REPO:-shaheeranser/retone-qwen-vl3b-v4}" \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --output-dir outputs/v4 \
  --batch-size 4 --grad-accum 2 \
  --max-steps "$STEPS" --save-steps 200 \
  --sample-size 16 --max-attempts 1 --resume
