#!/usr/bin/env bash
# Serve the quantized model so `retone` can reach it.
#
# The parameter JSON Schema is sent per request by the inference client as the
# response_format grammar, so the model can only emit valid JSON and the schema
# stays a single source of truth (retone.executor.json_schema). This script only
# launches the server; it does not duplicate the grammar.
#
# Required environment (or flags): MODEL_DIR holding the GGUF and mmproj files.
#   GGUF=${GGUF:-$MODEL_DIR/model-Q4_K_M.gguf}
#   MMPROJ=${MMPROJ:-$MODEL_DIR/mmproj-f16.gguf}
# Optional: HOST (127.0.0.1), PORT (8080), CTX (8192), NGL (0, CPU only),
# IMAGE_MIN_TOKENS (256). Qwen2.5-VL spends one token per 28x28 pixels, so the
# 448 px training thumbnails are ~144-256 tokens; 256 matches training and is
# the floor that keeps vision working. Forcing Qwen's 1024 grounding floor
# upscaled every image and roughly quadrupled CPU latency. The template embedded
# in the GGUF is used; forcing `--chat-template qwen2-vl` drops the image marker.
#
# Usage: scripts/serve.sh [--host HOST] [--port PORT] [extra llama-server flags...]
set -euo pipefail

HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8080}
CTX=${CTX:-8192}
NGL=${NGL:-0}
IMAGE_MIN_TOKENS=${IMAGE_MIN_TOKENS:-256}

while [ $# -gt 0 ]; do
  case "$1" in
    --host) HOST=$2; shift 2 ;;
    --port) PORT=$2; shift 2 ;;
    *) break ;;
  esac
done

: "${MODEL_DIR:?set MODEL_DIR to the folder with the GGUF and mmproj files}"
GGUF=${GGUF:-$MODEL_DIR/model-Q4_K_M.gguf}
MMPROJ=${MMPROJ:-$MODEL_DIR/mmproj-f16.gguf}

for file in "$GGUF" "$MMPROJ"; do
  [ -f "$file" ] || { echo "missing model file: $file" >&2; exit 2; }
done

exec llama-server \
  --model "$GGUF" \
  --mmproj "$MMPROJ" \
  --host "$HOST" \
  --port "$PORT" \
  --ctx-size "$CTX" \
  --n-gpu-layers "$NGL" \
  --image-min-tokens "$IMAGE_MIN_TOKENS" \
  --jinja \
  "$@"
