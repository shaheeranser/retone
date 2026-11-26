#!/usr/bin/env bash
# Build llama.cpp, download the private merged model from Hugging Face, and
# produce the GGUF quantizations retone serves.
#
# Verbose by design: every step prints what it is about to run and its result.
# Set TRACE=1 to also echo every shell command.
#
# Usage (from anywhere; run it with bash):
#   bash scripts/build_gguf.sh
#
# Environment (all optional):
#   MERGED_REPO   private merged HF model    (default shaheeranser/retone-qwen-vl3b-merged)
#   LLAMA_CPP     llama.cpp checkout          (default $HOME/llama.cpp)
#   OUT_DIR       where the GGUF files land   (default $HOME/.retone/model)
#   WORK_DIR      download + converter venv   (default $HOME/.retone/work)
#   MERGED_DIR    local merged model dir      (default $WORK_DIR/merged)
#   PYTHON        interpreter with retone     (default python3)
#   CONVERT_PY    python for the converter    (default $WORK_DIR/venv/bin/python)
#   JOBS          build parallelism           (default nproc)
#   HF_TOKEN      required for the private repo; read from .env when present
#   TRACE=1       echo every shell command
#
# The converter runs in its own venv with llama.cpp's conversion requirements,
# so the project's .venv is not polluted with torch/transformers.
set -euo pipefail

REPO_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
MERGED_REPO=${MERGED_REPO:-shaheeranser/retone-qwen-vl3b-merged}
LLAMA_CPP=${LLAMA_CPP:-$HOME/llama.cpp}
OUT_DIR=${OUT_DIR:-$HOME/.retone/model}
WORK_DIR=${WORK_DIR:-$HOME/.retone/work}
MERGED_DIR=${MERGED_DIR:-$WORK_DIR/merged}
PYTHON=${PYTHON:-python3}
JOBS=${JOBS:-$(nproc 2>/dev/null || echo 4)}
export MERGED_REPO MERGED_DIR

log() { printf '\n== %s ==\n' "$*"; }
step() { printf '+ %s\n' "$*"; "$@"; }
need() { command -v "$1" >/dev/null 2>&1 || { echo "missing required tool: $1" >&2; exit 2; }; }

# Read KEY=VALUE lines without overriding the environment. xtrace is paused so a
# token never appears in the trace.
load_env() {
  local file="$1" line key value
  [ -f "$file" ] || return 0
  local traced=0
  case $- in *x*) traced=1; set +x ;; esac
  while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in ''|'#'*) continue ;; esac
    line="${line#export }"
    key="${line%%=*}"
    value="${line#*=}"
    key=$(printf '%s' "$key" | tr -d '[:space:]')
    [ -n "$key" ] || continue
    case "$value" in
      \"*\") value="${value%\"}"; value="${value#\"}" ;;
      \'*\') value="${value%\'}"; value="${value#\'}" ;;
    esac
    if [ -z "${!key:-}" ]; then
      export "$key=$value"
    fi
  done < "$file"
  [ "$traced" = 1 ] && set -x || true
}

load_env "$PWD/.env"
load_env "$REPO_ROOT/.env"

[ -n "${HF_TOKEN:-}" ] || { echo "HF_TOKEN is not set; put it in .env or the environment" >&2; exit 2; }
need git
need cmake
need "$PYTHON"
"$PYTHON" -c "import retone" >/dev/null 2>&1 || {
  echo "retone is not importable with $PYTHON; activate the project .venv or set PYTHON=..." >&2
  exit 2
}

CONVERT_PY=${CONVERT_PY:-}
if [ -z "$CONVERT_PY" ]; then
  CONVERT_PY=$WORK_DIR/venv/bin/python
  MANAGE_VENV=1
else
  MANAGE_VENV=0
fi

if [ "${TRACE:-0}" = "1" ]; then
  set -x
fi

log "Configuration"
cat <<EOF
  repo root     : $REPO_ROOT
  merged repo   : $MERGED_REPO (private)
  merged dir    : $MERGED_DIR
  llama.cpp     : $LLAMA_CPP
  output dir    : $OUT_DIR
  converter py  : $CONVERT_PY ($([ "$MANAGE_VENV" = 1 ] && echo managed venv || echo user-provided))
  retone python : $PYTHON
  jobs          : $JOBS
EOF

log "1/6 llama.cpp checkout"
if [ -d "$LLAMA_CPP/.git" ]; then
  echo "  using existing checkout: $(git -C "$LLAMA_CPP" rev-parse --short HEAD)"
else
  step git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA_CPP"
fi

log "2/6 Build llama-quantize and llama-server"
if [ -x "$LLAMA_CPP/build/bin/llama-quantize" ]; then
  echo "  llama-quantize already built"
else
  step cmake -S "$LLAMA_CPP" -B "$LLAMA_CPP/build" -DCMAKE_BUILD_TYPE=Release
  step cmake --build "$LLAMA_CPP/build" --config Release \
    --target llama-quantize llama-server -j "$JOBS"
fi
[ -x "$LLAMA_CPP/build/bin/llama-quantize" ] || {
  echo "llama-quantize was not produced at $LLAMA_CPP/build/bin/llama-quantize" >&2
  exit 1
}

log "3/6 Converter environment"
if [ "$MANAGE_VENV" = 1 ] && [ ! -x "$CONVERT_PY" ]; then
  step "$PYTHON" -m venv "$WORK_DIR/venv"
  requirements="$LLAMA_CPP/requirements/requirements-convert_hf_to_gguf.txt"
  [ -f "$requirements" ] || requirements="$LLAMA_CPP/requirements.txt"
  if [ -f "$requirements" ]; then
    step "$CONVERT_PY" -m pip install --upgrade pip
    step "$CONVERT_PY" -m pip install -r "$requirements"
  else
    echo "  no conversion requirements file; assuming the converter imports work"
  fi
else
  echo "  reusing converter python: $CONVERT_PY"
fi
"$CONVERT_PY" -c "import huggingface_hub" >/dev/null 2>&1 || {
  echo "$CONVERT_PY cannot import huggingface_hub; delete ${WORK_DIR}/venv to rebuild it" >&2
  exit 2
}

log "4/6 Download merged model (private; token from the environment)"
mkdir -p "$MERGED_DIR"
printf '+ huggingface_hub.snapshot_download(repo_id=%s, local_dir=%s)\n' "$MERGED_REPO" "$MERGED_DIR"
"$CONVERT_PY" - <<'PY'
import os
from huggingface_hub import snapshot_download

print(f"downloading {os.environ['MERGED_REPO']} -> {os.environ['MERGED_DIR']}")
path = snapshot_download(
    repo_id=os.environ["MERGED_REPO"],
    local_dir=os.environ["MERGED_DIR"],
    token=os.environ["HF_TOKEN"],
)
print("download complete:", path)
PY

log "5/6 Convert to GGUF and quantize (Q4_K_M, Q8_0; mmproj f16)"
step "$PYTHON" -m retone.train.quantize \
  --model "$MERGED_DIR" \
  --llama-cpp "$LLAMA_CPP" \
  --quantize-tool "$LLAMA_CPP/build/bin/llama-quantize" \
  --python "$CONVERT_PY" \
  --out "$OUT_DIR"

log "6/6 Verify artifacts"
step ls -lh "$OUT_DIR"
missing=0
for name in model-f16.gguf model-Q4_K_M.gguf model-Q8_0.gguf; do
  if [ ! -s "$OUT_DIR/$name" ]; then
    echo "MISSING: $OUT_DIR/$name" >&2
    missing=1
  fi
done
if [ ! -s "$OUT_DIR/mmproj-f16.gguf" ]; then
  echo "WARNING: $OUT_DIR/mmproj-f16.gguf is absent; the converter may have named" >&2
  echo "         the projector differently. Adjust and keep it at f16 for serving." >&2
fi
[ "$missing" = 0 ] || exit 1

log "Done"
cat <<EOF
Serve the result with:
  MODEL_DIR=$OUT_DIR scripts/serve.sh --port 8080
Then:
  retone photo.jpg --server-url http://127.0.0.1:8080
EOF
