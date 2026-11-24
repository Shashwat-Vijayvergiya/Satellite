#!/usr/bin/env bash
# Run every pipeline stage in order; stop at the first failure (Linux / macOS).
# Stages: inspect_rasters -> prepare_dataset -> split_dataset -> create_patches
#         -> train -> evaluate -> predict
# Usage:  ./run_pipeline.sh [--smoke-test] [--config configs/config.yaml] [--python python]
set -u
cd "$(dirname "$0")"
SMOKE=""
CONFIG="configs/config.yaml"
# Interpreter: --python / $PYTHON if given, else the project venv if present, else `python`.
if [ -n "${PYTHON:-}" ]; then PY="$PYTHON"
elif [ -x ".venv/bin/python" ]; then PY=".venv/bin/python"
else PY="python"; fi
while [ $# -gt 0 ]; do
  case "$1" in
    --smoke-test) SMOKE="--smoke-test" ;;
    --config) CONFIG="$2"; shift ;;
    --python) PY="$2"; shift ;;
    *) echo "[pipeline] unknown argument: $1" >&2; exit 1 ;;
  esac
  shift
done
# Must actually run and import the core stack (a pyenv shim can exist without a usable Python).
"$PY" -c "import numpy, rasterio, yaml" >/dev/null 2>&1 || {
  echo "[pipeline] FAILED: '$PY' cannot run or lacks the requirements. Create the venv (see README) or pass --python." >&2; exit 1; }
echo "[pipeline] python: $("$PY" -c 'import sys; print(sys.executable, sys.version.split()[0])')"
[ -f "$CONFIG" ] || { echo "[pipeline] FAILED: config not found: $CONFIG" >&2; exit 1; }
export TF_CPP_MIN_LOG_LEVEL=2

STAGES=(inspect_rasters prepare_dataset split_dataset create_patches train evaluate predict)
echo "[pipeline] mode: ${SMOKE:-full} | config: $CONFIG"
i=0
for stage in "${STAGES[@]}"; do
  i=$((i + 1))
  echo
  echo "[pipeline] ($i/${#STAGES[@]}) running stage: $stage"
  start=$(date +%s)
  "$PY" "scripts/$stage.py" --config "$CONFIG" $SMOKE
  code=$?
  dur=$(( $(date +%s) - start ))
  if [ $code -ne 0 ]; then
    echo "[pipeline] FAILED at stage '$stage' (exit code $code) after ${dur} s. Stopping." >&2
    exit $code
  fi
  echo "[pipeline] stage '$stage' OK (${dur} s)"
done
echo
echo "[pipeline] all ${#STAGES[@]} stages completed successfully."
