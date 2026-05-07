#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PY="${VENV_PY:-"$ROOT/.venv/bin/python"}"
OUT_DIR="${OUT_DIR:-runs/patchtst_release_check}"
LOGS_DIR="${LOGS_DIR:-logs_patchtst_release_check}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<EOF
Usage: VENV_PY=/path/to/python $0

Launches clean PatchTST release checks, one dataset per GPU:
  GPU 0 -> nasdaq
  GPU 1 -> electricity
  GPU 2 -> traffic
  GPU 3 -> exchange

Environment:
  VENV_PY   Python executable with numpy/torch/gpytorch installed
            default: $ROOT/.venv/bin/python
  OUT_DIR   output directory relative to the project root
  LOGS_DIR  log directory relative to the project root
EOF
  exit 0
fi

if [[ ! -x "$VENV_PY" ]]; then
  echo "Python executable not found or not executable: $VENV_PY" >&2
  echo "Set VENV_PY=/path/to/python if needed." >&2
  exit 1
fi

"$VENV_PY" -c "import numpy, torch, gpytorch" >/dev/null

echo "Using Python: $VENV_PY"
echo "Output dir:   $ROOT/$OUT_DIR"
echo "Logs dir:     $ROOT/$LOGS_DIR"

"$VENV_PY" "$ROOT/scripts/launch_patchtst_nasdaq.py" \
  --python "$VENV_PY" --gpu 0 --out-dir "$OUT_DIR" --logs-dir "$LOGS_DIR"

"$VENV_PY" "$ROOT/scripts/launch_patchtst_electricity.py" \
  --python "$VENV_PY" --gpu 1 --out-dir "$OUT_DIR" --logs-dir "$LOGS_DIR"

"$VENV_PY" "$ROOT/scripts/launch_patchtst_traffic.py" \
  --python "$VENV_PY" --gpu 2 --out-dir "$OUT_DIR" --logs-dir "$LOGS_DIR"

"$VENV_PY" "$ROOT/scripts/launch_patchtst_exchange.py" \
  --python "$VENV_PY" --gpu 3 --out-dir "$OUT_DIR" --logs-dir "$LOGS_DIR"

echo "Launched clean PatchTST release checks:"
echo "  GPU 0 -> nasdaq"
echo "  GPU 1 -> electricity"
echo "  GPU 2 -> traffic"
echo "  GPU 3 -> exchange"
