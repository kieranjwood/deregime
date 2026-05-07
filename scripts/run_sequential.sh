#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 run_settings/<backbone>/<dataset>.json [out_dir]" >&2
  exit 2
fi

SETTINGS_JSON="$1"
OUT_DIR="${2:-runs/release}"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x ".venv/bin/python" ]]; then
    PYTHON_BIN=".venv/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi
export MPLCONFIGDIR="${MPLCONFIGDIR:-.matplotlib_cache}"
mkdir -p "${MPLCONFIGDIR}"

"${PYTHON_BIN}" -u -m deregime.run \
  --experiments_json "${SETTINGS_JSON}" \
  --out_dir "${OUT_DIR}"
