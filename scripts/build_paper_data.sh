#!/usr/bin/env bash
# End-to-end data preparation helper for the paper settings.
#
# Requires optional data dependencies:
#   uv pip install --python .venv/bin/python -e ".[data]"
#
# This downloads/extracts the TFB preprocessed bundle, applies the paper
# preprocessing, fetches the extended Nasdaq Composite series, and checks that
# all run-settings data files are present.

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
RAW_DIR="${RAW_DIR:-raw_data/tfb}"
OUT_DIR="${OUT_DIR:-data}"

"$PYTHON_BIN" scripts/download_tfb_data.py --extract_dir "$RAW_DIR"
"$PYTHON_BIN" scripts/prepare_datasets.py \
  --raw_dir "$RAW_DIR" \
  --out_dir "$OUT_DIR" \
  --datasets etth1 etth2 ettm1 ettm2 exchange electricity traffic illness weatherc
"$PYTHON_BIN" scripts/fetch_nasdaq_yfinance.py --out "$OUT_DIR/NASDAQ_1990_2025.csv"
"$PYTHON_BIN" scripts/check_setup.py

