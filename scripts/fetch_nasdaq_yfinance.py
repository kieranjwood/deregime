#!/usr/bin/env python3
"""Fetch the Nasdaq Composite close series used by the paper.

This script downloads ticker ``^IXIC`` from Yahoo Finance via the optional
``yfinance`` dependency and writes the long-format file expected by the run
settings:

    date,cols,data

The data provider terms are separate from the code license.  Use this script to
recreate the file when redistribution of a processed CSV is not appropriate.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="^IXIC")
    parser.add_argument("--start", default="1990-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--out", type=Path, default=Path("data/NASDAQ_1990_2025.csv"))
    parser.add_argument(
        "--column",
        default="Close",
        choices=["Close", "Adj Close", "Open", "High", "Low", "Volume"],
        help="Yahoo Finance column to export. The paper uses Close.",
    )
    args = parser.parse_args()

    try:
        import yfinance as yf
    except ImportError as exc:
        raise SystemExit(
            "Missing optional dependency 'yfinance'. Install with:\n"
            '  uv pip install --python .venv/bin/python -e ".[data]"'
        ) from exc

    print(f"Downloading {args.ticker} from {args.start} to {args.end}...")
    raw = yf.download(args.ticker, start=args.start, end=args.end, progress=False)
    if raw.empty:
        raise SystemExit("Download returned no rows.")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    if args.column not in raw.columns:
        raise SystemExit(f"Column {args.column!r} not found. Available: {list(raw.columns)}")

    out = pd.DataFrame(
        {
            "date": pd.to_datetime(raw.index).strftime("%Y-%m-%d %H:%M:%S"),
            "data": raw[args.column].astype(float).to_numpy(),
            "cols": args.column.lower().replace(" ", "_"),
        }
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(
        f"Wrote {args.out} ({len(out)} rows, "
        f"{out['date'].iloc[0]} to {out['date'].iloc[-1]})"
    )


if __name__ == "__main__":
    main()

