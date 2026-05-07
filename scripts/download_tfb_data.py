#!/usr/bin/env python3
"""Download and extract the TFB preprocessed dataset bundle.

The standard forecasting datasets used by the paper were taken from the TFB
preprocessed Google Drive bundle.  This helper downloads that bundle with the
optional ``gdown`` dependency and extracts it into ``raw_data/tfb`` by default.

After download, run:

    .venv/bin/python scripts/prepare_datasets.py --raw_dir raw_data/tfb --out_dir data

Nasdaq is handled separately by ``scripts/fetch_nasdaq_yfinance.py`` because
the paper uses an extended ``^IXIC`` close series rather than TFB's shorter
NASDAQ benchmark file.
"""

from __future__ import annotations

import argparse
import shutil
import tarfile
import zipfile
from pathlib import Path


TFB_FILE_ID = "1vgpOmAygokoUt235piWKUjfwao6KwLv7"
TFB_URL = f"https://drive.google.com/uc?id={TFB_FILE_ID}"


def _extract_archive(archive: Path, extract_dir: Path) -> None:
    extract_dir.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(extract_dir)
        return
    if tarfile.is_tarfile(archive):
        with tarfile.open(archive) as tf:
            tf.extractall(extract_dir)
        return
    raise SystemExit(
        f"Downloaded file {archive} is not a recognised zip/tar archive. "
        "Inspect it manually or pass --no_extract."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=TFB_URL, help="Google Drive URL or file URL.")
    parser.add_argument("--archive", type=Path, default=Path("raw_data/tfb_datasets.zip"))
    parser.add_argument("--extract_dir", type=Path, default=Path("raw_data/tfb"))
    parser.add_argument("--no_extract", action="store_true")
    parser.add_argument("--force", action="store_true", help="Overwrite existing archive/extract dir.")
    args = parser.parse_args()

    try:
        import gdown
    except ImportError as exc:
        raise SystemExit(
            "Missing optional dependency 'gdown'. Install with:\n"
            '  uv pip install --python .venv/bin/python -e ".[data]"'
        ) from exc

    args.archive.parent.mkdir(parents=True, exist_ok=True)
    if args.archive.exists() and not args.force:
        print(f"[skip] Archive already exists: {args.archive}")
    else:
        if args.archive.exists():
            args.archive.unlink()
        print(f"[download] {args.url} -> {args.archive}")
        try:
            gdown.download(args.url, str(args.archive), quiet=False, fuzzy=True)
        except TypeError:
            # gdown >= 6 removed/changed the fuzzy keyword.
            gdown.download(args.url, str(args.archive), quiet=False)

    if args.no_extract:
        return

    if args.extract_dir.exists() and args.force:
        shutil.rmtree(args.extract_dir)
    if args.extract_dir.exists() and any(args.extract_dir.iterdir()) and not args.force:
        print(f"[skip] Extract directory already populated: {args.extract_dir}")
        return
    print(f"[extract] {args.archive} -> {args.extract_dir}")
    _extract_archive(args.archive, args.extract_dir)
    print("[done] Now run:")
    print(f"  .venv/bin/python scripts/prepare_datasets.py --raw_dir {args.extract_dir} --out_dir data")


if __name__ == "__main__":
    main()
