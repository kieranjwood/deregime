#!/usr/bin/env python3
"""Check that the release environment and processed data are ready."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path


REQUIRED_MODULES = [
    "numpy",
    "pandas",
    "torch",
    "gpytorch",
    "sklearn",
    "matplotlib",
    "tqdm",
    "scipy",
]


def check_imports() -> list[str]:
    missing: list[str] = []
    for name in REQUIRED_MODULES:
        try:
            importlib.import_module(name)
        except Exception as exc:  # pragma: no cover - diagnostic path
            missing.append(f"{name}: {exc}")
    return missing


def referenced_data(settings_root: Path) -> set[Path]:
    files: set[Path] = set()
    for path in sorted(settings_root.rglob("*.json")):
        try:
            experiments = json.loads(path.read_text())
        except Exception as exc:
            raise RuntimeError(f"Could not parse {path}: {exc}") from exc
        for exp in experiments:
            overrides = exp.get("overrides", {})
            file_name = overrides.get("file")
            if file_name:
                files.add(Path(file_name))
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings-root", type=Path, default=Path("run_settings"))
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    os.environ.setdefault("MPLCONFIGDIR", str(repo_root / ".matplotlib_cache"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    missing_imports = check_imports()
    if missing_imports:
        print("[fail] Missing Python dependencies:")
        for item in missing_imports:
            print(f"  - {item}")
        print("\nInstall with:")
        print("  uv venv .venv")
        print("  uv pip install --python .venv/bin/python -e .")
        return 1

    print("[ok] Python dependencies import successfully.")

    data_files = referenced_data(repo_root / args.settings_root)
    missing_data = [path for path in sorted(data_files) if not (repo_root / path).exists()]
    if missing_data:
        print("[fail] Missing processed data files:")
        for path in missing_data:
            print(f"  - {path}")
        print("\nPrepare data with:")
        print('  uv pip install --python .venv/bin/python -e ".[data]"')
        print("  PYTHON_BIN=.venv/bin/python scripts/build_paper_data.sh")
        print("\nOr prepare manually with:")
        print("  .venv/bin/python scripts/download_tfb_data.py --extract_dir raw_data/tfb")
        print("  .venv/bin/python scripts/prepare_datasets.py --raw_dir raw_data/tfb --out_dir data")
        print("  .venv/bin/python scripts/fetch_nasdaq_yfinance.py --out data/NASDAQ_1990_2025.csv")
        print("\nOr, if you already have processed paper-format CSVs:")
        print("  ln -s /absolute/path/to/processed/data data")
        return 2

    print("[ok] All processed data files referenced by run_settings are present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
