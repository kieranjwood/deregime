#!/usr/bin/env python3
"""Small background launcher for release PatchTST settings."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def launch(dataset: str, argv: list[str] | None = None) -> int:
    root = _repo_root()
    settings = root / "run_settings" / "patchtst" / f"{dataset}.json"
    if not settings.exists():
        raise FileNotFoundError(f"Missing PatchTST settings file: {settings}")

    parser = argparse.ArgumentParser(
        description=f"Launch PatchTST release runs for {dataset}."
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use. Defaults to the current interpreter.",
    )
    parser.add_argument(
        "--gpu",
        default=None,
        help="GPU id to expose via CUDA_VISIBLE_DEVICES. Omit to inherit.",
    )
    parser.add_argument(
        "--out-dir",
        default="runs/patchtst_release_check",
        help="Output directory, relative to the project root unless absolute.",
    )
    parser.add_argument(
        "--logs-dir",
        default="logs_patchtst_release_check",
        help="Log directory, relative to the project root unless absolute.",
    )
    parser.add_argument(
        "--foreground",
        action="store_true",
        help="Run in the foreground instead of launching a detached process.",
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir

    logs_dir = Path(args.logs_dir)
    if not logs_dir.is_absolute():
        logs_dir = root / logs_dir
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"patchtst_{dataset}.log"

    env = os.environ.copy()
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    env["MPLCONFIGDIR"] = env.get("MPLCONFIGDIR", str(root / ".matplotlib_cache"))
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    python_cmd = args.python
    path_separators = [os.sep]
    if os.altsep is not None:
        path_separators.append(os.altsep)
    if any(sep in python_cmd for sep in path_separators):
        python_path = Path(python_cmd)
        if not python_path.is_absolute():
            python_path = Path.cwd() / python_path
        # Preserve virtual-environment symlinks.  Path.resolve() follows
        # .venv*/bin/python -> /bin/python3 on some systems, which drops the
        # venv site-packages and makes imports such as numpy fail.
        python_cmd = str(python_path.absolute())

    cmd = [
        python_cmd,
        "-u",
        "-m",
        "deregime.run",
        "--experiments_json",
        str(settings.relative_to(root)),
        "--out_dir",
        str(out_dir),
    ]

    if args.foreground:
        return subprocess.run(cmd, cwd=root, env=env, check=False).returncode

    log_fh = log_path.open("ab")
    log_fh.write(
        (
            "\n"
            f"===== Launch PatchTST {dataset} =====\n"
            f"python={python_cmd}\n"
            f"cwd={root}\n"
            f"cmd={' '.join(cmd)}\n"
            "====================================\n"
        ).encode("utf-8")
    )
    log_fh.flush()
    proc = subprocess.Popen(
        cmd,
        cwd=root,
        env=env,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    print(f"Launched PatchTST {dataset}: pid={proc.pid}")
    print(f"  log: {log_path}")
    print(f"  out: {out_dir}")
    if args.gpu is not None:
        print(f"  CUDA_VISIBLE_DEVICES={args.gpu}")
    return 0
