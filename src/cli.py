"""Shared command-line runner for scripts/*.py: argparse, config loading,
seeding, uniform error reporting and exit codes (0 = success, 1 = failure)."""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

ROOT = Path(__file__).resolve().parent.parent


def run_stage(
    stage: str,
    description: str,
    fn: Callable[[dict[str, Any], argparse.Namespace], None],
    add_args: Callable[[argparse.ArgumentParser], None] | None = None,
    argv: list[str] | None = None,
    needs_tf: bool = False,
) -> int:
    parser = argparse.ArgumentParser(prog=f"{stage}.py", description=description)
    parser.add_argument("--config", type=Path, required=True, help="Path to configs/config.yaml")
    parser.add_argument("--smoke-test", action="store_true", help="Tiny, fast run for an end-to-end check")
    if add_args:
        add_args(parser)
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return 0 if exc.code in (0, None) else 1
    t0 = time.time()
    try:
        from src import load_config, set_global_seed

        cfg = load_config(args.config)
        set_global_seed(int(cfg["project"]["seed"]), tensorflow=needs_tf)
        print(f"[{stage}] start{' (SMOKE TEST)' if args.smoke_test else ''}", flush=True)
        fn(cfg, args)
    except Exception as exc:  # noqa: BLE001 - every failure must exit with 1
        print(f"[{stage}] FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        traceback.print_exc()
        return 1
    print(f"[{stage}] OK ({time.time() - t0:.1f} s)", flush=True)
    return 0


def log(stage: str, msg: str) -> None:
    print(f"[{stage}] {msg}", flush=True)
