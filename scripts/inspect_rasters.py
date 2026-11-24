r"""Validate the raw Earth Engine exports in data/raw/ before anything else runs.

Checks existence/readability (full read, catches truncation), split-export
tiles, band counts, dtypes, CRS, transform, bounds and exact grid alignment,
NaN/inf, no-data conventions, per-band statistics (a constant band is a hard
error), reflectance plausibility, WorldCover class IDs and shares, Hansen
codes, the loss_target definition (recomputed pixel by pixel) and the
aoi.geojson metadata. Expected values from Earth Engine diagnostics are
compared and any difference is reported.

Writes outputs/reports/validation/run_<timestamp>/validation_report.json and
updates outputs/reports/validation/latest.json (status + raw-file SHA-256).

Usage (PowerShell, from the project root):
    python scripts\inspect_rasters.py --config configs\config.yaml

Exit codes: 0 = all hard checks passed (warnings may exist), 1 = any failure.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

STAGE = "inspect_rasters"
ROOT = Path(__file__).resolve().parent.parent


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True, help="Path to configs/config.yaml")
    parser.add_argument("--smoke-test", action="store_true", help="Accepted for pipeline uniformity; validation always runs in full")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        from src import load_config
        from src.validation import validate_raw_data, write_report

        cfg = load_config(args.config)
        print(f"[{STAGE}] Validating raw data in {cfg['paths']['raw_dir']} ...", flush=True)
        rep = validate_raw_data(cfg, ROOT)
        run_dir = write_report(rep, cfg, ROOT, args.config)
    except SystemExit as exc:
        return 0 if exc.code in (0, None) else 1
    except Exception as exc:  # noqa: BLE001 - every failure must exit with 1
        print(f"[{STAGE}] FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1

    for w in rep.warnings:
        print(f"[{STAGE}] WARNING: {w}")
    for e in rep.errors:
        print(f"[{STAGE}] ERROR: {e}", file=sys.stderr)
    print(f"[{STAGE}] Report: {run_dir / 'validation_report.json'}")
    if not rep.passed:
        print(f"[{STAGE}] FAILED: {len(rep.errors)} hard error(s). Fix the data before continuing.", file=sys.stderr)
        return 1
    print(f"[{STAGE}] OK: all hard checks passed ({len(rep.warnings)} warning(s)).")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    sys.exit(main())
