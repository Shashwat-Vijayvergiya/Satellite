"""Versioned run directories and lineage checks.

Every data or model stage writes into a fresh, timestamped run directory and
finishes by writing metadata.json with status "complete", then updates
<base>/latest.json. Downstream stages resolve their inputs only through
latest.json and check lineage (e.g. the patch run must come from the current
split run, which must come from the current prep run). Stale or half-written
outputs are therefore never used silently.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def jsonable(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(x) for x in o]
    if isinstance(o, Path):
        return str(o).replace("\\", "/")
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, float) and (o != o):  # NaN -> null (valid JSON)
        return None
    return o


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(jsonable(obj), indent=2), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def new_run_dir(base: Path, prefix: str) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run = base / f"{prefix}_{stamp}"
    i = 1
    while run.exists():
        run = base / f"{prefix}_{stamp}_{i}"
        i += 1
    run.mkdir()
    return run


def rel(path: Path, root: Path) -> str:
    return os.path.relpath(path, root).replace("\\", "/")


def finish_run(run_dir: Path, root: Path, meta: dict[str, Any], latest_name: str = "latest.json") -> None:
    """Write metadata.json (status complete) and point <base>/<latest_name> at this run."""
    meta = {"run_id": run_dir.name, "status": "complete", "completed_utc": utc_now(), **meta}
    write_json(run_dir / "metadata.json", meta)
    write_json(run_dir.parent / latest_name, {"run_id": run_dir.name, "run_dir": rel(run_dir, root),
                                               "completed_utc": meta["completed_utc"]})


def resolve_latest(base: Path, root: Path, what: str, latest_name: str = "latest.json") -> tuple[Path, dict[str, Any]]:
    """Return (run_dir, metadata) of the latest COMPLETE run under base, or raise RuntimeError."""
    pointer = base / latest_name
    if not pointer.is_file():
        raise RuntimeError(f"No {what} run found ({rel(pointer, root)} missing). Run the {what} stage first.")
    info = read_json(pointer)
    return load_run(root / info["run_dir"], what)


def load_run(run_dir: Path, what: str) -> tuple[Path, dict[str, Any]]:
    meta_path = run_dir / "metadata.json"
    if not meta_path.is_file():
        raise RuntimeError(f"{what} run {run_dir.name} has no metadata.json (incomplete run?)")
    meta = read_json(meta_path)
    if meta.get("status") != "complete":
        raise RuntimeError(f"{what} run {run_dir.name} is not complete (status={meta.get('status')})")
    return run_dir, meta


def require_parent(child_meta: dict[str, Any], key: str, parent_run_id: str, child_what: str, parent_what: str) -> None:
    """Raise if child_meta[key] != parent_run_id (the child was built from an older parent run)."""
    got = child_meta.get(key)
    if got != parent_run_id:
        raise RuntimeError(
            f"Stale {child_what}: it was built from {parent_what} run {got!r}, but the latest {parent_what} "
            f"run is {parent_run_id!r}. Re-run the {child_what} stage."
        )


def data_lineage(cfg: dict[str, Any], root: Path, upto: str) -> dict[str, Any]:
    """Resolve and cross-check the current data runs: validation -> prep -> split -> patches.

    upto: "prep", "split" or "patches". Raises RuntimeError on any stale link.
    """
    from src.validation import require_validated_raw_data

    stages = ["prep", "split", "patches"]
    if upto not in stages:
        raise ValueError(f"upto must be one of {stages}")
    p = cfg["paths"]
    out: dict[str, Any] = {"validation": require_validated_raw_data(cfg, root)}
    prep_run, prep_meta = resolve_latest(root / p["processed_dir"], root, "prepare_dataset")
    if prep_meta.get("raw_sha256") != out["validation"]["sha256"]:
        raise RuntimeError(f"Stale prepare_dataset run {prep_run.name}: raw files differ from the ones it was built from. Re-run prepare_dataset.")
    out["prep"] = (prep_run, prep_meta)
    if upto == "prep":
        return out
    split_run, split_meta = resolve_latest(root / p["splits_dir"], root, "split_dataset")
    require_parent(split_meta, "prep_run_id", prep_run.name, "split_dataset", "prepare_dataset")
    out["split"] = (split_run, split_meta)
    if upto == "split":
        return out
    patch_run, patch_meta = resolve_latest(root / p["patches_dir"], root, "create_patches")
    require_parent(patch_meta, "split_run_id", split_run.name, "create_patches", "split_dataset")
    out["patches"] = (patch_run, patch_meta)
    return out


def fingerprint(*parts: Any) -> str:
    """Stable SHA-256 of the inputs that determine a data stage's output."""
    import hashlib

    blob = json.dumps(jsonable(list(parts)), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def reusable_run(base: Path, root: Path, fp: str) -> Path | None:
    """Latest complete run under base if it was built with the same fingerprint, else None."""
    try:
        run_dir, meta = resolve_latest(base, root, "stage")
    except RuntimeError:
        return None
    return run_dir if meta.get("fingerprint") == fp else None
