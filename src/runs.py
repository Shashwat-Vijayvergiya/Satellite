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


def prep_fingerprint(cfg: dict[str, Any], raw_sha256: dict[str, Any]) -> str:
    return fingerprint(raw_sha256, cfg["grid"], cfg["s2"], cfg["hansen"], cfg["worldcover"])


def split_fingerprint(cfg: dict[str, Any], prep_run_id: str) -> str:
    return fingerprint(prep_run_id, cfg["split"], cfg["patching"])


def patches_fingerprint(cfg: dict[str, Any], prep_run_id: str, split_run_id: str) -> str:
    parts: list[Any] = [prep_run_id, split_run_id, cfg["patching"], cfg["s2"]["normalization"]]
    tp = cfg.get("train_patches", {})
    # The default (no extra overlapping train patches) is left out so that
    # fingerprints of runs built before this option existed stay valid.
    if int(tp.get("stride", cfg["patching"]["patch_size"])) != int(cfg["patching"]["patch_size"]):
        parts.append({"train_patches": tp})
    return fingerprint(*parts)


def fingerprint(*parts: Any) -> str:
    """Stable SHA-256 of the inputs that determine a data stage's output."""
    import hashlib

    blob = json.dumps(jsonable(list(parts)), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def reusable_run(base: Path, root: Path, fp: str) -> Path | None:
    """Newest COMPLETE run under base built with fingerprint fp (any run, not only the latest), else None."""
    if not base.is_dir():
        return None
    for run_dir in sorted((d for d in base.iterdir() if d.is_dir()), key=lambda d: d.name, reverse=True):
        meta_path = run_dir / "metadata.json"
        if not meta_path.is_file():
            continue
        meta = read_json(meta_path)
        if meta.get("status") == "complete" and meta.get("fingerprint") == fp:
            return run_dir
    return None


def point_latest(run_dir: Path, root: Path, latest_name: str = "latest.json") -> None:
    meta = read_json(run_dir / "metadata.json")
    write_json(run_dir.parent / latest_name, {"run_id": run_dir.name, "run_dir": rel(run_dir, root),
                                               "completed_utc": meta.get("completed_utc")})


def data_lineage(cfg: dict[str, Any], root: Path, upto: str) -> dict[str, Any]:
    """Find the data runs that match THIS config: validation -> prep -> split -> patches.

    Each run is located by its content fingerprint (raw-file hashes, parent run
    ID and the config sections it depends on), so several configurations can
    coexist and a model always resolves to the exact data it was built for.
    upto: "prep", "split" or "patches". Raises RuntimeError if a stage has not
    been run for this configuration.
    """
    from src.validation import require_validated_raw_data

    stages = ["prep", "split", "patches"]
    if upto not in stages:
        raise ValueError(f"upto must be one of {stages}")
    p = cfg["paths"]
    out: dict[str, Any] = {"validation": require_validated_raw_data(cfg, root)}

    def find(base: Path, fp: str, what: str) -> tuple[Path, dict[str, Any]]:
        run = reusable_run(base, root, fp)
        if run is None:
            raise RuntimeError(f"No {what} run matches the current raw data and config. Run the {what} stage first.")
        return load_run(run, what)

    out["prep"] = find(root / p["processed_dir"], prep_fingerprint(cfg, out["validation"]["sha256"]), "prepare_dataset")
    if upto == "prep":
        return out
    prep_id = out["prep"][0].name
    out["split"] = find(root / p["splits_dir"], split_fingerprint(cfg, prep_id), "split_dataset")
    if upto == "split":
        return out
    out["patches"] = find(root / p["patches_dir"], patches_fingerprint(cfg, prep_id, out["split"][0].name), "create_patches")
    return out
