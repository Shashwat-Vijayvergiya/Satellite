"""Strict validation of the raw Earth Engine exports in data/raw/.

Runs before every other stage. Any hard error makes the caller exit with a
nonzero code. Deviations from the expected values in config.validation (which
come from Earth Engine diagnostics) are reported as warnings: the real files
are trusted over those numbers.

The report records SHA-256 hashes of every raw file. Later stages compare
these against the current files and refuse to run on data that was never
validated or has changed since.
"""

from __future__ import annotations

import datetime as _dt
import glob
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

RASTER_KEYS = ("s2_pre", "s2_post", "worldcover", "hansen")


@dataclass
class Report:
    """Collects hard errors, warnings and measured facts."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    facts: dict[str, Any] = field(default_factory=dict)

    def error(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    @property
    def passed(self) -> bool:
        return not self.errors


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def find_tiles(path: Path) -> list[Path]:
    """Earth Engine split-export siblings, e.g. name-0000000000-0000000000.tif or name_r0_c1.tif."""
    stem, parent = path.stem, path.parent
    patterns = [f"{stem}-*.tif", f"{stem}_r*_c*.tif", f"{stem}[0-9]*.tif"]
    found: set[Path] = set()
    for pat in patterns:
        for p in glob.glob(str(parent / glob.escape(stem)) + pat[len(stem):]):
            if Path(p) != path:
                found.add(Path(p))
    return sorted(found)


def expected_transform(grid: dict[str, Any]) -> tuple[float, ...]:
    px = float(grid["pixel_size_m"])
    return (px, 0.0, float(grid["x0"]), 0.0, -px, float(grid["ytop"]))


def band_stats(a: np.ndarray) -> dict[str, float]:
    if a.size == 0:
        return {"count": 0}
    a64 = a.astype(np.float64, copy=False)
    p1, p50, p99 = np.percentile(a, [1, 50, 99])
    return {
        "count": int(a.size),
        "min": float(a.min()),
        "max": float(a.max()),
        "mean": float(a64.mean()),
        "std": float(a64.std()),
        "p1": float(p1),
        "p50": float(p50),
        "p99": float(p99),
    }


def value_counts(a: np.ndarray) -> dict[int, int]:
    counts = np.bincount(a.ravel(), minlength=1) if a.dtype == np.uint8 else None
    if counts is None:
        vals, cnts = np.unique(a, return_counts=True)
        return {int(v): int(c) for v, c in zip(vals, cnts)}
    return {int(v): int(c) for v, c in enumerate(counts) if c}


# ----------------------------------------------------------------------------
# individual checks
# ----------------------------------------------------------------------------
def check_files(cfg: dict[str, Any], root: Path, rep: Report) -> dict[str, Path]:
    raw = cfg["paths"]["raw"]
    paths: dict[str, Path] = {}
    files: dict[str, Any] = {}
    for key in (*RASTER_KEYS, "aoi"):
        p = root / raw[key]
        paths[key] = p
        info: dict[str, Any] = {"path": raw[key], "exists": p.is_file()}
        tiles = find_tiles(p) if p.suffix == ".tif" else []
        if tiles:
            info["tiles_found"] = [t.name for t in tiles]
            if p.is_file():
                rep.error(f"{key}: both {p.name} and split tiles exist {info['tiles_found']}; ambiguous, remove one set")
            else:
                rep.error(f"{key}: {p.name} missing but split tiles exist {info['tiles_found']}; mosaic them onto the grid first")
        if not p.is_file():
            if not tiles:
                rep.error(f"{key}: file not found: {raw[key]}")
            files[key] = info
            continue
        info["size_bytes"] = p.stat().st_size
        if info["size_bytes"] == 0:
            rep.error(f"{key}: file is empty: {raw[key]}")
        info["sha256"] = sha256_file(p)
        files[key] = info
    rep.facts["files"] = files
    return paths


def read_raster(key: str, path: Path, cfg: dict[str, Any], rep: Report) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Open, fully read (detects truncation/corruption) and check structure + grid."""
    import rasterio

    grid = cfg["grid"]
    v = cfg["validation"]
    meta: dict[str, Any] = {}
    try:
        with rasterio.open(path) as src:
            meta = {
                "driver": src.driver,
                "width": src.width,
                "height": src.height,
                "count": src.count,
                "dtypes": sorted(set(src.dtypes)),
                "crs": src.crs.to_string() if src.crs else None,
                "transform": list(src.transform)[:6],
                "bounds": list(src.bounds),
                "nodata": src.nodata,
                "descriptions": list(src.descriptions),
                "block_shapes": [list(b) for b in src.block_shapes[:1]],
                "compression": src.compression.value if src.compression else None,
            }
            arr = src.read()  # full read: raises on truncated / corrupt blocks
    except Exception as exc:  # noqa: BLE001
        rep.error(f"{key}: cannot open/read {path.name}: {type(exc).__name__}: {exc}")
        return None, meta

    exp_count = v["expected_band_counts"][key]
    exp_dtype = v["expected_dtypes"][key]
    if meta["count"] != exp_count:
        rep.error(f"{key}: band count {meta['count']} != expected {exp_count}")
    if meta["dtypes"] != [exp_dtype]:
        rep.error(f"{key}: dtype {meta['dtypes']} != expected {exp_dtype}")
    if (meta["width"], meta["height"]) != (grid["width"], grid["height"]):
        rep.error(f"{key}: size {meta['width']}x{meta['height']} != grid {grid['width']}x{grid['height']}")
    if meta["crs"] != grid["crs"]:
        rep.error(f"{key}: CRS {meta['crs']} != grid {grid['crs']}")
    exp_t = expected_transform(grid)
    if not np.allclose(meta["transform"], exp_t, rtol=0, atol=1e-6):
        rep.error(f"{key}: transform {meta['transform']} != expected {list(exp_t)}")
    exp_bounds = [grid["x0"], grid["ytop"] - grid["height"] * grid["pixel_size_m"],
                  grid["x0"] + grid["width"] * grid["pixel_size_m"], grid["ytop"]]
    if not np.allclose(meta["bounds"], exp_bounds, rtol=0, atol=1e-6):
        rep.error(f"{key}: bounds {meta['bounds']} != expected {exp_bounds}")

    names = {"s2_pre": cfg["s2"]["bands"], "s2_post": cfg["s2"]["bands"], "hansen": cfg["hansen"]["bands"]}.get(key)
    desc = meta["descriptions"]
    if names and any(desc):
        if [d or "" for d in desc] != list(names):
            rep.error(f"{key}: embedded band names {desc} != contract order {names}")
        meta["band_names_source"] = "embedded (match contract)"
    else:
        meta["band_names_source"] = "not embedded; contract order assumed"

    if np.issubdtype(arr.dtype, np.floating):
        meta["nan_count"] = int(np.isnan(arr).sum())
        meta["inf_count"] = int(np.isinf(arr).sum())
        if meta["nan_count"] or meta["inf_count"]:
            rep.error(f"{key}: contains NaN ({meta['nan_count']}) / inf ({meta['inf_count']})")
    else:
        meta["nan_count"] = meta["inf_count"] = 0
        meta["nan_inf_note"] = f"integer dtype {arr.dtype}: NaN/inf impossible"
    return arr, meta


def check_s2(key: str, arr: np.ndarray, cfg: dict[str, Any], rep: Report) -> dict[str, Any]:
    v = cfg["validation"]
    bands = cfg["s2"]["bands"]
    spec = [bands.index(b) for b in cfg["s2"]["spectral_bands"]]
    nobs = arr[bands.index(cfg["s2"]["nobs_band"])]
    refl = arr[spec]
    zero = refl == 0
    all_zero = zero.all(axis=0)
    any_zero = zero.any(axis=0)
    valid = ~all_zero
    n = all_zero.size
    out: dict[str, Any] = {
        "pixels": int(n),
        "valid_pixels": int(valid.sum()),
        "valid_fraction": float(valid.mean()),
        "all_reflectance_zero_pixels": int(all_zero.sum()),
        "partial_zero_pixels": int((any_zero & ~all_zero).sum()),
        "nobs_zero_pixels": int((nobs == 0).sum()),
        "nobs_zero_but_reflectance_nonzero": int(((nobs == 0) & valid).sum()),
        "reflectance_zero_but_nobs_positive": int(((nobs > 0) & all_zero).sum()),
        "bands": {},
    }
    if out["valid_fraction"] < v["min_valid_fraction"]:
        rep.error(f"{key}: valid coverage {out['valid_fraction']:.4f} < {v['min_valid_fraction']}")
    if out["partial_zero_pixels"]:
        rep.warn(f"{key}: {out['partial_zero_pixels']} pixels have some (not all) reflectance bands = 0")
    if out["nobs_zero_but_reflectance_nonzero"] or out["reflectance_zero_but_nobs_positive"]:
        rep.warn(f"{key}: nobs/reflectance no-data disagree "
                 f"(nobs=0 & refl>0: {out['nobs_zero_but_reflectance_nonzero']}, "
                 f"refl=0 & nobs>0: {out['reflectance_zero_but_nobs_positive']})")

    vmax = v["reflectance_max_plausible"]
    for i, name in enumerate(bands):
        band = arr[i]
        s_all = band_stats(band)
        s_valid = band_stats(band[valid])
        rec = {"all_pixels": s_all, "valid_pixels": s_valid}
        is_nobs = name == cfg["s2"]["nobs_band"]
        thr = v["min_band_std"]["nobs" if is_nobs else "reflectance"]
        if s_valid.get("count", 0) == 0 or s_valid["std"] < thr:
            rep.error(f"{key}/{name}: constant or near-constant band (std on valid pixels "
                      f"{s_valid.get('std', 0):.4f} < {thr})")
        if not is_nobs:
            over = int((band[valid] > vmax).sum())
            rec["over_max_plausible"] = over
            frac = over / max(int(valid.sum()), 1)
            rec["over_max_plausible_fraction"] = frac
            if frac > v["reflectance_over_max_fraction_error"]:
                rep.error(f"{key}/{name}: {frac:.2%} of valid pixels > {vmax} (implausible reflectance)")
            elif over:
                rep.warn(f"{key}/{name}: {over} valid pixels ({frac:.4%}) > {vmax} DN (bright targets / residual cloud)")
        out["bands"][name] = rec

    nobs_mean = out["bands"][cfg["s2"]["nobs_band"]]["valid_pixels"]["mean"]
    lo, hi = v["expected_nobs_mean_range"]
    if not lo <= nobs_mean <= hi:
        rep.warn(f"{key}: nobs mean on valid pixels {nobs_mean:.2f} outside expected {lo}-{hi}")
    return out


def check_worldcover(arr: np.ndarray, cfg: dict[str, Any], rep: Report) -> dict[str, Any]:
    v = cfg["validation"]
    wc = cfg["worldcover"]
    band = arr[0]
    counts = value_counts(band)
    allowed = set(wc["raw_ids_present"]) | {wc["nodata_value"]}
    unexpected = sorted(set(counts) - allowed)
    if unexpected:
        rep.error(f"worldcover: unexpected class IDs {unexpected} (counts {[counts[u] for u in unexpected]})")
    unmapped = sorted(set(counts) - set(int(k) for k in wc["raw_to_model"]))
    if unmapped:
        rep.error(f"worldcover: IDs {unmapped} have no entry in worldcover.raw_to_model")
    labelled = sum(c for k, c in counts.items() if k != wc["nodata_value"])
    shares = {k: 100.0 * c / labelled for k, c in counts.items() if k != wc["nodata_value"]} if labelled else {}
    diffs = {}
    for cid, exp in v["expected_worldcover_share_pct"].items():
        act = shares.get(int(cid), 0.0)
        diffs[int(cid)] = {"expected_pct": exp, "actual_pct": round(act, 4), "diff_pp": round(act - exp, 4)}
        if abs(act - exp) > v["worldcover_share_tolerance_pp"]:
            rep.warn(f"worldcover: class {cid} share {act:.2f}% vs expected {exp}% "
                     f"(diff {act - exp:+.2f} pp > {v['worldcover_share_tolerance_pp']} pp)")
    absent = [int(c) for c in wc["raw_ids_present"] if int(c) not in counts]
    if absent:
        rep.warn(f"worldcover: expected IDs absent from raster: {absent}")
    model_counts: dict[int, int] = {}
    for raw_id, c in counts.items():
        m = int(wc["raw_to_model"].get(raw_id, wc["ignore_index"]))
        model_counts[m] = model_counts.get(m, 0) + c
    return {
        "counts": counts,
        "nodata_pixels": counts.get(wc["nodata_value"], 0),
        "labelled_pixels": labelled,
        "share_pct_of_labelled": {k: round(s, 4) for k, s in sorted(shares.items())},
        "share_vs_expected": diffs,
        "model_class_counts_before_loss_masking": dict(sorted(model_counts.items())),
    }


def check_hansen(arr: np.ndarray, cfg: dict[str, Any], rep: Report) -> dict[str, Any]:
    v = cfg["validation"]
    codes = v["hansen_valid_codes"]
    b = {name: arr[i] for i, name in enumerate(cfg["hansen"]["bands"])}
    tc, ly, dm, lt = b["treecover2000"], b["lossyear"], b["datamask"], b["loss_target"]
    out: dict[str, Any] = {
        "treecover2000": {"min": int(tc.min()), "max": int(tc.max()), "mean": float(tc.mean())},
        "lossyear_counts": value_counts(ly),
        "datamask_counts": value_counts(dm),
        "loss_target_counts": value_counts(lt),
    }
    if tc.min() < codes["treecover_range"][0] or tc.max() > codes["treecover_range"][1]:
        rep.error(f"hansen/treecover2000: values outside {codes['treecover_range']} (min {tc.min()}, max {tc.max()})")
    bad_ly = sorted(k for k in out["lossyear_counts"] if not codes["lossyear_range"][0] <= k <= codes["lossyear_range"][1])
    if bad_ly:
        rep.error(f"hansen/lossyear: codes outside {codes['lossyear_range']}: {bad_ly}")
    bad_dm = sorted(set(out["datamask_counts"]) - set(codes["datamask"]))
    if bad_dm:
        rep.error(f"hansen/datamask: unexpected codes {bad_dm} (counts {[out['datamask_counts'][k] for k in bad_dm]})")
    bad_lt = sorted(set(out["loss_target_counts"]) - set(codes["loss_target"]))
    if bad_lt:
        rep.error(f"hansen/loss_target: unexpected values {bad_lt}")

    # Recompute loss_target from its definition and compare pixel by pixel.
    land = dm == 1
    in_window = np.isin(ly, v["loss_years"])
    forest = tc >= v["tree_cover_threshold"]
    exp = np.full(lt.shape, 255, dtype=np.uint8)
    exp[land & (ly == 0)] = 0
    exp[land & in_window & forest] = 1
    mismatch = exp != lt
    n_mis = int(mismatch.sum())
    pairs = {}
    if n_mis:
        e, a = exp[mismatch], lt[mismatch]
        for ev in (0, 1, 255):
            for av in np.unique(a[e == ev]):
                pairs[f"expected {ev} -> file {int(av)}"] = int(((e == ev) & (a == av)).sum())
        rep.error(f"hansen/loss_target: {n_mis} pixels disagree with the definition "
                  f"(land & lossyear in {v['loss_years']} & treecover2000 >= {v['tree_cover_threshold']} -> 1; "
                  f"land & lossyear 0 -> 0; else 255): {pairs}")
    out["loss_target_recompute_mismatches"] = n_mis
    out["loss_target_mismatch_breakdown"] = pairs

    ignore = lt == 255
    out["ignore_reasons"] = {
        "not_land (water or no data)": int((ignore & ~land).sum()),
        "land, lossyear 1-20": int((ignore & land & (ly >= 1) & (ly <= 20)).sum()),
        "land, lossyear 24": int((ignore & land & (ly == 24)).sum()),
        f"land, lossyear 21-23, treecover2000 < {v['tree_cover_threshold']}": int((ignore & land & in_window & ~forest).sum()),
    }
    pos, neg, ign = (out["loss_target_counts"].get(k, 0) for k in (1, 0, 255))
    out["positive_fraction_of_labelled"] = pos / max(pos + neg, 1)
    out["positives_by_lossyear"] = {int(y): int(((lt == 1) & (ly == y)).sum()) for y in v["loss_years"]}

    for name, act in (("positive", pos), ("negative", neg), ("ignore", ign)):
        exp_c = v["expected_loss_target_counts"][name]
        rel = (act - exp_c) / exp_c if exp_c else float("inf")
        out.setdefault("loss_target_vs_expected", {})[name] = {"expected": exp_c, "actual": act, "diff": act - exp_c}
        if act != exp_c:
            msg = f"hansen/loss_target: {name} count {act} vs expected {exp_c} (diff {act - exp_c:+d}, {rel:+.3%})"
            if abs(rel) > v["loss_target_count_tolerance_rel"]:
                rep.warn(msg + " -- exceeds tolerance")
            else:
                rep.warn(msg + " -- within tolerance")
    for code, name in ((1, "land"), (2, "water")):
        act = out["datamask_counts"].get(code, 0)
        exp_c = v["expected_datamask_counts"][name]
        if abs(act - exp_c) / exp_c > v["datamask_count_tolerance_rel"]:
            rep.warn(f"hansen/datamask: {name} {act} px vs about {exp_c} expected")
    return out


def check_aoi(path: Path, cfg: dict[str, Any], rep: Report) -> dict[str, Any]:
    from rasterio.warp import transform_bounds

    try:
        gj = json.loads(path.read_text(encoding="utf-8"))
        feat = gj["features"][0] if gj.get("type") == "FeatureCollection" else gj
        props = feat.get("properties", {}) or {}
        coords = np.array(feat["geometry"]["coordinates"][0], dtype=float)
    except Exception as exc:  # noqa: BLE001
        rep.error(f"aoi: cannot parse {path.name}: {type(exc).__name__}: {exc}")
        return {}
    g, v = cfg["grid"], cfg["validation"]
    checks = {
        "crs": (props.get("crs"), g["crs"]),
        "width_px": (props.get("width_px"), g["width"]),
        "height_px": (props.get("height_px"), g["height"]),
        "x0": (props.get("x0"), g["x0"]),
        "ytop": (props.get("ytop"), g["ytop"]),
        "scale_m": (props.get("scale_m"), g["pixel_size_m"]),
        "s2_bands": (props.get("s2_bands"), ",".join(cfg["s2"]["bands"])),
        "hansen_bands": (props.get("hansen_bands"), ",".join(cfg["hansen"]["bands"])),
        "tree_thresh": (props.get("tree_thresh"), v["tree_cover_threshold"]),
        "loss_years": (props.get("loss_years"), f"{min(v['loss_years'])}-{max(v['loss_years'])}"),
        "tiles_x": (props.get("tiles_x"), 1),
        "tiles_y": (props.get("tiles_y"), 1),
    }
    out: dict[str, Any] = {"properties": props, "comparisons": {}}
    for name, (actual, expected) in checks.items():
        ok = actual == expected
        out["comparisons"][name] = {"aoi": actual, "config": expected, "match": ok}
        if not ok:
            rep.error(f"aoi: property {name}={actual!r} disagrees with config {expected!r}")
    px = g["pixel_size_m"]
    rb = (g["x0"], g["ytop"] - g["height"] * px, g["x0"] + g["width"] * px, g["ytop"])
    geo = transform_bounds(g["crs"], "EPSG:4326", *rb, densify_pts=101)
    aoi_bb = (coords[:, 0].min(), coords[:, 1].min(), coords[:, 0].max(), coords[:, 1].max())
    diff = float(np.max(np.abs(np.array(geo) - np.array(aoi_bb))))
    out["raster_bounds_lonlat"] = [round(x, 6) for x in geo]
    out["aoi_bbox_lonlat"] = [round(float(x), 6) for x in aoi_bb]
    out["max_bbox_diff_deg"] = diff
    if diff > v["aoi_bounds_tolerance_deg"]:
        rep.error(f"aoi: polygon bbox {out['aoi_bbox_lonlat']} differs from raster footprint "
                  f"{out['raster_bounds_lonlat']} by {diff:.5f} deg")
    return out


def cross_checks(s2: dict[str, np.ndarray], wc: np.ndarray, hz: np.ndarray, cfg: dict[str, Any]) -> dict[str, Any]:
    """Informational joint statistics (no hard errors)."""
    spec = [cfg["s2"]["bands"].index(b) for b in cfg["s2"]["spectral_bands"]]
    valid_pre = ~(s2["s2_pre"][spec] == 0).all(axis=0)
    valid_post = ~(s2["s2_post"][spec] == 0).all(axis=0)
    both = valid_pre & valid_post
    lt = hz[cfg["hansen"]["bands"].index("loss_target")]
    dm = hz[cfg["hansen"]["bands"].index("datamask")]
    wcb = wc[0]
    lc_model_valid = np.isin(wcb, [k for k, m in cfg["worldcover"]["raw_to_model"].items() if m != 255])
    return {
        "s2_valid_both_dates": int(both.sum()),
        "s2_valid_both_fraction": float(both.mean()),
        "loss_target_positive_with_valid_s2": int(((lt == 1) & both).sum()),
        "loss_target_negative_with_valid_s2": int(((lt == 0) & both).sum()),
        "landcover_trainable_pixels (mapped class, valid S2, loss_target != 1)": int((lc_model_valid & both & (lt != 1)).sum()),
        "worldcover_water80_vs_hansen_water2": {
            "both": int(((wcb == 80) & (dm == 2)).sum()),
            "wc80_only": int(((wcb == 80) & (dm != 2)).sum()),
            "hansen2_only": int(((wcb != 80) & (dm == 2)).sum()),
        },
        "loss_target_positive_by_worldcover_raw_id": value_counts(wcb[lt == 1]),
    }


# ----------------------------------------------------------------------------
# entry point
# ----------------------------------------------------------------------------
def validate_raw_data(cfg: dict[str, Any], root: Path) -> Report:
    rep = Report()
    rep.facts["started_utc"] = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    paths = check_files(cfg, root, rep)
    arrays: dict[str, np.ndarray] = {}
    rep.facts["rasters"] = {}
    for key in RASTER_KEYS:
        if not paths[key].is_file():
            continue
        arr, meta = read_raster(key, paths[key], cfg, rep)
        rep.facts["rasters"][key] = {"metadata": meta}
        if arr is None:
            continue
        arrays[key] = arr
        expect = cfg["validation"]["expected_band_counts"][key]
        if arr.shape[0] != expect or arr.shape[1:] != (cfg["grid"]["height"], cfg["grid"]["width"]):
            continue  # structure already reported; content checks would be meaningless
        if key.startswith("s2_"):
            rep.facts["rasters"][key]["content"] = check_s2(key, arr, cfg, rep)
        elif key == "worldcover":
            rep.facts["rasters"][key]["content"] = check_worldcover(arr, cfg, rep)
        else:
            rep.facts["rasters"][key]["content"] = check_hansen(arr, cfg, rep)
    if paths["aoi"].is_file():
        rep.facts["aoi"] = check_aoi(paths["aoi"], cfg, rep)
    if all(k in arrays for k in RASTER_KEYS) and not rep.errors:
        rep.facts["cross_checks"] = cross_checks(
            {"s2_pre": arrays["s2_pre"], "s2_post": arrays["s2_post"]}, arrays["worldcover"], arrays["hansen"], cfg
        )
    rep.facts["finished_utc"] = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    return rep


def _jsonable(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(x) for x in o]
    if isinstance(o, np.generic):
        return o.item()
    return o


def write_report(rep: Report, cfg: dict[str, Any], root: Path, config_path: Path) -> Path:
    """Write the report into a fresh timestamped run dir and update latest.json."""
    base = root / cfg["validation"]["report_root"]
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base / f"run_{stamp}"
    i = 1
    while run_dir.exists():
        run_dir = base / f"run_{stamp}_{i}"
        i += 1
    run_dir.mkdir(parents=True)
    doc = {
        "status": "passed" if rep.passed else "failed",
        "n_errors": len(rep.errors),
        "n_warnings": len(rep.warnings),
        "errors": rep.errors,
        "warnings": rep.warnings,
        "config_file": str(config_path),
        "config": cfg,
        **rep.facts,
    }
    (run_dir / "validation_report.json").write_text(json.dumps(_jsonable(doc), indent=2), encoding="utf-8")
    latest = {
        "status": doc["status"],
        "run_dir": os.path.relpath(run_dir, root).replace("\\", "/"),
        "finished_utc": rep.facts.get("finished_utc"),
        "sha256": {k: v.get("sha256") for k, v in rep.facts.get("files", {}).items()},
    }
    (base / "latest.json").write_text(json.dumps(latest, indent=2), encoding="utf-8")
    return run_dir


def require_validated_raw_data(cfg: dict[str, Any], root: Path) -> dict[str, Any]:
    """Guard for every later stage: the latest validation must have passed and
    the raw files must be byte-identical (SHA-256) to the ones validated.

    Raises RuntimeError otherwise, so stale or unvalidated data is never used.
    """
    latest_path = root / cfg["validation"]["report_root"] / "latest.json"
    if not latest_path.is_file():
        raise RuntimeError("No validation report found. Run scripts/inspect_rasters.py first.")
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    if latest.get("status") != "passed":
        raise RuntimeError(f"Latest raw-data validation did not pass ({latest.get('run_dir')}). Fix the data and re-run inspect_rasters.py.")
    for key, rel in cfg["paths"]["raw"].items():
        p = root / rel
        recorded = latest.get("sha256", {}).get(key)
        if not p.is_file() or recorded is None or sha256_file(p) != recorded:
            raise RuntimeError(f"{rel} changed or is missing since validation ({latest.get('run_dir')}). Re-run inspect_rasters.py.")
    return latest
