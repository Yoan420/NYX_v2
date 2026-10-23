"""Immutable, receipted carry-over of genuine NYX daily prediction histories.

Future predictions may become historical predictions on a later run. Their
stored target column is never accepted as a label: the baseline producer must
attach newly queried real EPEX prices before fitting its bounded correctors.
"""
from __future__ import annotations

from datetime import timedelta
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
from uuid import uuid4

import numpy as np
import pandas as pd

PROTOCOL = "nyx_daily_cpu_carry_state_v1"
ZONES = ("FR", "DE", "BE", "NL")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_contract(seed_bundle, seed_output, *, threads=2):
    from chronos2_hourly import nyx_annual_cpu_baseline as annual
    from chronos2_hourly.nyx_annual_epex_prices import EPEX_SERIES
    seed_bundle, seed_output = Path(seed_bundle).resolve(), Path(seed_output).resolve()
    baseline_receipt = _json(seed_bundle / "source_receipts/nyx_quantiles.json")
    configurations = {zone: annual._configuration(zone, threads)[1] for zone in ZONES}
    weights = {}
    for zone in ZONES:
        relative = f"baseline_audits/{zone}.json"
        path = seed_bundle / relative
        require(baseline_receipt["artifact_sha256"].get(relative) == sha(path), "Seed baseline audit checksum differs")
        weights[zone] = _json(path)["chronos_model_sha256"]
    require(len(set(weights.values())) == 1, "Seed countries use different Chronos weights")
    extension_files = ("nyx_daily_two_day_baseline.py", "nyx_saved_horizon_inference.py",
                       "nyx_saved_horizon_sources.py", "nyx_daily_state_cache.py")
    directory = Path(__file__).resolve().parent
    return {"protocol": PROTOCOL,
        "seed_report_receipt_sha256": sha(seed_output / "receipt.json"),
        "seed_baseline_receipt_sha256": sha(seed_bundle / "source_receipts/nyx_quantiles.json"),
        "chronos_model_sha256": next(iter(weights.values())), "price_policy": "epex_only_v1",
        "price_series": dict(EPEX_SERIES), "cpu_threads": threads,
        "dependencies": {relative: annual.sha256_text(annual.ROOT / relative) for relative in annual.CODE_FILES},
        "configuration_sha256": configurations,
        "runtime": {name: importlib.metadata.version(name) for name in
            ("torch", "chronos-forecasting", "numpy", "pandas", "catboost", "pykalman", "scipy", "holidays")},
        "extension_code_sha256": {name: annual.sha256_text(directory / name) for name in extension_files}}


def _checked_frame(path, expected_hash):
    require(path.is_file() and not path.is_symlink() and sha(path) == expected_hash,
            "Rolling daily cache artifact checksum differs: " + path.name)
    frame = pd.read_parquet(path)
    require(sha(path) == expected_hash, "Rolling daily cache changed during read")
    index = frame.index
    require(isinstance(index, pd.DatetimeIndex) and str(index.tz) == "UTC" and len(index)
            and index.is_unique and index.is_monotonic_increasing and not index.hasnans
            and index.equals(index.floor("h")), "Rolling daily cache has an invalid physical-hour grid")
    return frame


def load_state(cache_root, contract, *, first_delivery_day, cutoff):
    """Choose the latest compatible completed state no later than this issue."""
    root = Path(cache_root).resolve()
    cutoff = pd.Timestamp(cutoff).tz_convert("UTC")
    first = pd.Timestamp(first_delivery_day, tz="Europe/Paris").tz_convert("UTC")
    if not root.exists():
        return None
    require(root.is_dir() and not root.is_symlink(), "Redirected rolling daily cache directory")
    candidates = []
    for directory in root.iterdir():
        if not directory.is_dir() or directory.name.startswith("."):
            continue
        require(not directory.is_symlink(), "Redirected rolling daily cache state")
        path = directory / "receipt.json"
        if not path.is_file():
            continue
        receipt = _json(path)
        stamp = pd.Timestamp(receipt.get("asof_cutoff_utc"))
        require(not pd.isna(stamp) and stamp.tzinfo is not None, "Rolling state cutoff missing")
        stamp = stamp.tz_convert("UTC")
        if stamp > cutoff or receipt.get("contract") != contract:
            continue
        require(receipt.get("protocol") == PROTOCOL and receipt.get("state") == "COMPLETE"
                and receipt.get("future_labels_used") is False, "Invalid completed rolling daily state")
        candidates.append((stamp, receipt.get("created_at_utc", ""), directory, receipt))
    if not candidates:
        return None
    stamp, _, directory, receipt = max(candidates, key=lambda item: (item[0], item[1], item[2].name))
    hashes = receipt["artifact_sha256"]
    expected_names = {"covariates.parquet"} | {f"{zone}/{name}.parquet" for zone in ZONES
        for name in ("raw_history", "residual_history", "history")}
    require(set(hashes) == expected_names, "Rolling cache history inventory differs")
    frames = {name: _checked_frame(directory / name, hashes[name]) for name in hashes}
    zones, last_times = {}, []
    for zone in ZONES:
        zones[zone] = {}
        for name in ("raw_history", "residual_history", "history"):
            # Even cached future rows are never handed to historical fitting.
            frame = frames[f"{zone}/{name}.parquet"]
            selected = frame.loc[frame.index < first].copy()
            require(len(selected), "Rolling state lacks strictly past history")
            quantile_columns = [f"nyx__{q}" for q in ("q10", "q50", "q90")] if name == "history" else ["q10", "q50", "q90"]
            if name == "residual_history":
                quantile_columns = [f"residual_corrected__{q}" for q in ("q10", "q50", "q90")]
            require(set(quantile_columns) <= set(selected)
                    and np.isfinite(selected[quantile_columns].to_numpy(float)).all(), "Invalid rolling cached quantiles")
            if "forecast_origin_utc" in selected:
                origins = pd.to_datetime(selected.forecast_origin_utc, utc=True)
                require(origins.notna().all() and (origins <= stamp).all(), "Cached forecast originated after its state cutoff")
            selected["actual"] = np.nan
            zones[zone][name] = selected
            last_times.append(selected.index[-1])
    require(len(set(last_times)) == 1, "Rolling histories end on different physical hours")
    expected_last = pd.Timestamp(last_times[0]).tz_convert("Europe/Paris")
    require(expected_last.hour == 23, "Rolling state ends on an incomplete civil day")
    return {"contract": contract, "asof_cutoff_utc": stamp.isoformat(), "zones": zones,
        "covariates": frames["covariates.parquet"], "last_history_day": str(expected_last.date()),
        "receipt_sha256": sha(directory / "receipt.json"), "receipt_path": str(directory / "receipt.json"),
        "covariates_evidence": {"sha256": hashes["covariates.parquet"],
            "receipt_sha256": sha(directory / "receipt.json"), "path": str(directory / "covariates.parquet"),
            "seed_receipt_sha256": contract["seed_report_receipt_sha256"]}}


def publish_state(cache_root, contract, *, produced, covariates, cutoff):
    """Publish a new immutable owned cache; interrupted folders are ignored."""
    root = Path(cache_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    require(not root.is_symlink(), "Redirected rolling daily cache directory")
    stamp = pd.Timestamp(cutoff).tz_convert("UTC")
    name = stamp.strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:12]
    attempt = root / ("." + name + ".attempt")
    final = root / name
    attempt.mkdir()
    hashes = {}
    try:
        data = {"covariates.parquet": covariates}
        for zone in ZONES:
            for history_name in ("raw_history", "residual_history", "history"):
                data[f"{zone}/{history_name}.parquet"] = produced["zones"][zone][history_name]
        for relative, frame in data.items():
            path = attempt / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(path)
            hashes[relative] = sha(path)
        receipt = {"protocol": PROTOCOL, "state": "COMPLETE", "contract": contract,
            "asof_cutoff_utc": stamp.isoformat(), "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "artifact_sha256": hashes, "future_labels_used": False,
            "cached_actual_values_used_for_training": False, "baseline_audit": produced["audit"]}
        (attempt / "receipt.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False,
            allow_nan=False, default=str) + "\n", encoding="utf-8")
        attempt.rename(final)
    except BaseException:
        shutil.rmtree(attempt, ignore_errors=True)
        raise
    return {"receipt_path": str(final / "receipt.json"), "receipt_sha256": sha(final / "receipt.json"),
            "asof_cutoff_utc": stamp.isoformat(), "new_state_published": True}
