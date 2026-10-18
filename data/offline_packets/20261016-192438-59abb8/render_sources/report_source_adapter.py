"""Verified EPEX observations and official Storm snapshots for report exports.

This adapter reads completed annual artifacts. Optional refresh publishes new
report-only snapshots; it never changes model inputs, calibrations or forecasts.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import yaml


TIMEZONES = {"FR": "Europe/Paris", "DE": "Europe/Berlin",
             "BE": "Europe/Brussels", "NL": "Europe/Amsterdam"}
EPEX_SERIES = {zone: f"power.price.{zone.lower()}.euromwh.h.obs.epex"
               for zone in TIMEZONES}
CONFIG_FILES = {
    "FR": "chronos2_hourly_fr_residual_v1.yaml",
    "DE": "chronos2_hourly_de_residual_candidate_v1.yaml",
    "BE": "chronos2_hourly_be_residual_candidate_v1.yaml",
    "NL": "chronos2_hourly_nl_residual_candidate_v1.yaml",
}
REPORT_SOURCE_DEPENDENCIES = (
    "chronos2_hourly/nuclear_reporting_refresh.py",
    "chronos2_hourly/nuclear_report_benchmark.py",
    "chronos2_hourly/reporting_observations.py",
    "chronos2_hourly/storm_dashboard.py",
    "chronos2_hourly/hourly_contract.py",
    "chronos2_hourly/atomic_directory.py",
)


class ReportSourceError(ValueError):
    """A source cannot be used without changing or inventing its provenance."""


def _require(condition, message):
    if not condition:
        raise ReportSourceError(message)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path):
    path = Path(path)
    _require(path.is_file(), f"Missing report source evidence: {path}")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as error:
        raise ReportSourceError(f"Unreadable source receipt: {path}") from error
    _require(isinstance(value, dict), f"Source receipt must be an object: {path}")
    _require(path.read_bytes() == raw, f"Source receipt changed during read: {path}")
    return value, hashlib.sha256(raw).hexdigest()


def _bound(root, receipt, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    expected = receipt.get("artifact_sha256", {}).get(relative)
    _require(path.is_relative_to(root) and path.is_file(),
             f"Source artifact missing or outside its bundle: {relative}")
    _require(isinstance(expected, str) and _sha(path) == expected,
             f"Source artifact checksum differs: {relative}")
    return path, expected


def _read_frame(path, expected):
    _require(_sha(path) == expected, f"Source artifact changed before read: {path}")
    frame = pd.read_parquet(path)
    _require(_sha(path) == expected, f"Source artifact changed during read: {path}")
    return frame


def _series(value, label):
    if isinstance(value, pd.DataFrame):
        _require("actual" in value and value.columns.is_unique,
                 f"{label}: an unambiguous actual column is required")
        stamps = value.index
        if not isinstance(stamps, pd.DatetimeIndex):
            names = [name for name in ("delivery_start_utc", "timestamp_utc", "timestamp")
                     if name in value]
            _require(len(names) == 1, f"{label}: explicit physical timestamps required")
            stamps = pd.DatetimeIndex(value[names[0]])
        value = pd.Series(value["actual"].to_numpy(), index=stamps, name="actual")
    _require(isinstance(value, pd.Series) and isinstance(value.index, pd.DatetimeIndex),
             f"{label}: an hourly actual Series is required")
    index = value.index
    _require(index.tz is not None and not index.hasnans and index.is_unique
             and index.is_monotonic_increasing,
             f"{label}: unique increasing timezone-aware timestamps required")
    index = index.tz_convert("UTC")
    _require(index.equals(index.floor("h")), f"{label}: physical hourly UTC grid required")
    _require(not np.iscomplexobj(value.to_numpy()), f"{label}: complex prices rejected")
    try:
        numeric = pd.to_numeric(value.to_numpy(), errors="raise").astype(float)
    except (ValueError, TypeError) as error:
        raise ReportSourceError(f"{label}: nonnumeric actual prices") from error
    _require(not np.isinf(numeric).any(), f"{label}: infinite actual prices rejected")
    return pd.Series(numeric, index=index, name="actual")


def _support(day, timezone):
    begin = pd.Timestamp(day - timedelta(days=365), tz=timezone)
    middle = pd.Timestamp(day, tz=timezone)
    finish = pd.Timestamp(day + timedelta(days=1), tz=timezone)
    expected = pd.date_range(begin, finish, freq="h", inclusive="left").tz_convert("UTC")
    current = pd.date_range(middle, finish, freq="h", inclusive="left").tz_convert("UTC")
    return expected, current


def _epex_identity(receipt, series_key):
    _require(receipt.get("price_reference") == "EPEX"
             and receipt.get("price_reference_policy") == "epex_only_v1"
             and receipt.get("source_mixing") is False
             and receipt.get("actual_values_imputed") is False
             and receipt.get("price_fallback_policy") == "none",
             "Annual report sources must retain the declared EPEX-only model policy")
    _require(receipt.get(series_key) == EPEX_SERIES,
             f"Annual {series_key} must name the four explicit EPEX hourly sources")


def _annual_evidence(run_dir, day):
    bundle = run_dir / "bundle"
    base_path = bundle / "source_receipts/nyx_quantiles.json"
    base_receipt, base_hash = _json(base_path)
    _require(base_receipt.get("source_group") == "nyx_quantiles"
             and base_receipt.get("delivery_day") == str(day)
             and base_receipt.get("state") in ("COMPLETE", "UNQUALIFIED"),
             "Annual baseline source receipt has a different day, group or state")
    saturn_path = bundle / "source_receipts/saturn.json"
    saturn, saturn_hash = _json(saturn_path)
    _require(base_receipt.get("source_receipts_sha256", {}).get("saturn") == saturn_hash,
             "Annual baseline no longer binds its original Saturn source receipt")
    _epex_identity(saturn, "price_series")
    _require(saturn.get("delivery_day") == str(day), "Frozen EPEX target day differs")
    target_path, target_hash = _bound(
        bundle, saturn, "source_artifacts/saturn/targets/prices.parquet")
    targets = _read_frame(target_path, target_hash)
    _require(set(EPEX_SERIES).issubset(targets.columns) and targets.columns.is_unique,
             "Frozen EPEX price snapshot must contain every country")
    return bundle, base_receipt, {
        "baseline_receipt_path": str(base_path), "baseline_receipt_sha256": base_hash,
        "saturn_receipt_path": str(saturn_path), "saturn_receipt_sha256": saturn_hash,
        "target_artifact_path": str(target_path), "target_artifact_sha256": target_hash,
        "original_price_cutoff_utc": saturn.get("target_revision_utc"),
    }, targets


def _frozen_observed(bundle, receipt, provenance, targets, zone, supplied, expected, current):
    relative = f"baseline/{zone}.parquet"
    if not (bundle / relative).exists():
        relative = f"baseline_history/{zone}.parquet"
    path, digest = _bound(bundle, receipt, relative)
    stored = _series(_read_frame(path, digest), f"{zone} frozen baseline")
    proposed = _series(supplied, f"{zone} supplied baseline")
    _require(not len(expected.difference(stored.index))
             and not len(expected.difference(proposed.index)),
             f"{zone}: frozen baseline must cover D-365 through D")
    observed = stored.reindex(expected)
    _require(np.array_equal(observed.to_numpy(), proposed.reindex(expected).to_numpy(), equal_nan=True),
             f"{zone}: supplied actual prices differ from the checksum-bound annual baseline")
    historical = expected.difference(current)
    _require(np.isfinite(observed.reindex(historical).to_numpy()).all(),
             f"{zone}: frozen historical EPEX actuals are incomplete")
    _require(observed.reindex(current).isna().all(),
             f"{zone}: the frozen forecast day must have no observed labels")
    target = _series(targets[zone], f"{zone} frozen EPEX target")
    _require(np.array_equal(observed.reindex(historical).to_numpy(),
                           target.reindex(historical).to_numpy(), equal_nan=True),
             f"{zone}: baseline actuals disagree with their original EPEX price snapshot")
    source = {
        "status": "historical_only", "kind": "annual_epex_frozen_actuals",
        "series": EPEX_SERIES[zone], "actual_reference": "EPEX", "actual_reference_label": "EPEX",
        "timezone": TIMEZONES[zone], "delivery_day_local": str(current[0].tz_convert(TIMEZONES[zone]).date()),
        "artifact_path": str(path), "artifact_sha256": digest,
        "current_delivery_actual_status": "pending_placeholder",
        "current_delivery_expected_hours": len(current),
        "current_delivery_raw_available_hours": 0,
        "partial_daily_average_forbidden": True,
        "source_mixing": False, "used_for_prediction": False,
        **provenance,
    }
    return observed.copy(), source


def _guard_refresh_output(repo, run_dir, output_dir):
    output_dir = Path(output_dir).resolve()
    forbidden = [repo, run_dir, run_dir / "bundle", run_dir / "cache"]
    _require(output_dir not in (repo, run_dir), "Use a separate report-only source output directory")
    _require(not any(output_dir.is_relative_to(path) for path in forbidden[2:]),
             "Report sources cannot be written into model bundles or training caches")
    record_path = run_dir / "run.json"
    if record_path.is_file():
        record, _ = _json(record_path)
        if record.get("output"):
            completed_output = Path(record["output"])
            completed_output = (repo / completed_output).resolve() if not completed_output.is_absolute() else completed_output.resolve()
            _require(not output_dir.is_relative_to(completed_output),
                     "Report source refresh cannot modify the completed forecast output")
    return output_dir


def _config(repo, zone):
    path = repo / CONFIG_FILES[zone]
    _require(path.is_file(), f"Missing annual Saturn connection config: {path}")
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    except (ValueError, yaml.YAMLError) as error:
        raise ReportSourceError(f"Unreadable annual source config: {path}") from error
    _require(isinstance(config, dict) and isinstance(config.get("data"), dict)
             and zone in config.get("zones", {}), f"{zone}: invalid annual connection config")
    # The unchanged YAML target identifies its original configuration. The
    # annual EPEX model policy is an explicit override, also for this report.
    config = deepcopy(config)
    config["zones"][zone]["target"]["series"] = EPEX_SERIES[zone]
    config["zones"][zone]["target"]["naive_timezone"] = "UTC"
    return config, path


def _storm_snapshot(archive, zone, expected):
    from chronos2_hourly.nuclear_report_benchmark import _load_verified_snapshot
    loaded = _load_verified_snapshot(Path(archive), zone=zone, timezone=TIMEZONES[zone])
    _require(loaded is not None, f"{zone}: no verified official Storm snapshot in {archive}")
    values, materialization, provenance = loaded
    _require(not len(expected.difference(values.index)),
             f"{zone}: official Storm snapshot must span D-365 through D")
    _require(_sha(provenance["artifact_path"]) == provenance["artifact_sha256"]
             and _sha(provenance["audit_path"]) == provenance["audit_sha256"],
             f"{zone}: Storm snapshot changed after verification")
    return values.reindex(expected).copy(), {
        "status": "complete" if values.reindex(expected).notna().all() else "partial",
        "kind": "verified_official_storm_report_snapshot", "used_for_prediction": False,
        "materialization": materialization, **provenance,
    }


def load_sources(repo, run_dir, day, base_by_zone, output_dir, refresh=False, *, storm_archives=None):
    """Return observed/Storm hourly Series and their per-country audit.

    ``base_by_zone`` contains the already verified annual baseline frames with
    an actual column. Frozen mode leaves the forecast day's actuals missing.
    ``storm_archives`` may explicitly name local audited comparator directories.
    Refresh mode queries only reporting sources, then verifies both artifacts.
    """
    repo, run_dir = Path(repo).resolve(), Path(run_dir).resolve()
    try:
        parsed_day = date.fromisoformat(str(day))
    except (TypeError, ValueError) as error:
        raise ReportSourceError("Report day must use YYYY-MM-DD") from error
    _require(str(parsed_day) == str(day), "Report day must use YYYY-MM-DD")
    _require(isinstance(base_by_zone, Mapping) and set(base_by_zone) == set(TIMEZONES),
             "Four verified annual baseline country frames are required")
    storm_archives = {} if storm_archives is None else dict(storm_archives)
    _require(set(storm_archives).issubset(TIMEZONES), "Unknown Storm comparator country")
    bundle, receipt, provenance, targets = _annual_evidence(run_dir, parsed_day)
    destination = _guard_refresh_output(repo, run_dir, output_dir) if refresh else None
    observed_by_zone, storm_by_zone, audit_by_zone = {}, {}, {}
    for zone, timezone in TIMEZONES.items():
        expected, current = _support(parsed_day, timezone)
        frozen, frozen_audit = _frozen_observed(
            bundle, receipt, provenance, targets, zone, base_by_zone[zone], expected, current)
        observed, observed_audit = frozen, frozen_audit
        storm = pd.Series(np.nan, index=expected, name="storm")
        storm_audit = {"status": "unavailable", "kind": "no_verified_storm_snapshot",
                       "reason": "No comparator supplied; archived HTML values are not source evidence",
                       "used_for_prediction": False}
        if refresh:
            from chronos2_hourly.nuclear_reporting_refresh import (
                refresh_nuclear_reporting_sources, verify_refreshed_observations,
            )
            config, config_path = _config(repo, zone)
            try:
                observed, archive, refresh_audit = refresh_nuclear_reporting_sources(
                    config, zone, timezone, str(parsed_day), destination / zone.lower())
                verified = verify_refreshed_observations(
                    observed, refresh_audit, zone=zone, timezone=timezone, delivery_day=str(parsed_day))
                _require(verified.get("actual_reference") == "EPEX",
                         f"{zone}: refreshed observations must retain the single EPEX source")
                storm, storm_audit = _storm_snapshot(archive, zone, expected)
            except Exception as error:
                # Refresh exceptions can contain proxy/connection material.
                try:
                    from experiment_console.security import redact_text
                    detail = redact_text(str(error))
                except ImportError:
                    detail = type(error).__name__
                raise ReportSourceError(f"{zone}: report-only EPEX/official Storm refresh failed: {detail}") from None
            observed = _series(observed, f"{zone} refreshed EPEX actuals").reindex(expected)
            observed_audit = {
                **refresh_audit["observed"]["source"], **verified,
                "status": "complete" if observed.loc[current].notna().all() else "historical_only",
                "artifact_path": refresh_audit["observed"]["artifact_path"],
                "artifact_sha256": refresh_audit["observed"]["artifact_sha256"],
                "snapshot_directory": str(archive), "source_config_path": str(config_path),
                "source_config_sha256": _sha(config_path), "report_refresh_audit": refresh_audit,
            }
            _require(_sha(observed_audit["artifact_path"]) == observed_audit["artifact_sha256"],
                     f"{zone}: refreshed EPEX snapshot changed after verification")
        elif zone in storm_archives:
            storm, storm_audit = _storm_snapshot(storm_archives[zone], zone, expected)
        observed_by_zone[zone] = observed.copy().rename("actual")
        storm_by_zone[zone] = storm.copy().rename("storm")
        audit_by_zone[zone] = {
            "report_only": True, "used_for_prediction": False,
            "frozen_inputs_modified": False, "observed": observed_audit, "storm": storm_audit,
        }
    return observed_by_zone, storm_by_zone, audit_by_zone


def load_horizon_reporting_sources(repo, first_day, base_by_zone, output_dir, *, refresh=True):
    """Collect latest report comparators for the new two-day prediction only."""
    from chronos2_hourly.nyx_saved_horizon_reporting_sources import collect_horizon_reporting_sources
    expected = base_by_zone["FR"].index
    _require(all(frame.index.equals(expected) for frame in base_by_zone.values()),
             "Four-country reporting support differs")
    if not refresh:
        observed = {zone: frame.actual.copy().rename("actual") for zone, frame in base_by_zone.items()}
        storm = {zone: pd.Series(np.nan, index=expected, name="storm") for zone in base_by_zone}
        audits = {zone: {"report_only": True, "observed": {
            "kind": "model_asof_epex_reporting_copy", "actual_reference": "EPEX",
            "actual_reference_label": "EPEX", "series": EPEX_SERIES[zone],
            "future_observations_available": False, "used_for_prediction": False,
            "original_source_role": "model_source_snapshot"}, "storm": {
            "status": "unavailable", "reason": "Report-only source refresh disabled",
            "used_for_prediction": False}} for zone in base_by_zone}
        return observed, storm, audits
    collected = collect_horizon_reporting_sources(repo, first_day,
        Path(output_dir) / "report_only_sources", expected)
    return collected["observed"], collected["storm"], collected["audits"]
