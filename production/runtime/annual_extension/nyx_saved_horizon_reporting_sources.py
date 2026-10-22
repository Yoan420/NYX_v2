"""Fresh EPEX/Storm comparisons, isolated from every NYX model input.

These are latest reporting extractions, not the model's shared-cutoff state.
Missing comparisons remain missing. Every returned series is materialized
with its own checksum and report-only provenance in a new output directory.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd


ZONES = ("FR", "DE", "BE", "NL")
TIMEZONES = {"FR": "Europe/Paris", "DE": "Europe/Berlin",
             "BE": "Europe/Brussels", "NL": "Europe/Amsterdam"}
PROTOCOL = "nyx_saved_horizon_reporting_sources_v1"


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _materialize(path, frame):
    _require(not path.exists(), f"Report-only snapshot already exists: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + uuid4().hex + ".tmp.parquet")
    try:
        frame.to_parquet(tmp)
        pd.testing.assert_frame_equal(pd.read_parquet(tmp), frame, check_freq=False,
                                      check_exact=True)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return _sha(path)


def _error(exc):
    from experiment_console.security import redact_text
    return {"status": "unavailable", "error_type": type(exc).__name__,
            "reason": redact_text(str(exc))[:1500], "used_for_prediction": False,
            "used_for_training": False, "fallback_to_model_inputs": False}


def collect_horizon_reporting_sources(repo, first_day, outputdir, expected_index,
                                     *, client=None, now_utc=None):
    from chronos2_modular.saturn import create_saturn_client
    from chronos2_hourly.reporting_observations import fetch_epex_reporting_observations
    from chronos2_hourly.storm_dashboard import (
        STORM_DASHBOARD_COLUMN, fetch_native_dashboard_snapshot,
        build_dashboard_comparator,
    )
    from run_nyx_annual_auction_prices_source import load_plan

    repo, output = Path(repo).resolve(), Path(outputdir).resolve()
    _require(output != repo and output != Path(output.anchor), "Isolated report output required")
    _require(not any((output / marker).exists() for marker in
        ("source_receipts", "model_receipt.json", "input_snapshot.json", "resolved_config.yaml")),
        "Cannot write comparisons into a completed model bundle")
    expected = pd.DatetimeIndex(expected_index)
    _require(str(expected.tz) == "UTC" and len(expected) and not expected.hasnans
        and expected.is_unique and expected.is_monotonic_increasing
        and expected.equals(pd.date_range(expected[0], expected[-1], freq="h")),
        "Complete physical-hour UTC reporting grid required")
    first = pd.Timestamp(first_day)
    _require(first.tzinfo is None and first == first.normalize(), "Civil first-day date required")
    first_utc = first.tz_localize("Europe/Paris").tz_convert("UTC")
    _require(first_utc in expected and expected[0] < first_utc,
        "Reporting grid must include historical statistics and future delivery")
    extracted = pd.Timestamp.now(tz="UTC") if now_utc is None else pd.Timestamp(now_utc)
    _require(extracted.tzinfo is not None and not pd.isna(extracted), "Explicit extraction timestamp required")
    extracted = extracted.tz_convert("UTC")
    snapshot = output / "reporting_sources"
    _require(not snapshot.exists(), "Fresh reporting source directory required")
    if client is None:
        plan = load_plan(repo)
        client = create_saturn_client(plan["saturn_url"], os.getenv("SATURN_AUTHOR") or plan["saturn_author"])
        if hasattr(client, "session"):
            from functools import partial
            client.session.request = partial(client.session.request, timeout=60.0)
    observed, storm, audits, artifacts = {}, {}, {}, {}
    for zone in ZONES:
        timezone = TIMEZONES[zone]
        actual = pd.Series(np.nan, index=expected, name="actual")
        dashboard = pd.Series(np.nan, index=expected, name=STORM_DASHBOARD_COLUMN)
        actual_audit, storm_audit = {}, {}
        try:
            raw, source = fetch_epex_reporting_observations(client, zone=zone,
                timezone=timezone, expected_index=expected, extracted_at_utc=extracted)
            _require(source.get("used_for_prediction") is False
                and source.get("reference_role") == "reporting_and_scoring_only"
                and source.get("policy") == "epex_only_v1", "Reporting EPEX identity differs")
            actual = raw.reindex(expected).astype(float).rename("actual")
            _require(not np.isinf(actual.to_numpy()).any(), "Infinite report observation")
            # Keep no partial future-day average, including a 23/25-hour day.
            placeholders = []
            for day in sorted(set(expected[expected >= first_utc].tz_convert(timezone).date)):
                mask = expected.tz_convert(timezone).date == day
                available = int(actual.loc[mask].notna().sum())
                if available != int(mask.sum()):
                    actual.loc[mask] = np.nan
                    placeholders.append({"delivery_day": str(day), "raw_available_hours": available,
                        "expected_hours": int(mask.sum()), "status": "pending_placeholder"})
            historical_missing = int(actual.loc[expected < first_utc].isna().sum())
            actual_audit = {"status": "complete" if historical_missing == 0 else "partial",
                "source": source, "historical_missing_hours": historical_missing,
                "future_placeholders": placeholders, "actual_reference": "EPEX",
                "available_hours": int(actual.notna().sum()), "used_for_prediction": False,
                "used_for_training": False, "partial_daily_average_forbidden": True}
        except Exception as exc:
            actual_audit = _error(exc)
        try:
            raw_storm, source = fetch_native_dashboard_snapshot(client, zone=zone,
                expected_index=expected, extracted_at_utc=extracted)
            _require(source.get("used_for_prediction") is False
                and source.get("zone") == zone, "Reporting Storm identity differs")
            selected = raw_storm.reindex(expected)
            _require(not np.isinf(selected.to_numpy(dtype=float)).any(), "Infinite Storm report comparator")
            missing = expected[~np.isfinite(selected.to_numpy(dtype=float))]
            actual_missing = expected[~np.isfinite(actual.to_numpy(dtype=float))]
            comparator = build_dashboard_comparator(raw_storm, expected_index=expected,
                actual=actual, source=source, timezone=timezone, minimum_coverage=0,
                maximum_missing_hours=len(missing), allowed_missing_actual_index=actual_missing)
            dashboard = comparator.values.reindex(expected)
            storm_audit = {"status": "complete" if len(missing) == 0 else "partial",
                "source": source, "comparator": comparator.audit,
                "historical_missing_hours": int(dashboard.loc[expected < first_utc].isna().sum()),
                "forecast_missing_hours": int(dashboard.loc[expected >= first_utc].isna().sum()),
                "quantile_interval_available": False, "used_for_prediction": False,
                "used_for_training": False}
        except Exception as exc:
            storm_audit = _error(exc)
        observed[zone], storm[zone] = actual, dashboard
        audit = {"zone": zone, "timezone": timezone, "observed": actual_audit,
                 "storm": storm_audit, "model_inputs_untouched": True}
        for label, values in (("observed", actual), ("storm", dashboard)):
            relative = f"reporting_sources/{zone}/{label}.parquet"
            artifacts[relative] = _materialize(output / relative, values.to_frame())
        audits[zone] = audit
    receipt = {"protocol": PROTOCOL, "extracted_at_utc": extracted.isoformat(),
        "first_forecast_day": str(first.date()), "first_utc": expected[0].isoformat(),
        "last_utc": expected[-1].isoformat(), "expected_hours": len(expected),
        "used_for_prediction": False, "used_for_training": False,
        "latest_reporting_extraction_not_model_asof_state": True,
        "countries": audits, "artifact_sha256": artifacts}
    receipt_path = snapshot / "audit.json"
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return {"observed": observed, "storm": storm, "audit": receipt,
            "audits": audits, "receipt_path": str(receipt_path)}
