"""Resume report rendering from a completed two-day CPU calculation.

This reader does not fetch sources, load Chronos, fit a corrector or update the
rolling daily state. It checks the existing source and baseline receipts, then
rebuilds feature matrices and predicts with the same saved final estimators.
"""
from __future__ import annotations

from datetime import timedelta
import hashlib
import importlib.metadata
import json
import re
from pathlib import Path
from pathlib import PureWindowsPath

import numpy as np
import pandas as pd

PROTOCOL = "nyx_saved_horizon_report_resume_v1"
ZONES = ("FR", "DE", "BE", "NL")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _receipt(path):
    raw = Path(path).read_bytes()
    value = json.loads(raw.decode("utf-8"))
    require(isinstance(value, dict) and Path(path).read_bytes() == raw,
            "Resume receipt changed during read")
    return value, hashlib.sha256(raw).hexdigest()


def _normalized_hashes(value, label):
    """Read Windows/POSIX relative manifest names without ambiguous aliases."""
    require(isinstance(value, dict) and value, label + ": nonempty artifact hashes required")
    normalized, aliases = {}, set()
    for relative, expected in value.items():
        require(isinstance(relative, str) and relative and not any(ord(char) < 32 for char in relative),
                label + ": invalid relative artifact path")
        name = relative.replace("\\", "/")
        parts = name.split("/")
        require(not name.startswith("/") and not PureWindowsPath(relative).drive
                and ":" not in name and all(part and part not in (".", "..")
                    and not part.endswith((".", " ")) for part in parts),
                label + ": absolute, traversing or ambiguous artifact path")
        alias = name.casefold()
        require(name not in normalized and alias not in aliases,
                label + ": normalized artifact path collision")
        require(isinstance(expected, str) and len(expected) == 64
                and all(char in "0123456789abcdef" for char in expected),
                label + ": invalid artifact SHA-256")
        normalized[name] = expected
        aliases.add(alias)
    return normalized


def _inside(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    require(path.is_relative_to(root) and path.is_file(), "Missing or redirected resume artifact: " + relative)
    return path


def _owned_report_directory(repo, candidate, owned_report_roots):
    """Accept annual reports or one configured app execution's exact outputs."""
    lexical = Path(candidate).expanduser().absolute()
    require(lexical.is_dir() and ".." not in lexical.parts,
            "Resume source must be an existing unambiguous report directory")
    for path in (lexical, *lexical.parents):
        require(not path.is_symlink() and not getattr(path, "is_junction", lambda: False)(),
                "Redirected resume source directory is forbidden")
    resolved = lexical.resolve()
    standard = Path(repo).resolve() / "runs/reports/nyx_annual_standard"
    if resolved.is_relative_to(standard):
        return resolved, {"kind": "annual_standard_reports", "root": str(standard)}
    require(isinstance(owned_report_roots, (tuple, list)),
            "Owned report roots must be configured execution directories")
    for value in owned_report_roots:
        root = Path(value).expanduser().absolute()
        for path in (root, *root.parents):
            require(not path.is_symlink() and not getattr(path, "is_junction", lambda: False)(),
                    "Redirected configured report root is forbidden")
        root = root.resolve()
        if not resolved.is_relative_to(root):
            continue
        parts = resolved.relative_to(root).parts
        require(len(parts) == 2 and parts[1] == "outputs"
                and re.fullmatch(r"[A-Za-z0-9_-]{1,100}", parts[0]) is not None,
                "Managed resume source must be an execution ID's exact outputs directory")
        require(not any(path.is_symlink() or getattr(path, "is_junction", lambda: False)()
                        for path in resolved.rglob("*")),
                "Managed resume source contains redirected artifacts")
        return resolved, {"kind": "configured_app_execution", "root": str(root), "run_id": parts[0]}
    raise ValueError("Resume source is outside owned standard reports and configured app executions")


def _bound_frame(root, relative, hashes):
    path = _inside(root, relative)
    require(relative in hashes and sha(path) == hashes[relative], "Resume artifact checksum differs: " + relative)
    frame = pd.read_parquet(path)
    require(sha(path) == hashes[relative], "Resume artifact changed during read: " + relative)
    index = frame.index
    require(isinstance(index, pd.DatetimeIndex) and str(index.tz) == "UTC" and len(index)
            and index.is_unique and index.is_monotonic_increasing and not index.hasnans
            and index.equals(index.floor("h")), "Resume artifact UTC hourly grid differs: " + relative)
    return frame


def _reporting_sources(failed, first_day, expected, baselines):
    root = failed / "report_only_sources"
    path = root / "reporting_sources/audit.json"
    if not path.is_file():
        from chronos2_hourly.nyx_annual_epex_prices import EPEX_SERIES
        observed = {zone: baselines[zone].actual.copy().rename("actual") for zone in ZONES}
        storm = {zone: pd.Series(np.nan, index=expected, name="storm") for zone in ZONES}
        audits = {zone: {"report_only": True, "observed": {
            "kind": "model_asof_epex_reporting_copy", "actual_reference": "EPEX",
            "actual_reference_label": "EPEX", "series": EPEX_SERIES[zone],
            "future_observations_available": False, "used_for_prediction": False,
            "used_for_training": False, "original_source_role": "model_source_snapshot"},
            "storm": {"status": "unavailable", "reason": "Original report-only Storm snapshot missing",
                "used_for_prediction": False, "used_for_training": False}} for zone in ZONES}
        return observed, storm, audits, {"status": "missing_snapshot_frozen_epex_reporting_copy",
            "storm_available": False, "provider_requests": 0}
    receipt, receipt_hash = _receipt(path)
    reporting_hashes = _normalized_hashes(receipt.get("artifact_sha256"), "Original reporting snapshot")
    names = {f"reporting_sources/{zone}/{kind}.parquet" for zone in ZONES for kind in ("observed", "storm")}
    require(receipt.get("protocol") == "nyx_saved_horizon_reporting_sources_v1"
            and receipt.get("first_forecast_day") == first_day
            and receipt.get("used_for_prediction") is False and receipt.get("used_for_training") is False
            and receipt.get("latest_reporting_extraction_not_model_asof_state") is True
            and set(reporting_hashes) == names
            and set(receipt.get("countries", {})) == set(ZONES), "Original reporting snapshot receipt differs")
    require(receipt.get("expected_hours") == len(expected)
            and pd.Timestamp(receipt["first_utc"]) == expected[0]
            and pd.Timestamp(receipt["last_utc"]) == expected[-1], "Original reporting support differs from resumed baseline")
    observed, storm = {}, {}
    for zone in ZONES:
        for kind, mapping in (("observed", observed), ("storm", storm)):
            frame = _bound_frame(root, f"reporting_sources/{zone}/{kind}.parquet", reporting_hashes)
            require(frame.shape[1] == 1 and frame.index.equals(expected)
                    and not np.isinf(frame.to_numpy(dtype=float)).any(), "Original report comparator grid/schema differs")
            mapping[zone] = frame.iloc[:, 0].copy()
    require(sha(path) == receipt_hash, "Original report source receipt changed during resumption")
    return observed, storm, receipt["countries"], {"status": "reused_original_extraction",
        "receipt_path": str(path), "receipt_sha256": receipt_hash,
        "extracted_at_utc": receipt["extracted_at_utc"], "provider_requests": 0}


def resume_horizon(repo, seed_bundle, seed_output, failed_report_directory, *, owned_report_roots=()):
    """Return the original J+1/J+2 calculation without any scientific refitting."""
    from chronos2_hourly import nyx_annual_cpu_baseline as annual
    from chronos2_hourly.nyx_annual_live_preflight import ROOT as installed_root
    from chronos2_hourly.nyx_annual_epex_prices import EPEX_SERIES, validate_price_identity
    from chronos2_hourly.nyx_saved_horizon_inference import (
        grid, assemble_horizon_features, infer_saved_horizon,
    )
    from chronos2_hourly.nyx_local_extra_features import cutoff_times
    from chronos2_hourly.nyx_daily_two_day_baseline import __file__ as baseline_code_file

    repo, seed_bundle, seed_output = map(lambda value: Path(value).resolve(),
        (repo, seed_bundle, seed_output))
    require(repo == Path(installed_root).resolve(), "Resume repository differs from installed annual pipeline")
    failed, ownership = _owned_report_directory(repo, failed_report_directory, owned_report_roots)
    require(not failed.is_relative_to(seed_bundle) and not failed.is_relative_to(seed_output),
            "Resume source must be a separate owned standard report directory")
    extension, baseline_root = failed / "forecast_extension", failed / "forecast_extension/baseline_extension"
    baseline_receipt_path = baseline_root / "receipt.json"
    baseline_receipt, baseline_receipt_hash = _receipt(baseline_receipt_path)
    source_receipt_path = extension / "horizon_sources/receipt.json"
    source_receipt, source_receipt_hash = _receipt(source_receipt_path)
    identity = baseline_receipt.get("identity", {})
    first_day = identity.get("first_delivery_day")
    require(isinstance(first_day, str) and pd.Timestamp(first_day).date().isoformat() == first_day,
            "Resumable calculation first delivery day missing")
    require(identity.get("protocol") == "nyx_daily_two_day_baseline_v1"
            and identity.get("second_delivery_day") == (pd.Timestamp(first_day).date() + timedelta(days=1)).isoformat(),
            "Resumable baseline is not the original two-day calculation")
    cutoff = pd.Timestamp(identity.get("asof_cutoff_utc"))
    require(not pd.isna(cutoff) and cutoff.tzinfo is not None, "Resumable forecast cutoff missing")
    cutoff = cutoff.tz_convert("UTC")
    required_cutoff = pd.Timestamp(str(pd.Timestamp(first_day).date() - timedelta(days=1)) + " 08:00", tz="Europe/Paris").tz_convert("UTC")
    require(cutoff == required_cutoff, "Resumable forecast cutoff is not its original J 08:00")
    require(source_receipt.get("protocol") == "nyx_saved_horizon_sources_v1"
            and source_receipt.get("forecast_first_day") == first_day and source_receipt.get("forecast_days") == 2
            and source_receipt.get("shared_cutoff_utc") == cutoff.isoformat()
            and source_receipt.get("future_labels_used") is False,
            "Resumable source and baseline dates/cutoffs differ")
    source_audit = {name: value for name, value in source_receipt.items() if name != "source_artifact_sha256"}
    require(identity.get("source_audit") == source_audit, "Resumable baseline did not consume this recorded source audit")
    require(baseline_receipt.get("audit", {}).get("future_labels_used") is False
            and baseline_receipt["audit"].get("annual_835_day_replay_run") is False,
            "Resumable baseline provenance is incomplete")

    seed_report, seed_report_hash = _receipt(seed_output / "receipt.json")
    require(seed_report.get("status") == "REPORTS_COMPLETE" and seed_report.get("future_labels_used") is False
            and isinstance(seed_report.get("input_sha256"), dict) and seed_report["input_sha256"],
            "Completed seed model receipt missing")
    bound_inputs = _normalized_hashes(seed_report["input_sha256"], "Completed seed inputs")
    for relative, expected in bound_inputs.items():
        require(sha(_inside(seed_bundle, relative)) == expected, "Original training input checksum differs: " + relative)
    seed_baseline, seed_baseline_hash = _receipt(seed_bundle / "source_receipts/nyx_quantiles.json")
    seed_baseline_hashes = _normalized_hashes(seed_baseline.get("artifact_sha256"), "Completed seed baseline")
    require(identity.get("seed_receipt_sha256") == seed_baseline_hash,
            "Resumable baseline belongs to a different completed annual seed")
    seed_saturn, seed_saturn_hash = _receipt(seed_bundle / "source_receipts/saturn.json")
    require(seed_baseline.get("source_receipts_sha256", {}).get("saturn") == seed_saturn_hash,
            "Completed seed baseline/source receipt linkage differs")
    validate_price_identity(seed_saturn)
    require(seed_saturn.get("price_series") == dict(EPEX_SERIES), "Resumable forecast must retain the four EPEX sources")
    require(identity.get("cpu_threads") == 2
            and identity.get("dependency_sha256") == {name: annual.sha256_text(annual.ROOT / name) for name in annual.CODE_FILES}
            and identity.get("configuration_sha256") == {zone: annual._configuration(zone, 2)[1] for zone in ZONES}
            and identity.get("implementation_sha256") == annual.sha256_text(Path(baseline_code_file)),
            "Resumable scientific implementation or fixed configuration differs")
    require(identity.get("runtime") == {name: importlib.metadata.version(name) for name in
        ("torch", "chronos-forecasting", "numpy", "pandas", "catboost", "pykalman", "scipy", "holidays")},
        "Resumable scientific environment differs")

    source_names = {"horizon_sources/" + name + ".parquet" for name in
        ("covariates", "fuel", "hydro_features", "exchange_features", "jao_original", "jao_refreshed")}
    source_names |= {f"horizon_sources/{group}/{zone}.parquet" for group in ("prices", "thermal_features") for zone in ZONES}
    source_hashes = _normalized_hashes(source_receipt.get("source_artifact_sha256"), "Resumable source")
    require(set(source_hashes) == source_names,
            "Resumable source artifact inventory differs")
    source = {name: _bound_frame(extension, f"horizon_sources/{name}.parquet", source_hashes) for name in
        ("covariates", "fuel", "hydro_features", "exchange_features", "jao_original", "jao_refreshed")}
    source["thermal_features"] = {zone: _bound_frame(extension, f"horizon_sources/thermal_features/{zone}.parquet", source_hashes) for zone in ZONES}
    source["prices"] = {}
    current = grid(first_day)
    for zone in ZONES:
        frame = _bound_frame(extension, f"horizon_sources/prices/{zone}.parquet", source_hashes)
        require(list(frame) == ["price_eur_mwh"] and (frame.index < current[0]).all()
                and np.isfinite(frame.to_numpy(float)).all(), "Resumable source contains future or nonfinite actual prices")
        source["prices"][zone] = frame.price_eur_mwh
    require(identity.get("covariates_sha256") == annual.frame_digest(source["covariates"])
            and identity.get("targets") == {zone: annual.frame_digest(source["prices"][zone]) for zone in ZONES},
            "Resumable baseline numerical inputs differ from recorded sources")
    source["audit"] = source_audit

    names = {f"{zone}/{name}.parquet" for zone in ZONES for name in
        ("chronos", "residual", "baseline", "history", "raw_history", "residual_history")}
    names |= {f"{zone}/{name}" for zone in ZONES for name in
        ("kalman_inference_state.json", "residual_corrector.joblib", "future_interaction.parquet")}
    baseline_hashes = _normalized_hashes(baseline_receipt.get("artifact_sha256"), "Resumable baseline")
    require(set(baseline_hashes) == names, "Resumable baseline artifact inventory differs")
    for relative, expected in baseline_hashes.items():
        require(sha(_inside(baseline_root, relative)) == expected, "Resumable baseline artifact checksum differs: " + relative)
    covariance = source["covariates"]
    require(np.isfinite(covariance.to_numpy(float)).all(), "Resumable covariate history contains missing values")
    logical = cutoff_times(covariance.index, "Europe/Paris")
    donor = _bound_frame(seed_bundle, "source_artifacts/saturn/origins.parquet", bound_inputs)
    require("forecast_origin_utc" in donor, "Original logical profile origins missing")
    retained = logical.index.intersection(donor.index)
    logical.loc[retained] = pd.to_datetime(donor.loc[retained, "forecast_origin_utc"], utc=True)
    logical.loc[current] = cutoff
    quantiles, origins, baselines = {}, {}, {}
    expected_report_index = pd.date_range(pd.Timestamp(first_day).date() - timedelta(days=365),
        pd.Timestamp(first_day).date() + timedelta(days=2), tz="Europe/Paris", freq="h", inclusive="left").tz_convert("UTC")
    for zone in ZONES:
        prior = _bound_frame(baseline_root, f"{zone}/history.parquet", baseline_hashes)
        future = _bound_frame(baseline_root, f"{zone}/baseline.parquet", baseline_hashes)
        require(future.index.equals(current) and {"nyx__q10", "nyx__q50", "nyx__q90", "actual", "forecast_origin_utc"} <= set(future)
                and future.actual.isna().all() and pd.to_datetime(future.forecast_origin_utc, utc=True).eq(cutoff).all(),
                "Resumable future baseline grid/labels/origins differ")
        require(prior.loc[current, "actual"].isna().all(), "Resumable history contains realized future labels")
        combined = pd.concat([prior.loc[~prior.index.isin(current)], future]).sort_index()
        qcolumns = ["nyx__q10", "nyx__q50", "nyx__q90"]
        require(combined.index.is_unique and np.isfinite(combined[qcolumns].to_numpy(float)).all()
                and (np.diff(combined[qcolumns].to_numpy(float), axis=1) >= 0).all(), "Resumable baseline quantiles invalid")
        stamps = pd.to_datetime(combined.forecast_origin_utc, utc=True)
        require(stamps.notna().all() and (stamps <= cutoff).all(), "Resumable prediction origin after information cutoff")
        combined["actual"] = source["prices"][zone].reindex(combined.index)
        combined.loc[current, "actual"] = np.nan
        quantiles[zone] = combined[qcolumns].rename(columns=lambda column: column[5:])
        origins[zone] = logical.copy()
        require(expected_report_index.isin(combined.index).all(), "Resumable report lacks 365 historical days and two future days")
        baselines[zone] = combined.loc[expected_report_index].copy()
        require(np.isfinite(baselines[zone].loc[baselines[zone].index < current[0], "actual"].to_numpy(float)).all(),
                "Resumable historical report labels incomplete")
        audit_relative = f"baseline_audits/{zone}.json"
        audit_path = _inside(seed_bundle, audit_relative)
        require(seed_baseline_hashes.get(audit_relative) == sha(audit_path), "Seed Chronos audit checksum differs")
        require(_receipt(audit_path)[0]["chronos_model_sha256"] == baseline_receipt["audit"]["zones"][zone]["chronos_model_sha256"],
                "Resumable baseline uses different pretrained Chronos weights")

    built = assemble_horizon_features(first_day, cutoff, sources=source, quantiles=quantiles, origins=origins)
    forecasts, model_audit = infer_saved_horizon(seed_bundle, seed_output, first_day, cutoff, built, quantiles)
    observed, storm, reporting_audits, reporting_resume = _reporting_sources(failed, first_day, expected_report_index, baselines)
    require(sha(baseline_receipt_path) == baseline_receipt_hash and sha(source_receipt_path) == source_receipt_hash
            and sha(seed_output / "receipt.json") == seed_report_hash, "Receipts changed during report resumption")
    for root, hashes in ((seed_bundle, bound_inputs), (extension, source_hashes),
                         (baseline_root, baseline_hashes)):
        for relative, expected in hashes.items():
            require(sha(_inside(root, relative)) == expected, "An existing resume input changed during prediction")
    resume_audit = {"protocol": PROTOCOL, "failed_report_directory": str(failed),
        "source_ownership": ownership,
        "source_receipt_sha256": source_receipt_hash, "baseline_receipt_sha256": baseline_receipt_hash,
        "forecast_first_day": first_day, "forecast_days": 2, "original_cutoff_utc": cutoff.isoformat(),
        "chronos_calls": 0, "corrector_fits": 0, "kalman_fits": 0, "final_expert_fits": 0,
        "model_source_fetches": 0, "rolling_state_updates": 0,
        "receipt_path_separator_normalization": "relative Windows or POSIX names; collisions and traversal rejected",
        "saved_model_inference_repeated": True, "reporting_snapshots": reporting_resume}
    model_audit["bounded_baseline"] = baseline_receipt["audit"]
    model_audit["report_resume"] = resume_audit
    context = {"delivery_day": first_day, "asof_utc": cutoff.isoformat(),
        "source_capture_started_at_utc": source_audit.get("generated_at_utc"),
        "original_requested_at_utc_available": False, "resumed_completed_forecast": True}
    return {"forecasts": forecasts, "baseline": baselines, "covariates": covariance,
        "model_audit": model_audit, "sources_audit": source_receipt, "cutoff": cutoff,
        "first_delivery_day": first_day, "forecast_days": 2, "context": context,
        "observed": observed, "storm": storm, "source_audits": reporting_audits,
        "resume_audit": resume_audit}
