"""Current-cutoff training inputs for daily refits of every NYX expert.

The sealed annual run is an immutable source of historical forecasts, not a
source of newly fitted estimators. Targets always come from this launch's
receipted EPEX extraction. Recovered historical profiles and bridge forecasts
remain explicitly current-fit reconstructions, never certified past vintages.
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

PROTOCOL = "nyx_daily_refit_training_v1"
CACHE_PROTOCOL = "nyx_daily_refit_history_cache_v1"
ZONES = ("FR", "DE", "BE", "NL")
QUANTILES = ("q10", "q50", "q90")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2,
        ensure_ascii=False, allow_nan=False, default=str) + "\n", encoding="utf-8")


def _utc(value, label):
    stamp = pd.Timestamp(value)
    require(not pd.isna(stamp) and stamp.tzinfo is not None,
            label + ": explicit timezone required")
    return stamp.tz_convert("UTC")


def _grid(first, stop):
    return pd.date_range(str(first), str(stop), tz="Europe/Paris", freq="h",
                         inclusive="left").tz_convert("UTC")


def _hourly(frame, label, *, finite=False):
    index = frame.index
    require(isinstance(index, pd.DatetimeIndex) and str(index.tz) == "UTC"
        and len(index) and index.is_unique and index.is_monotonic_increasing
        and not index.hasnans and index.equals(index.floor("h")),
        label + ": ordered unique UTC physical hours required")
    if finite:
        require(np.isfinite(frame.to_numpy(dtype=float)).all(),
                label + ": finite numeric values required")


def _normalized_hashes(hashes):
    require(isinstance(hashes, dict) and hashes, "Nonempty artifact hashes required")
    result, folded = {}, set()
    for relative, value in hashes.items():
        require(isinstance(relative, str) and isinstance(value, str)
            and len(value) == 64 and all(c in "0123456789abcdef" for c in value),
            "Malformed artifact hash")
        name = relative.replace("\\", "/")
        parts = name.split("/")
        require(parts and not name.startswith("/") and not any(
            part in ("", ".", "..") or ":" in part for part in parts)
            and "//" not in name and not name.endswith("/")
            and name.casefold() not in folded, "Unsafe or duplicate artifact path")
        result[name] = value
        folded.add(name.casefold())
    return result


def _bound_frame(root, relative, hashes):
    root = Path(root).resolve()
    require(relative in hashes, "Unreceipted daily training input: " + relative)
    lexical = root / relative
    require(not any(p.is_symlink() or getattr(p, "is_junction", lambda: False)()
                    for p in (lexical, *lexical.parents) if p != root.parent),
            "Redirected daily training artifact")
    path = lexical.resolve()
    require(path.is_relative_to(root) and path.is_file() and sha(path) == hashes[relative],
            "Changed daily training input: " + relative)
    result = pd.read_parquet(path)
    require(sha(path) == hashes[relative], "Daily training input changed during read")
    _hourly(result, relative)
    return result


def _merge_frames(*frames):
    """Later explicit rows win; missing values are never forward-filled."""
    frames = [frame for frame in frames if frame is not None and len(frame)]
    require(bool(frames), "At least one historical frame required")
    columns = list(frames[0].columns)
    for frame in frames:
        _hourly(frame, "Merged training source")
        require(list(frame.columns) == columns, "Merged training source schema differs")
    result = pd.concat(frames)
    return result.loc[~result.index.duplicated(keep="last")].sort_index()


def _missing(index, columns):
    result = pd.DataFrame(np.nan, index=index, columns=list(columns))
    for name in columns:
        if name.endswith("__available"):
            result[name] = 0.0
    return result


def _preserved_oof(donor, cached):
    if cached is not None:
        overlap = donor.index.intersection(cached.index)
        require(list(cached) == list(donor) and np.array_equal(
            donor.loc[overlap].to_numpy(float), cached.loc[overlap].to_numpy(float)),
            "Retained reference state changed the donor OOF predictions")
    return _merge_frames(donor, cached)


def _fresh_actual(prices, first_day, support):
    first = _grid(first_day, first_day + timedelta(days=1))[0]
    require(set(prices) == set(ZONES), "Four fresh EPEX target histories required")
    output = {}
    past = support[support < first]
    for zone in ZONES:
        frame = prices[zone]
        _hourly(frame, zone + " fresh EPEX prices", finite=True)
        require(isinstance(frame, pd.Series) and not (frame.index >= first).any(),
                zone + ": future target rows are forbidden")
        result = frame.reindex(support).astype(float)
        require(np.isfinite(result.loc[past].to_numpy()).all(),
                zone + ": incomplete fresh EPEX training labels")
        require(result.loc[support >= first].isna().all(),
                zone + ": future labels entered daily refit")
        output[zone] = result.rename("actual")
    return output


def _contract(repo, seed_bundle, seed_output):
    from chronos2_hourly import nyx_annual_live_preflight as gate
    modules = ("nyx_daily_refit_history.py", "nyx_daily_refit_reference.py",
               "nyx_daily_refit_experts.py", "nyx_daily_refit.py")
    directory = Path(__file__).resolve().parent
    codes = {name: sha(directory / name) for name in modules}
    canonical = {name: sha(Path(repo) / name) for name in gate.MATERIALIZER_CODE}
    return {"protocol": CACHE_PROTOCOL, "seed_report_receipt_sha256": sha(seed_output / "receipt.json"),
        "seed_materialization_sha256": sha(seed_bundle / gate.MATERIALIZATION_PATH),
        "schema_sha256": sha(gate.SCHEMA), "daily_refit_code_sha256": codes,
        "canonical_feature_code_sha256": canonical,
        "runtime": {name: importlib.metadata.version(name) for name in
                    ("numpy", "pandas", "scikit-learn", "catboost", "joblib")},
        "target_policy": "fresh_epex_only_every_launch", "training_days": 365,
        "reference_support_days": 462, "cached_targets_used": False}


def _cache_state(root, contract, cutoff, first_day):
    if not root.exists():
        return None
    require(root.is_dir() and not root.is_symlink(), "Redirected daily refit cache")
    choices = []
    for directory in root.iterdir():
        if directory.name.startswith(".") or not directory.is_dir():
            continue
        require(not directory.is_symlink() and not getattr(directory, "is_junction", lambda: False)(),
                "Redirected daily refit state")
        path = directory / "receipt.json"
        if not path.is_file():
            continue
        receipt = _json(path)
        if receipt.get("contract") != contract:
            continue
        stamp = _utc(receipt.get("asof_cutoff_utc"), "Daily refit cache cutoff")
        if stamp > cutoff or pd.Timestamp(receipt.get("first_delivery_day")).date() > first_day:
            continue
        require(receipt.get("protocol") == CACHE_PROTOCOL and receipt.get("state") == "COMPLETE"
            and receipt.get("future_labels_used") is False and receipt.get("cached_targets_used") is False,
            "Invalid completed daily refit state")
        choices.append((stamp, receipt.get("created_at_utc", ""), directory, receipt))
    if not choices:
        return None
    _, _, directory, receipt = max(choices, key=lambda row: (row[0], row[1], row[2].name))
    hashes = _normalized_hashes(receipt.get("artifact_sha256"))
    expected = {f"reference_oof/{zone}.parquet" for zone in ZONES}
    expected |= {f"sources/{name}.parquet" for name in
                 ("hydro_features", "exchange_features", "jao_original", "jao_refreshed")}
    expected |= {f"sources/thermal_features/{zone}.parquet" for zone in ZONES}
    require(set(hashes) == expected, "Daily refit cache inventory differs")
    frames = {name: _bound_frame(directory, name, hashes) for name in sorted(hashes)}
    return {"frames": frames, "receipt_path": str(directory / "receipt.json"),
            "receipt_sha256": sha(directory / "receipt.json"), "receipt": receipt}


def _persist_training(root, history):
    root = Path(root)
    require(not root.exists(), "Daily refit training output already exists")
    root.mkdir(parents=True)
    built = history["built"]
    frames = {}
    for family, mapping in built.features.items():
        frames.update({f"features/{family}/{zone}.parquet": frame for zone, frame in mapping.items()})
    for label, mapping in (("base292", built.base292), ("augmented334", built.augmented334),
                           ("test2", built.test2_features)):
        frames.update({f"reference_inputs/{label}/{zone}.parquet": frame for zone, frame in mapping.items()})
    for label in ("quantiles", "actual", "prior_oof"):
        for zone, frame in history[label].items():
            frames[f"{label}/{zone}.parquet"] = frame.to_frame() if isinstance(frame, pd.Series) else frame
    hashes = {}
    for relative, frame in frames.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path)
        hashes[relative] = sha(path)
    receipt = {"protocol": PROTOCOL, "state": "COMPLETE", **history["evidence"],
               "artifact_sha256": hashes, "feature_assembly": built.audits}
    _write_json(root / "receipt.json", receipt)
    history["training_receipt"] = {"path": str(root / "receipt.json"),
        "sha256": sha(root / "receipt.json"), "artifact_sha256": hashes}


def prepare_training_history(repo, seed_bundle, seed_output, horizon, output_dir, *, progress=None):
    """Build exactly 365 expert days, 462 reference days and both future days.

    ``output_dir`` is the existing forecast-extension directory containing the
    two immutable baseline/source receipts produced for this launch. No cached
    target column or previous final/reference estimator is read here.
    """
    from chronos2_hourly import nyx_annual_live_preflight as gate
    from chronos2_hourly.nyx_annual_cpu_bundle_builder import BuiltFeatures, build_feature_matrices
    from chronos2_hourly.nyx_local_extra_features import cutoff_times
    from chronos2_hourly.nyx_saved_horizon_inference import assemble_horizon_features, grid
    from chronos2_hourly import nyx_thermal_capacity_features as thermal
    from chronos2_hourly.nyx_annual_prior90_cpu import FEATURES as OOF_COLUMNS

    repo, seed_bundle, seed_output, extension = map(lambda value: Path(value).resolve(),
        (repo, seed_bundle, seed_output, output_dir))
    require(repo == Path(gate.ROOT).resolve(), "Daily refit repository differs from annual pipeline")
    require(not extension.is_relative_to(seed_bundle) and not extension.is_relative_to(seed_output),
            "Daily refit cannot overwrite the completed annual run")
    first = pd.Timestamp(horizon["first_delivery_day"]).date()
    cutoff = _utc(horizon["cutoff"], "Daily refit cutoff")
    require(cutoff == pd.Timestamp(f"{first - timedelta(days=1)} 08:00", tz="Europe/Paris").tz_convert("UTC"),
            "Daily refit must use the shared first-day D-1 08:00 cutoff")
    current = grid(str(first))
    expert_index = _grid(first - timedelta(days=365), first + timedelta(days=2))
    reference_index = _grid(first - timedelta(days=462), first + timedelta(days=2))
    one_day_reference = reference_index[reference_index < _grid(first + timedelta(days=1), first + timedelta(days=2))[0]]
    one_day_expert = expert_index[expert_index < _grid(first + timedelta(days=1), first + timedelta(days=2))[0]]
    seed_report_path, source_receipt_path = seed_output / "receipt.json", extension / "horizon_sources/receipt.json"
    baseline_receipt_path = extension / "baseline_extension/receipt.json"
    before = {str(path): sha(path) for path in (seed_report_path, source_receipt_path, baseline_receipt_path)}
    seed_report = _json(seed_report_path)
    require(seed_report.get("status") == "REPORTS_COMPLETE" and seed_report.get("future_labels_used") is False,
            "Completed annual donor report required")
    donor_hashes = _normalized_hashes(seed_report.get("input_sha256"))
    source_receipt, baseline_receipt = _json(source_receipt_path), _json(baseline_receipt_path)
    require(source_receipt.get("protocol") == "nyx_saved_horizon_sources_v1"
        and source_receipt.get("future_labels_used") is False
        and source_receipt.get("forecast_first_day") == str(first)
        and _utc(source_receipt.get("shared_cutoff_utc"), "Source cutoff") == cutoff,
        "Daily refit source receipt identity differs")
    identity = baseline_receipt.get("identity", {})
    require(identity.get("protocol") == "nyx_daily_two_day_baseline_v1"
        and identity.get("first_delivery_day") == str(first)
        and identity.get("seed_receipt_sha256") == sha(seed_bundle / "source_receipts/nyx_quantiles.json")
        and _utc(identity.get("asof_cutoff_utc"), "Baseline cutoff") == cutoff
        and baseline_receipt.get("audit", {}).get("future_labels_used") is False,
        "Daily refit baseline receipt identity differs")
    source_hashes = _normalized_hashes(source_receipt.get("source_artifact_sha256"))
    source = {name: _bound_frame(extension, f"horizon_sources/{name}.parquet", source_hashes)
              for name in ("covariates", "fuel", "hydro_features", "exchange_features", "jao_original", "jao_refreshed")}
    source["prices"], source["thermal_features"] = {}, {}
    for zone in ZONES:
        prices = _bound_frame(extension, f"horizon_sources/prices/{zone}.parquet", source_hashes)
        require(list(prices) == ["price_eur_mwh"], "Fresh EPEX price schema differs")
        source["prices"][zone] = prices.price_eur_mwh
        source["thermal_features"][zone] = _bound_frame(extension,
            f"horizon_sources/thermal_features/{zone}.parquet", source_hashes)
    source["audit"] = source_receipt
    actual = _fresh_actual(source["prices"], first, reference_index)
    contract = _contract(repo, seed_bundle, seed_output)
    cache_root = repo / "runs/nyx_daily_refit_states" / contract["seed_report_receipt_sha256"]
    cache = _cache_state(cache_root, contract, cutoff, first)
    cached = cache["frames"] if cache else {}
    baseline_hashes = _normalized_hashes(baseline_receipt.get("artifact_sha256"))
    quantiles, prediction_origins = {}, {}
    for zone in ZONES:
        frame = _bound_frame(extension / "baseline_extension", f"{zone}/history.parquet", baseline_hashes)
        require({"nyx__q10", "nyx__q50", "nyx__q90", "forecast_origin_utc", "actual"} <= set(frame),
                "Baseline training history schema differs")
        require(reference_index.isin(frame.index).all(), zone + ": 462-day baseline support missing")
        quantile = frame[["nyx__" + q for q in QUANTILES]].rename(columns=lambda name: name[5:])
        _hourly(quantile, zone + " baseline quantiles", finite=True)
        require((np.diff(quantile.to_numpy(float), axis=1) >= 0).all(), "Crossed baseline quantiles")
        origins = pd.to_datetime(frame.forecast_origin_utc, utc=True)
        require(origins.notna().all() and (origins <= cutoff).all()
            and origins.loc[current].eq(cutoff).all() and frame.loc[current, "actual"].isna().all(),
            "Baseline future labels/origins differ")
        quantiles[zone], prediction_origins[zone] = quantile, origins
    covariates = source["covariates"]
    _hourly(covariates, "Daily refit covariates", finite=True)
    logical = cutoff_times(covariates.index, "Europe/Paris")
    donor_origins = _bound_frame(seed_bundle, "source_artifacts/saturn/origins.parquet", donor_hashes)
    overlap = logical.index.intersection(donor_origins.index)
    logical.loc[overlap] = pd.to_datetime(donor_origins.loc[overlap, "forecast_origin_utc"], utc=True)
    origins = {zone: logical.copy() for zone in ZONES}

    # Historical auxiliary values retain original captured bytes or a prior
    # completed launch's receipt. Uncaptured bridge rows stay explicitly absent.
    auxiliary = {}
    donor_paths = {"hydro_features": "source_artifacts/public_hydro/features.parquet",
        "exchange_features": "source_artifacts/lagged_exchange/features.parquet",
        "jao_original": "source_artifacts/jao_initial/original_features.parquet",
        "jao_refreshed": "source_artifacts/jao_initial/refreshed_features.parquet"}
    for name, relative in donor_paths.items():
        if name.startswith("jao_") and relative not in donor_hashes:
            relative = "source_artifacts/jao_initial/features_365d_plus_delivery.parquet"
        donor = _bound_frame(seed_bundle, relative, donor_hashes)
        auxiliary[name] = _merge_frames(donor, cached.get(f"sources/{name}.parquet"), source[name])
        missing = one_day_expert[~one_day_expert.isin(auxiliary[name].index)]
        if len(missing):
            auxiliary[name] = _merge_frames(auxiliary[name], _missing(missing, auxiliary[name].columns))

    # Normalize both historical thermal collector formats to the canonical
    # prospective Pmax format. This allows missing bridge capacities without
    # inventing finite values or weakening the trained availability flags.
    thermal_sources = {}
    for name in thermal.SOURCES:
        donor = _bound_frame(seed_bundle, f"source_artifacts/thermal_capacity/{name}.parquet", donor_hashes)
        if list(donor) == ["pmax_gw", "asof_query_utc"]:
            values = donor.pmax_gw
        else:
            values = thermal.validate_source(donor, name)
        levels = values.to_frame("pmax_gw")
        cached_thermal = cached.get("sources/thermal_features/FR.parquet")
        if cached_thermal is not None:
            levels = _merge_frames(levels, cached_thermal[["thermal__" + name + "_pmax_gw"]].rename(
                columns={"thermal__" + name + "_pmax_gw": "pmax_gw"}))
        fresh = source["thermal_features"]["FR"][["thermal__" + name + "_pmax_gw"]].rename(
            columns={"thermal__" + name + "_pmax_gw": "pmax_gw"})
        levels = _merge_frames(levels, fresh).reindex(one_day_expert)
        levels["asof_query_utc"] = cutoff_times(one_day_expert, "Europe/Paris").to_numpy()
        thermal_sources[name] = levels
    if progress:
        progress("training_history", status="running", phase="assemble", training_days=365)
    built = build_feature_matrices(str(first), prices=source["prices"], nyx_quantiles=quantiles,
        forecast_origins=origins, covariates=covariates, fuel=source["fuel"],
        hydro_features=auxiliary["hydro_features"], exchange_features=auxiliary["exchange_features"],
        jao_original=auxiliary["jao_original"], jao_refreshed=auxiliary["jao_refreshed"],
        thermal_sources=thermal_sources, feature_index=one_day_expert, reference_index=one_day_reference)
    future_origins = {zone: value.copy() for zone, value in origins.items()}
    for zone in ZONES:
        future_origins[zone].loc[current] = cutoff
    future = assemble_horizon_features(str(first), cutoff, sources=source,
        quantiles=quantiles, origins=future_origins)
    all_features = {family: {zone: _merge_frames(built.features[family][zone], future.features[family][zone]).loc[expert_index]
        for zone in ZONES} for family in gate.FAMILIES}
    base = {zone: _merge_frames(built.base292[zone], future.base292[zone]).loc[reference_index] for zone in ZONES}
    augmented = {zone: _merge_frames(built.augmented334[zone], future.augmented334[zone]).loc[reference_index] for zone in ZONES}
    test2 = {zone: _merge_frames(built.test2_features[zone], future.test2_features[zone]).loc[reference_index] for zone in ZONES}
    for family, mapping in all_features.items():
        for zone, frame in mapping.items():
            gate.validate_feature_frame(frame, gate.load_schema()["families"][family]["columns"][zone],
                                        expert_index, f"daily refit {family}/{zone}")
    prior_oof = {}
    for zone in ZONES:
        donor = _bound_frame(seed_bundle, f"reference_oof/{zone}.parquet", donor_hashes)
        require(set(OOF_COLUMNS) <= set(donor), "Seed reference OOF schema differs")
        cached_oof = cached.get(f"reference_oof/{zone}.parquet")
        if cached_oof is not None:
            require(list(cached_oof) == list(OOF_COLUMNS), "Cached reference OOF schema differs")
        oof = _preserved_oof(donor.loc[:, list(OOF_COLUMNS)], cached_oof)
        prior_oof[zone] = oof.loc[oof.index < current[0]].copy()
        _hourly(prior_oof[zone], "Historical reference OOF", finite=True)
    logical_limits = cutoff_times(reference_index, "Europe/Paris")
    late = {zone: int((prediction_origins[zone].reindex(reference_index) > logical_limits).sum()) for zone in ZONES}
    evidence = {"first_delivery_day": str(first), "second_delivery_day": str(first + timedelta(days=1)),
        "asof_cutoff_utc": cutoff.isoformat(), "training_days": 365, "reference_support_days": 462,
        "future_labels_used": False, "cached_targets_used": False, "target_policy": "fresh_epex_only_every_launch",
        "qualified_for_production": False, "historical_original_origins_certified": False,
        "historical_profile_policy": "retained donor/cached profiles plus declared current-cutoff bridge reconstruction",
        "logical_feature_origins_are_provider_vintages": False,
        "baseline_rows_generated_after_logical_daily_origin": late,
        "prior_oof_policy": "seed/cached past-only HGB-Test2 reference predictions, never final annual experts",
        "source_receipt_sha256": before[str(source_receipt_path)],
        "baseline_receipt_sha256": before[str(baseline_receipt_path)],
        "seed_report_receipt_sha256": before[str(seed_report_path)],
        "retained_reference_state": {key: cache[key] for key in ("receipt_path", "receipt_sha256")} if cache else None,
        "public_bridge_descriptors": "missing values retain NaN and explicit zero availability; no imputation"}
    audits = {"protocol": PROTOCOL, **evidence, "historical": built.audits, "future": future.audits}
    result = {"built": BuiltFeatures(all_features, base, augmented, test2, audits),
        "quantiles": quantiles, "actual": actual, "prior_oof": prior_oof,
        "sources": source, "evidence": evidence,
        "cache": {"root": str(cache_root), "contract": contract,
                  "sources": {**auxiliary, "thermal_features": {zone: _merge_frames(
                      cached.get(f"sources/thermal_features/{zone}.parquet"), source["thermal_features"][zone])
                      for zone in ZONES}}}}
    require(before == {name: sha(name) for name in before}, "Training receipts changed during assembly")
    for root, hashes in ((seed_bundle, donor_hashes), (extension, source_hashes),
                         (extension / "baseline_extension", baseline_hashes)):
        for relative, expected in hashes.items():
            path = root / relative
            require(path.is_file() and sha(path) == expected,
                    "Training input changed during assembly: " + relative)
    _persist_training(extension / "daily_refit_training", result)
    return result


def publish_training_state(history, *, reference_oof):
    """Commit only source descriptors and genuine past OOF after fresh fits.

    A completed state never stores training labels or fitted estimators. Its
    creation does not authorize skipping any model fit on a later launch.
    """
    from chronos2_hourly.nyx_annual_prior90_cpu import FEATURES
    cache, evidence = history["cache"], history["evidence"]
    root = Path(cache["root"])
    root.mkdir(parents=True, exist_ok=True)
    require(not root.is_symlink() and not getattr(root, "is_junction", lambda: False)(),
            "Redirected daily refit cache root")
    first = pd.Timestamp(evidence["first_delivery_day"], tz="Europe/Paris").tz_convert("UTC")
    require(set(reference_oof) == set(ZONES), "Four updated OOF histories required")
    frames = {}
    for zone, frame in reference_oof.items():
        _hourly(frame, "Updated reference OOF", finite=True)
        require(list(frame) == list(FEATURES) and (frame.index < first).all(),
                "Updated OOF must contain only canonical historical reference rows")
        frames[f"reference_oof/{zone}.parquet"] = frame
    for name in ("hydro_features", "exchange_features", "jao_original", "jao_refreshed"):
        frames[f"sources/{name}.parquet"] = cache["sources"][name]
    for zone, frame in cache["sources"]["thermal_features"].items():
        frames[f"sources/thermal_features/{zone}.parquet"] = frame
    stamp = _utc(evidence["asof_cutoff_utc"], "Publication cutoff")
    name = stamp.strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:12]
    attempt, final = root / ("." + name + ".attempt"), root / name
    attempt.mkdir()
    try:
        hashes = {}
        for relative, frame in frames.items():
            path = attempt / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(path)
            hashes[relative] = sha(path)
        receipt = {"protocol": CACHE_PROTOCOL, "state": "COMPLETE", "contract": cache["contract"],
            "first_delivery_day": evidence["first_delivery_day"], "asof_cutoff_utc": stamp.isoformat(),
            "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(), "artifact_sha256": hashes,
            "future_labels_used": False, "cached_targets_used": False,
            "model_fits_may_be_skipped": False, "training_receipt": history["training_receipt"]}
        _write_json(attempt / "receipt.json", receipt)
        attempt.rename(final)
    except BaseException:
        shutil.rmtree(attempt, ignore_errors=True)
        raise
    return {"receipt_path": str(final / "receipt.json"), "receipt_sha256": sha(final / "receipt.json"),
            "new_state_published": True, "model_weights_reused": False}


__all__ = ["prepare_training_history", "publish_training_state"]
