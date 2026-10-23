"""Daily CPU fitting of NYX's seven final experts and fresh calibration.

Historical feature production and the HGB/Test2 reference are supplied by the
caller. This module does not collect sources, refit the Chronos foundation
model, or load the seven final estimators from an older run.
"""
from __future__ import annotations

from datetime import date, timedelta
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd


PROTOCOL = "nyx_daily_refit_final_experts_v1"
ZONES = ("FR", "DE", "BE", "NL")
PRICE_EXPERTS = ("fr_residual_1000", "cwe_residual_2000", "cwe_absolute_2000")
PRICE_THREADS = 8
NEGATIVE_THREADS = 2


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(first, stop):
    return pd.date_range(str(first), str(stop), tz="Europe/Paris", freq="h",
                         inclusive="left").tz_convert("UTC")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists(), f"Daily expert artifact already exists: {path.name}")
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                               allow_nan=False, default=str) + "\n", encoding="utf-8")


def _save_classifier(estimator, path):
    path = Path(path)
    require(not path.exists(), "Daily classifier destination already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, filename = tempfile.mkstemp(prefix=path.stem + ".", suffix=".tmp.cbm",
                                        dir=path.parent)
    os.close(handle)
    temporary = Path(filename)
    try:
        estimator.save_model(str(temporary), format="cbm")
        require(temporary.stat().st_size > 0, "Empty daily negative-price CBM")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return {"saved": True, "path": str(path), "sha256": _sha(path),
            "format": "cbm", "bytes": path.stat().st_size}


def _aligned(value, index, label, *, finite=True):
    require(isinstance(value, pd.Series) and value.index.equals(index),
            f"Aligned {label} required")
    if finite:
        require(np.isfinite(value.to_numpy(dtype=float)).all(),
                f"Finite {label} required")


def _validate(first_day, cutoff, built, quantiles, actual, references, distributions):
    first = date.fromisoformat(str(first_day))
    require(first.isoformat() == str(first_day), "Expected YYYY-MM-DD delivery day")
    cutoff = pd.Timestamp(cutoff)
    expected_cutoff = pd.Timestamp(f"{first - timedelta(days=1)} 08:00",
                                   tz="Europe/Paris").tz_convert("UTC")
    require(not pd.isna(cutoff) and cutoff.tzinfo is not None
            and cutoff.tz_convert("UTC") == expected_cutoff,
            "Daily experts require the shared first-day D-1 08:00 cutoff")
    train = _grid(first - timedelta(days=365), first)
    current = _grid(first, first + timedelta(days=2))
    full = train.append(current)
    require(set(built.features) == set(PRICE_EXPERTS),
            "Exactly three final expert feature families required")
    require(all(set(mapping) == set(ZONES) for mapping in
                (quantiles, actual, references, distributions)),
            "Exactly four countries required by the daily expert fit")
    for family in PRICE_EXPERTS:
        require(set(built.features[family]) == set(ZONES),
                f"{family}: exactly four country feature matrices required")
        for zone in ZONES:
            frame = built.features[family][zone]
            require(isinstance(frame, pd.DataFrame) and frame.index.equals(full)
                    and frame.columns.is_unique and len(frame.columns)
                    and not np.isinf(frame.to_numpy(dtype=float)).any(),
                    f"{family}/{zone}: exact 365-day history and two-day features required")
    for zone in ZONES:
        _aligned(actual[zone], full, f"{zone} observed history", finite=False)
        require(np.isfinite(actual[zone].loc[train].to_numpy(dtype=float)).all(),
                f"{zone}: complete finite 365-day real training labels required")
        require(actual[zone].loc[current].isna().all(),
                f"{zone}: future labels must remain missing")
        quantile = quantiles[zone]
        require(isinstance(quantile, pd.DataFrame) and quantile.index.equals(full)
                and list(quantile) == ["q10", "q50", "q90"]
                and np.isfinite(quantile.to_numpy(dtype=float)).all()
                and (np.diff(quantile.to_numpy(dtype=float), axis=1) >= 0).all(),
                f"{zone}: aligned finite ordered NYX quantiles required")
        _aligned(references[zone], current, f"{zone} two-day reference")
        distribution = distributions[zone]
        names = ["test2__q10", "test2__q50", "test2__q90", "spike_probability"]
        require(isinstance(distribution, pd.DataFrame)
                and distribution.index.equals(current)
                and set(names) <= set(distribution)
                and np.isfinite(distribution.loc[:, names].to_numpy(dtype=float)).all()
                and (np.diff(distribution.loc[:, names[:3]].to_numpy(dtype=float), axis=1) >= 0).all()
                and distribution.spike_probability.between(0., 1.).all(),
                f"{zone}: valid two-day Test2 distributions required")
    return first, expected_cutoff, train, current, full


def fit_daily_experts(first_day, cutoff, built, quantiles, actual, references,
                      distributions, output_dir, progress=None):
    """Fit the fixed three price/four event recipes and forecast J+1/J+2.

    ``built.features``, ``quantiles`` and ``actual`` must cover exactly 365
    earlier civil days followed by the two future civil days. Future labels
    must be NaN. The canonical event recipe retains its honest empirical/raw
    fallback if its historical event classes do not support fitting or Platt
    calibration; audit counts distinguish attempts and fitted estimators.

    ``progress(stage, **details)`` receives running/complete events for each
    price expert and negative classifier. Models are always fitted afresh;
    an existing model destination is rejected instead of reused.
    """
    from chronos2_hourly.nyx_pooled_cpu_price_model import PooledConfig, fit_pooled_block
    from chronos2_hourly.nyx_negative_probability_cpu import fit_predict_block
    from chronos2_hourly.nyx_annual_cpu_live import compose_price

    first, cutoff, train, current, full = _validate(
        first_day, cutoff, built, quantiles, actual, references, distributions)
    output = Path(output_dir).resolve()
    model_dir = output / "models"
    audit_dir = output / "model_audits"
    # Check every destination before the first expensive fit or filesystem write.
    model_names = list(PRICE_EXPERTS) + ["negative_" + zone for zone in ZONES]
    require(not any((model_dir / (name + ".cbm")).exists()
                    or (audit_dir / (name + ".json")).exists() for name in model_names)
            and not (output / "experts_daily_refit.json").exists(),
            "Daily expert outputs already exist; choose a new immutable run directory")
    stop = str(first + timedelta(days=2))
    configurations = {
        "fr_residual_1000": PooledConfig(target_mode="residual", iterations=1000),
        "cwe_residual_2000": PooledConfig(target_mode="residual", iterations=2000),
        "cwe_absolute_2000": PooledConfig(target_mode="absolute", iterations=2000),
    }
    nyx = {zone: quantiles[zone].q50 for zone in ZONES}
    points, price_audits, negative_audits, negative_points = {}, {}, {}, {}
    fit_started = pd.Timestamp.now(tz="UTC").isoformat()
    for family in PRICE_EXPERTS:
        if progress:
            progress("price_expert", expert=family, status="running")
        model_path = model_dir / (family + ".cbm")
        predictions, audit = fit_pooled_block(built.features[family], actual, nyx,
            origin_day=str(first), stop_day=stop, config=configurations[family],
            model_path=model_path, initial_training_day=str(first - timedelta(days=365)),
            thread_count=PRICE_THREADS)
        require(set(predictions) == set(ZONES) and audit.get("tree_count") == configurations[family].iterations
                and audit.get("training_days") == 365
                and audit.get("training_labels_before_origin") is True
                and audit.get("Storm_used_as_input") is False,
                f"{family}: daily pooled fit does not match the fixed recipe")
        model = audit.get("model", {})
        require(model.get("saved") is True and model_path.is_file()
                and model.get("sha256") == _sha(model_path),
                f"{family}: newly fitted daily CBM missing or checksum differs")
        for zone in ZONES:
            frame = predictions[zone]
            require(isinstance(frame, pd.DataFrame) and list(frame) == ["point"],
                    f"{family}/{zone}: invalid daily price prediction schema")
            _aligned(frame.point, current, f"{family}/{zone} daily point")
        audit = {**audit, "fitted_in_this_launch": True,
                 # The canonical one-day audit has a broader origin claim.
                 # J+2 deliberately uses the same information state as J+1.
                 "daily_features_keep_their_own_D_minus_1_08_origin": False,
                 "historical_features_use_daily_eligibility_limits": True,
                 "future_days_use_shared_first_day_cutoff": True,
                 "second_day_information_cutoff_is_D_minus_2_08": True,
                 "shared_cutoff_utc": cutoff.isoformat(), "future_labels_used": False}
        _json_write(audit_dir / (family + ".json"), audit)
        points[family], price_audits[family] = predictions, audit
        if progress:
            progress("price_expert", expert=family, status="complete")
    for zone in ZONES:
        if progress:
            progress("negative_classifier", zone=zone, status="running")
        compact = built.features["cwe_residual_2000"][zone]
        block = fit_predict_block(compact.loc[train], actual[zone].loc[train],
            compact.loc[current], zone=zone, origin_day=str(first), stop_day=stop,
            threads=NEGATIVE_THREADS)
        audit, probabilities = block.audit, block.probabilities
        count = audit.get("models_fitted")
        require(count in (0, 1) and audit.get("history_civil_days") == 365
                and audit.get("classifier_civil_days") == 337
                and audit.get("calibration_civil_days") == 28
                and audit.get("all_training_labels_before_origin") is True
                and audit.get("forecast_labels_used") is False
                and audit.get("storm_used") is False,
                f"{zone}: daily negative-price fit audit differs from the canonical recipe")
        if count:
            require(block.estimator is not None and audit.get("tree_count") == 120
                    and audit.get("fallback_reason") is None,
                    f"{zone}: fitted classifier identity is inconsistent")
            model = _save_classifier(block.estimator, model_dir / ("negative_" + zone + ".cbm"))
        else:
            require(block.estimator is None and audit.get("tree_count") == 0
                    and audit.get("fallback_reason") in ("one_class", "constant_features")
                    and audit.get("calibration_status") == "frequency_fallback",
                    f"{zone}: unreported daily classifier fitting fallback")
            model = {"saved": False, "path": None, "sha256": None,
                     "format": "cbm", "bytes": None,
                     "reason": audit["fallback_reason"]}
        required = ["p_negative_raw", "p_negative", "is_negative_predicted"]
        require(isinstance(probabilities, pd.DataFrame)
                and probabilities.index.equals(current) and set(required) <= set(probabilities)
                and probabilities.p_negative.between(0., 1.).all()
                and probabilities.p_negative_raw.between(0., 1.).all()
                and (probabilities.is_negative_predicted ==
                     (probabilities.p_negative >= .5)).all(),
                f"{zone}: invalid daily negative-price probabilities")
        audit = {**audit, "model": model, "refit_attempted_in_this_launch": True,
                 "fitted_in_this_launch": bool(count),
                 "calibration_recomputed_in_this_launch": True,
                 "shared_cutoff_utc": cutoff.isoformat(), "future_labels_used": False}
        _json_write(audit_dir / ("negative_" + zone + ".json"), audit)
        negative_audits[zone], negative_points[zone] = audit, probabilities
        if progress:
            progress("negative_classifier", zone=zone, status="complete")
    frames = {}
    for zone in ZONES:
        experts = {family: points[family][zone].point for family in PRICE_EXPERTS}
        price = compose_price(zone, experts, references[zone])
        negative = negative_points[zone]
        distribution = distributions[zone]
        frame = pd.DataFrame({
            "price_eur_mwh": price,
            "p_negative": negative.p_negative,
            "p_negative_raw": negative.p_negative_raw,
            "is_negative_predicted": negative.is_negative_predicted,
            "reference_eur_mwh": references[zone],
            **{family + "_eur_mwh": point for family, point in experts.items()},
            **{"nyx__" + q: quantiles[zone].loc[current, q] for q in ("q10", "q50", "q90")},
            **{name: distribution[name] for name in
               ("test2__q10", "test2__q50", "test2__q90", "spike_probability")},
            "forecast_origin_utc": cutoff,
            "horizon_day": [(day - first).days + 1 for day in current.tz_convert("Europe/Paris").date],
            "timestamp_local": [stamp.isoformat() for stamp in current.tz_convert("Europe/Paris")],
        }, index=current)
        frame.index.name = "timestamp_utc"
        frames[zone] = frame
    negative_count = sum(audit["models_fitted"] for audit in negative_audits.values())
    calibrations = sum(audit["calibration_status"] == "platt" for audit in negative_audits.values())
    audit = {"protocol": PROTOCOL, "first_delivery_day": str(first),
        "forecast_days": 2, "shared_cutoff_utc": cutoff.isoformat(),
        "model_fit_started_at_utc": fit_started,
        "model_fit_completed_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "final_model_refit_attempts": 7, "price_models_fitted": 3,
        "negative_models_fitted": negative_count, "final_models_fitted": 3 + negative_count,
        "negative_platt_calibrators_fitted": calibrations,
        "negative_fallbacks": {zone: item["fallback_reason"] for zone, item in negative_audits.items()
                               if item["fallback_reason"] is not None},
        "all_final_model_recipes_refit_attempted": True,
        "all_final_model_recipes_refitted": negative_count == 4,
        "negative_calibration": "recomputed from this launch's chronological 28-day calibration window",
        "foundation_model_refitted": False, "full_835_day_replay": False,
        "future_labels_used": False, "qualified_for_production": False,
        "price_experts": price_audits, "negative_countries": negative_audits,
        "feature_assembly": built.audits,
        "price_threads": PRICE_THREADS, "negative_threads": NEGATIVE_THREADS,
        "model_artifact_sha256": {str(path.relative_to(output)).replace("\\", "/"): _sha(path)
                                  for path in model_dir.glob("*.cbm")},
        "audit_artifact_sha256": {str(path.relative_to(output)).replace("\\", "/"): _sha(path)
                                  for path in audit_dir.glob("*.json")}}
    _json_write(output / "experts_daily_refit.json", audit)
    return frames, audit


__all__ = ["fit_daily_experts"]
