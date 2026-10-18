"""Fresh daily HGB/Test2 reference fits and strictly earlier prior90 policies.

The supplied historical features and observed prices are reconstructed at the
current outer cutoff. Retained OOF predictions are never overwritten by the
current fit. Missing recent OOF days are produced by earlier-label fits in
blocks of at most seven civil days, rather than an annual replay.
"""
from __future__ import annotations

from datetime import date, timedelta
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROTOCOL = "nyx_daily_refit_reference_v1"
ZONES = ("FR", "DE", "BE", "NL")

# Ordered input manifest of nyx_live_hybrid.build_pair_features, followed by
# the canonical CPU reference builder's drop-four/add-baseline preparation.
# The 123-column compact price-expert matrix is a different feature family.
_TEST2_COMPONENTS = (
    "wind_gw", "solar_gw", "residual_load_gw", "wind_ratio", "solar_ratio",
    "low_wind", "low_solar", "joint_deficit", "residual_stress", "deficit_stress",
    "rl_ramp_previous", "rl_ramp_next1", "rl_ramp_next2", "wind_drop_next2",
    "solar_drop_next2", "stress_next1", "stress_next2",
    "daily_peak_residual_stress", "daily_peak_deficit_stress",
)


def _test2_columns(pair):
    """The two country-specific canonical Test2 schemas, in producer order."""
    from chronos2_hourly.nyx_annual_test2_cpu import DAILY_PEAK_COLUMNS, PAIRS
    pair = tuple(pair)
    require(pair in PAIRS, "Explicit BE/FR or DE/NL Test2 pair required")
    names = [prefix + name for prefix in ("own_", "other_") for name in _TEST2_COMPONENTS]
    names.extend(("regional_max_residual_stress", "regional_mean_residual_stress",
                  "regional_joint_deficit", "country_is_fr" if pair == ("BE", "FR")
                  else "country_is_nl"))
    names.extend(f"{cycle}_{part}" for cycle in ("hour", "weekday", "annual")
                 for part in ("sin", "cos"))
    return tuple(name for name in sorted(names) if name not in DAILY_PEAK_COLUMNS) + ("baseline_p50",)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(first, stop):
    return pd.date_range(str(first), str(stop), tz="Europe/Paris", freq="h",
                         inclusive="left").tz_convert("UTC")


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists(), f"Daily reference artifact already exists: {path.name}")
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False,
                               default=str) + "\n", encoding="utf-8")


def _index(index):
    require(isinstance(index, pd.DatetimeIndex) and str(index.tz) == "UTC"
            and len(index) and not index.hasnans and index.is_unique
            and index.is_monotonic_increasing and index.equals(index.floor("h")),
            "Ordered unique physical hourly UTC reference grid required")


def _notify(progress, stage, **details):
    if progress is not None:
        progress(stage, **details)


def _prepare(first_day, cutoff, built, quantiles, actual, prior_oof):
    from chronos2_hourly.nyx_annual_prior90_cpu import FEATURES, _checked
    from chronos2_hourly.nyx_annual_test2_cpu import DAILY_PEAK_COLUMNS, PAIRS, _check_matrix
    from chronos2_hourly.nyx_historical_hgb_reference import archived_columns

    first = date.fromisoformat(str(first_day))
    require(str(first_day) == first.isoformat(), "Expected YYYY-MM-DD first delivery day")
    expected_cutoff = pd.Timestamp(f"{first - timedelta(days=1)} 08:00",
                                   tz="Europe/Paris").tz_convert("UTC")
    supplied_cutoff = pd.Timestamp(cutoff)
    require(not pd.isna(supplied_cutoff) and supplied_cutoff.tzinfo is not None
            and supplied_cutoff.tz_convert("UTC") == expected_cutoff,
            "Daily reference requires the shared first-day D-1 08:00 cutoff")
    current = _grid(first, first + timedelta(days=2))
    prior_index = _grid(first - timedelta(days=90), first)
    mappings = (built.base292, built.augmented334, built.test2_features,
                quantiles, actual, prior_oof)
    require(all(set(value) == set(ZONES) for value in mappings),
            "Exactly four countries required by the paired daily reference")
    prepared, old = {}, {}
    support = built.base292[ZONES[0]].index
    _index(support)
    require(current.isin(support).all(), "Complete J+1/J+2 reference support required")
    for zone in ZONES:
        for family, variant in ((built.base292, "hist_residual_400"),
                                (built.augmented334, "augmented_hist_residual_400")):
            matrix = family[zone]
            require(isinstance(matrix, pd.DataFrame) and matrix.index.equals(support)
                    and tuple(matrix.columns) == archived_columns(zone, variant)
                    and not np.isinf(matrix.to_numpy(dtype=float)).any(),
                    f"{zone}: historical HGB reference schema or support differs")
        nyx = quantiles[zone]
        require(isinstance(nyx, pd.DataFrame) and list(nyx) == ["q10", "q50", "q90"]
                and support.isin(nyx.index).all()
                and np.isfinite(nyx.loc[support].to_numpy(dtype=float)).all()
                and (np.diff(nyx.loc[support].to_numpy(dtype=float), axis=1) >= 0).all(),
                f"{zone}: finite ordered baseline quantiles required")
        observed = actual[zone]
        require(isinstance(observed, pd.Series) and support.isin(observed.index).all()
                and "storm" not in str(observed.name).lower(),
                f"{zone}: aligned observed EPEX price support required")
        _index(observed.index)
        require(observed.loc[observed.index >= current[0]].isna().all(),
                f"{zone}: future observed price labels must remain missing")
        require(np.isfinite(observed.loc[support[support < current[0]]].to_numpy(dtype=float)).all(),
                f"{zone}: finite historical EPEX labels required")
        matrix = built.test2_features[zone]
        require(isinstance(matrix, pd.DataFrame) and matrix.index.equals(support),
                f"{zone}: Test2 support differs from the HGB reference")
        matrix = matrix.drop(columns=list(DAILY_PEAK_COLUMNS), errors="ignore").copy()
        matrix["baseline_p50"] = nyx.loc[support, "q50"]
        _check_matrix(matrix)
        pair = next(pair for pair in PAIRS if zone in pair)
        require(tuple(matrix.columns) == _test2_columns(pair),
                f"{zone}: canonical Test2 feature names/order differ for {'/'.join(pair)}")
        prepared[zone] = matrix
        oof = prior_oof[zone]
        require(isinstance(oof, pd.DataFrame) and set(FEATURES) <= set(oof.columns),
                f"{zone}: seven canonical prior90 OOF signals required")
        _index(oof.index)
        require((oof.index < current[0]).all(),
                f"{zone}: prior90 OOF may contain only strictly earlier deliveries")
        old[zone] = oof.loc[:, list(FEATURES)].copy()
        _checked(old[zone])
        require(prior_index[0] >= old[zone].index[0],
                f"{zone}: retained OOF does not reach the prior90 start")
        first_old = old[zone].index[0].tz_convert("Europe/Paris").date()
        last_old = old[zone].index[-1].tz_convert("Europe/Paris").date()
        require(old[zone].index.equals(_grid(first_old, last_old + timedelta(days=1))),
                f"{zone}: retained OOF must contain complete contiguous civil days")
    require(all(old[zone].index.equals(old[ZONES[0]].index) for zone in ZONES),
            "Four-country retained OOF must use the same civil-day history")
    for pair in PAIRS:
        require(tuple(prepared[pair[0]].columns) == tuple(prepared[pair[1]].columns),
                "Test2 pair training columns differ")
    bridge_first = old[ZONES[0]].index[-1].tz_convert("Europe/Paris").date() + timedelta(days=1)
    earliest = min(first, bridge_first) - timedelta(days=365)
    require(_grid(earliest, first + timedelta(days=2)).isin(support).all(),
            "Complete 365-day training support required before every current or OOF fit")
    return first, expected_cutoff, current, prior_index, prepared, old, bridge_first


def _fit_block(origin, stop, *, built, quantiles, actual, prepared, directory,
               purpose, progress, source_cutoff=None):
    """Fit on 365 earlier days; prediction labels are always masked."""
    import joblib
    from chronos2_hourly.nyx_historical_hgb_reference import VARIANTS, fit_hgb_block, frame_hash
    from chronos2_hourly.nyx_annual_test2_cpu import PAIRS, fit_test2_origin
    from chronos2_hourly.nyx_annual_equal_ensemble import combine_equal_ensemble
    from chronos2_hourly.nyx_annual_prior90_cpu import FEATURES

    train = _grid(origin - timedelta(days=365), origin)
    current = _grid(origin, stop)
    available = train.append(current)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    hgb, models, records = {}, {}, []
    for zone in ZONES:
        labels = actual[zone].reindex(available).copy()
        labels.loc[current] = np.nan
        nyx = quantiles[zone].loc[available, "q50"]
        hgb[zone] = {}
        for variant in VARIANTS:
            _notify(progress, "reference_hgb_fit", purpose=purpose, status="running",
                    zone=zone, variant=variant, origin_day=str(origin))
            matrix = (built.augmented334 if VARIANTS[variant][0] == "augmented"
                      else built.base292)[zone].loc[available]
            fitted, prediction, audit = fit_hgb_block(
                matrix, labels, nyx, zone=zone, variant=variant,
                origin_day=str(origin), stop_day=str(stop),
                initial_training_day=str(origin - timedelta(days=365)))
            require(prediction.index.equals(current) and "point" in prediction
                    and np.isfinite(prediction.point.to_numpy(dtype=float)).all(),
                    "Invalid newly fitted HGB prediction")
            require(audit.get("training_days") == 365
                    and audit.get("training_last_day") == str(origin - timedelta(days=1))
                    and audit.get("training_labels_before_origin") is True,
                    "HGB daily fit must use exactly 365 strictly earlier days")
            hgb[zone][variant] = prediction.point
            path = directory / f"{zone}_{variant}.joblib"
            joblib.dump((fitted, audit), path)
            records.append({"kind": "hgb", "zone": zone, "variant": variant,
                            "audit": audit, "model": path.name})
            _notify(progress, "reference_hgb_fit", purpose=purpose, status="complete",
                    zone=zone, variant=variant, origin_day=str(origin))
    for pair in PAIRS:
        _notify(progress, "reference_test2_fit", purpose=purpose, status="running",
                pair=list(pair), origin_day=str(origin))
        x = pd.concat([prepared[zone].loc[train] for zone in pair])
        residual = np.concatenate([
            actual[zone].loc[train].to_numpy(dtype=float)
            - quantiles[zone].loc[train, "q50"].to_numpy(dtype=float) for zone in pair])
        fitted = fit_test2_origin(x, residual, list(train.tz_convert("Europe/Paris").date) * 2,
                                  pair=pair, origin_day=str(origin), iterations=120, threads=1)
        require(fitted.audit.get("train_unique_days") == 365
                and fitted.audit.get("train_last_day") == str(origin - timedelta(days=1))
                and fitted.audit.get("future_labels_used") is False,
                "Test2 daily fit must use exactly 365 strictly earlier days")
        models[pair] = fitted
        path = directory / ("Test2_" + "_".join(pair) + ".joblib")
        joblib.dump((fitted, fitted.audit), path)
        records.append({"kind": "test2", "pair": list(pair), "audit": fitted.audit,
                        "training_features_sha256": frame_hash(x),
                        "training_residual_sha256": frame_hash(pd.Series(residual, name="residual")),
                        "model": path.name})
        _notify(progress, "reference_test2_fit", purpose=purpose, status="complete",
                pair=list(pair), origin_day=str(origin))
    features, distributions = {}, {}
    for zone in ZONES:
        pair = next(pair for pair in PAIRS if zone in pair)
        daily = []
        for delivery in pd.date_range(str(origin), str(stop), freq="D", inclusive="left").date:
            index = _grid(delivery, delivery + timedelta(days=1))
            # Logical inner replay cutoffs do not claim historical provider vintages.
            issued = source_cutoff if source_cutoff is not None else pd.Timestamp(
                f"{delivery - timedelta(days=1)} 08:00", tz="Europe/Paris").tz_convert("UTC")
            daily.append(models[pair].predict_day(prepared[zone].loc[index],
                         quantiles[zone].loc[index, "q50"], zone=zone,
                         forecast_issued_at_utc=issued))
        distribution = pd.concat(daily)
        require(distribution.index.equals(current), "Test2 prediction civil grid differs")
        nyx = quantiles[zone].loc[current, "q50"]
        ensemble = combine_equal_ensemble(nyx=nyx, test2=distribution.test2__q50,
            price_residual=hgb[zone]["hist_residual_400"],
            price_absolute=hgb[zone]["hist_absolute_400"],
            augmented_residual=hgb[zone]["augmented_hist_residual_400"])
        local_days = current.tz_convert("Europe/Paris").date
        frame = pd.DataFrame({"ensemble__q50": ensemble, "nyx__q50": nyx,
            "test2__q50": distribution.test2__q50,
            "own_joint_deficit": prepared[zone].loc[current, "own_joint_deficit"],
            "own_residual_stress": prepared[zone].loc[current, "own_residual_stress"],
            "nyx_daily_peak_gap": nyx.groupby(local_days).transform("max") - nyx,
            "spike_probability": distribution.spike_probability}, index=current)
        features[zone], distributions[zone] = frame.loc[:, list(FEATURES)], distribution
    _write_json(directory / "fit_audit.json", {
        "purpose": purpose, "origin_day": str(origin), "stop_day_exclusive": str(stop),
        "training_first_day": str(origin - timedelta(days=365)), "training_days": 365,
        "training_last_day": str(origin - timedelta(days=1)),
        "shared_forecast_cutoff_utc": source_cutoff.isoformat() if source_cutoff is not None else None,
        "future_labels_used": False, "prediction_labels_masked": True,
        "historical_features_reconstructed_at_outer_cutoff": True,
        "historical_source_vintages_certified": False, "models": records})
    return features, distributions


def fit_daily_reference(first_day, cutoff, built, quantiles, actual, prior_oof,
                        output_dir, *, progress=None):
    """Return references, Test2 distributions, audit, updated strictly past OOF.

    All current estimators are fitted on every call. A nonempty destination is
    rejected; no fitted-estimator cache or previous seed estimators are loaded.
    Only retained past OOF predictions are reusable. Missing OOF tail days are
    reconstructed with models whose labels strictly precede their prediction
    blocks. Policies are newly selected on exactly the preceding 90 civil days.
    """
    from chronos2_hourly.nyx_annual_prior90_cpu import select_prior90_policy, apply_prior90_daily
    from chronos2_hourly.nyx_annual_reference_pair import PARTNER, SIGNALS, build_confirmed_pair

    first, cutoff, current, prior_index, prepared, updated, bridge_first = _prepare(
        first_day, cutoff, built, quantiles, actual, prior_oof)
    root = Path(output_dir) / "reference"
    require(not root.exists(), "Daily reference destination already exists; estimators are never reused")
    root.mkdir(parents=True)
    original_hashes = {zone: hashlib.sha256(pd.util.hash_pandas_object(
        updated[zone], index=True).to_numpy().tobytes()).hexdigest() for zone in ZONES}
    bridges = []
    origin = bridge_first
    while origin < first:
        stop = min(first, origin + timedelta(days=7))
        oof, _ = _fit_block(origin, stop, built=built, quantiles=quantiles,
                            actual=actual, prepared=prepared,
                            directory=root / "bridge_models" / str(origin),
                            purpose="bridge", progress=progress)
        for zone in ZONES:
            require(not updated[zone].index.intersection(oof[zone].index).size,
                    "Current fit cannot replace retained OOF predictions")
            updated[zone] = pd.concat([updated[zone], oof[zone]])
        bridges.append({"origin_day": str(origin), "stop_day_exclusive": str(stop),
                        "days": (stop - origin).days, "hgb_fits": 12, "test2_pair_fits": 2})
        origin = stop
    require(all(prior_index.isin(updated[zone].index).all() for zone in ZONES),
            "Prior90 OOF history remains incomplete")
    future, distributions = _fit_block(
        first, first + timedelta(days=2), built=built, quantiles=quantiles, actual=actual,
        prepared=prepared, directory=root / "models" / str(first),
        purpose="current", progress=progress, source_cutoff=cutoff)
    references, policy_audits = {}, {}
    for zone in ZONES:
        _notify(progress, "reference_prior90", status="running", zone=zone,
                origin_day=str(first))
        policy = select_prior90_policy(updated[zone].loc[prior_index],
                                      actual[zone].loc[prior_index],
                                      origin_day=str(first), zone=zone)
        selected = []
        for delivery in (first, first + timedelta(days=1)):
            index = _grid(delivery, delivery + timedelta(days=1))
            selected.append(apply_prior90_daily(future[zone].loc[index], policy,
                                                forecast_issued_at_utc=cutoff))
        active = pd.concat(selected).prior90_active
        signals = future[zone][["ensemble__q50", "nyx__q50", "test2__q50", "spike_probability"]].copy()
        signals["prior90_active"] = active
        signals["partner_nyx__q50"] = quantiles[PARTNER[zone]].loc[current, "q50"]
        signals = signals.loc[:, list(SIGNALS)]
        result = build_confirmed_pair(signals, zone=zone, partner_zone=PARTNER[zone])
        references[zone] = result.scarcity_confirmed_pair.rename("reference")
        _write_json(root / "policies" / (zone + ".json"), policy)
        for folder, frame in (("oof", updated[zone]), ("future_inputs", future[zone]),
                              ("signals", signals), ("distributions", distributions[zone]),
                              ("forecast", references[zone].to_frame())):
            directory = root / folder
            directory.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(directory / (zone + ".parquet"))
        policy_audits[zone] = {key: policy[key] for key in (
            "history_start_day", "history_end_exclusive", "history_hours",
            "selected_rule_index", "selected_rule")}
        _notify(progress, "reference_prior90", status="complete", zone=zone,
                origin_day=str(first), selected_rule_index=policy["selected_rule_index"])
    hashes = {path.relative_to(root).as_posix(): _digest(path)
              for path in sorted(root.rglob("*")) if path.is_file()}
    audit = {"protocol": PROTOCOL, "first_delivery_day": str(first),
        "last_delivery_day": str(first + timedelta(days=1)),
        "shared_cutoff_utc": cutoff.isoformat(), "training_days": 365,
        "training_first_day": str(first - timedelta(days=365)),
        "training_last_day": str(first - timedelta(days=1)),
        "current_fit_counts": {"hgb": 12, "test2_pairs": 2},
        "test2_feature_counts": {zone: len(prepared[zone].columns) for zone in ZONES},
        "test2_feature_names": {zone: list(prepared[zone].columns) for zone in ZONES},
        "bridge_fit_counts": {"hgb": 12 * len(bridges), "test2_pairs": 2 * len(bridges)},
        "reference_models_fitted": 14, "total_model_fits": 14 * (1 + len(bridges)),
        "bridge_blocks": bridges, "prior90_policies_reselected": 4,
        "policies": policy_audits, "retained_oof_sha256": original_hashes,
        "retained_oof_overwritten": False, "current_fit_used_as_past_oof": False,
        "estimator_cache_used": False, "saved_seed_estimators_loaded": False,
        "future_labels_used": False, "storm_used_as_input": False,
        "historical_features_reconstructed_at_outer_cutoff": True,
        "historical_provider_vintages_certified": False,
        "annual_performance_qualification_claimed": False,
        "artifact_sha256": hashes}
    _write_json(root / "receipt.json", audit)
    return references, distributions, audit, updated


__all__ = ["fit_daily_reference", "PROTOCOL"]
