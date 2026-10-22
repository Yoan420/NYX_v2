"""Two civil-day forecasts from a completed annual CPU model bundle.

Only inference happens here. Sources and bounded baseline reconstruction are
separate producers. Unknown first-day prices remain missing in second-day lags;
they are never recast as observed labels. The saved weekly reference may be
older than its original week, so the receipt declares frozen model reuse rather
than claiming the original weekly training schedule or annual qualification.
"""
from __future__ import annotations

from datetime import timedelta
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ZONES = ("FR", "DE", "BE", "NL")
PRICE_EXPERTS = ("fr_residual_1000", "cwe_residual_2000", "cwe_absolute_2000")
PROTOCOL = "nyx_saved_two_day_inference_v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def grid(first_day, days=2):
    first = pd.Timestamp(first_day).date()
    require(days in (1, 2), "One or two complete civil delivery days required")
    return pd.date_range(str(first), str(first + timedelta(days=days)),
                         tz="Europe/Paris", freq="h", inclusive="left").tz_convert("UTC")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _bound_path(root, relative, hashes):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    require(path.is_relative_to(root) and path.is_file(), f"Missing or redirected saved artifact: {relative}")
    require(relative in hashes and digest(path) == hashes[relative], f"Saved artifact checksum differs: {relative}")
    return path


def _finite_series(value, index, label):
    require(isinstance(value, pd.Series) and value.index.equals(index)
            and np.isfinite(value.to_numpy(dtype=float)).all(), f"Aligned finite {label} required")


def assemble_horizon_features(first_day, cutoff, *, sources, quantiles, origins):
    """Assemble the established feature schema on fresh J+1/J+2 inputs.

    Sources must provide genuine future forecast covariates at one cutoff.
    Optional public descriptors can carry schema-supported NaN/zero flags.
    Historical covariate origins are logical replay dates and do not certify
    original provider revisions. Future origins must equal the shared cutoff.
    """
    from chronos2_hourly import nyx_annual_live_preflight as gate
    from chronos2_hourly import nyx_live_hybrid as test2
    from chronos2_hourly.nyx_annual_cpu_bundle_builder import BuiltFeatures, known_columns, OWN_TEST2, JAO_COLUMNS, _frame
    from chronos2_hourly.nyx_annual_feature_projection import POOLED, FULL, COMPACT, project_country
    from chronos2_hourly.nyx_local_price_features import build_price_features
    from chronos2_hourly.nyx_local_extra_features import build_additional_features, cutoff_times
    from chronos2_hourly.nyx_pooled_calendar import build_pooled_calendar_features
    from chronos2_hourly.nyx_forecast_profile_features import build_forecast_profile_features
    from chronos2_hourly import nyx_lagged_exchange_features as exchange
    from chronos2_hourly import nyx_thermal_capacity_features as thermal

    current, cutoff = grid(first_day), pd.Timestamp(cutoff)
    require(cutoff.tzinfo is not None, "An explicit UTC forecast cutoff is required")
    cutoff = cutoff.tz_convert("UTC")
    require(cutoff <= cutoff_times(current, "Europe/Paris").min(), "Shared source cutoff is after J+1 origin")
    require(set(quantiles) == set(origins) == set(ZONES), "Four country quantiles and origin histories required")
    prices, covariates = sources["prices"], sources["covariates"]
    require(set(prices) == set(ZONES), "Four-country EPEX price history required")
    for zone in ZONES:
        require(not (prices[zone].index >= current[0]).any(), f"{zone}: future actual price supplied")
        require(set(quantiles[zone]) == {"q10", "q50", "q90"} and current.isin(quantiles[zone].index).all(),
                f"{zone}: complete historical and two-day baseline quantiles required")
        selected = pd.to_datetime(origins[zone].reindex(current), utc=True)
        require(selected.notna().all() and (selected == cutoff).all(), f"{zone}: future origin differs from shared cutoff")
    known = known_columns()
    require(set(known) <= set(covariates) and covariates.index.is_unique
            and covariates.index.is_monotonic_increasing and str(covariates.index.tz) == "UTC",
            "Ordered retained history and fresh covariates required")
    source = covariates.loc[:, known]
    require(current.isin(source.index).all() and np.isfinite(source.to_numpy(dtype=float)).all(),
            "Finite Test2 normalization history and genuine future covariates required")
    pairs, pair_audits = {}, {}
    for pair in test2.PAIRS:
        values, audit = test2.build_pair_features({zone: source for zone in pair}, pair)
        pairs.update(values)
        pair_audits["_".join(pair)] = audit
    previous = {zone: quantiles[zone].q50.rename("nyx_q50") for zone in ZONES}
    base, augmented, country_audits = {}, {}, {}
    schema = gate.load_schema()
    for zone in ZONES:
        base[zone], price_audit = build_price_features(prices, zone=zone, delivery_index=current,
            nyx_predictions=quantiles[zone], past_predictions=previous,
            known_covariates=source, known_columns=known, known_delta_columns=known,
            forecast_origins=origins[zone])
        additional, extra_audit = build_additional_features(zone=zone, delivery_index=current,
            nyx_forecasts=quantiles, forecast_origins=origins, fuel=sources["fuel"])
        augmented[zone] = pd.concat([base[zone], additional], axis=1)
        require(list(base[zone]) == schema["families"][POOLED]["columns"][zone][:292]
                and list(augmented[zone]) == schema["families"][POOLED]["columns"][zone][:334],
                f"{zone}: saved HGB input schema differs")
        country_audits[zone] = {"price": price_audit, "additional": extra_audit}
        for item in country_audits[zone].values():
            item["actual_forecast_information_cutoff_utc"] = cutoff.isoformat()
            item["daily_nominal_cutoffs_are_eligibility_limits"] = True
    hydro_columns = [name for name in schema["families"][POOLED]["columns"]["FR"] if name.startswith("extra_hydro_")]
    hydro = _frame(sources["hydro_features"], current, hydro_columns, "future hydro")
    original = _frame(sources["jao_original"], current, JAO_COLUMNS, "future original JAO")
    refreshed = _frame(sources["jao_refreshed"], current, JAO_COLUMNS, "future refreshed JAO")
    exchanges = _frame(sources["exchange_features"], current, exchange.COLUMNS, "future exchange")
    features = {family: {} for family in gate.FAMILIES}
    for zone in ZONES:
        common = augmented[zone]
        calendar, calendar_audit = build_pooled_calendar_features(zone=zone, delivery_index=current)
        profiles, profiles_audit = build_forecast_profile_features(common)
        profiles_audit["source"] = "Both future civil-day profiles queried at the same recorded information cutoff"
        profiles_audit["actual_forecast_information_cutoff_utc"] = cutoff.isoformat()
        capacities = _frame(sources["thermal_features"][zone], current, thermal.COLUMNS, f"future thermal {zone}")
        before, after = [common, calendar, hydro], [profiles]
        pooled = pd.concat([*before, original, *after], axis=1)
        own = pairs[zone].loc[current, list(OWN_TEST2)].rename(columns=lambda name: "canon__" + name)
        full = pd.concat([*before, refreshed, *after, own, capacities, exchanges], axis=1)
        compact = project_country(full, pooled, zone=zone, expected_index=current, schema=schema)
        for family, frame in ((POOLED, pooled), (FULL, full), (COMPACT, compact)):
            gate.validate_feature_frame(frame, schema["families"][family]["columns"][zone], current, family + "/" + zone)
            features[family][zone] = frame
        country_audits[zone].update(calendar=calendar_audit, profiles=profiles_audit)
    return BuiltFeatures(features, base, augmented,
        {zone: pairs[zone].loc[current].copy() for zone in ZONES},
        {"protocol": PROTOCOL, "first_delivery_day": str(first_day), "forecast_days": 2,
         "shared_cutoff_utc": cutoff.isoformat(), "future_labels_used": False,
         "unknown_price_lags": "NaN with explicit unavailable flags; no predicted price recast as observation",
         "source_audit": sources["audit"], "countries": country_audits, "test2": pair_audits})


def _frozen_test2(fitted, matrix, nyx):
    """Use stored parameters beyond weekly expiry without rewriting origin metadata."""
    from chronos2_hourly.nyx_annual_test2_cpu import _check_matrix
    from chronos2_hourly import solar_wind_scarcity_regime as regime
    _check_matrix(matrix, expected=fitted.feature_names)
    _finite_series(nyx, matrix.index, "NYX baseline")
    require(np.array_equal(matrix.baseline_p50.to_numpy(float), nyx.to_numpy(float)), "Test2 NYX base differs")
    data = matrix.to_numpy(dtype=float)
    if fitted.gate is None:
        probability = np.full(len(data), fitted.empirical_probability, dtype=float)
    elif fitted.calibrator is None:
        probability = fitted.gate.predict_proba(data)[:, 1]
    else:
        logits = np.asarray(fitted.gate.predict(data, prediction_type="RawFormulaVal"), dtype=float).reshape(-1, 1)
        probability = fitted.calibrator.predict_proba(logits)[:, 1]
    require(np.isfinite(probability).all() and ((probability >= 0) & (probability <= 1)).all(), "Invalid saved Test2 probabilities")
    residual = regime.mixture_quantiles(fitted.normal.predict(data), fitted.spike.predict(data), probability)
    result = pd.DataFrame({"test2__" + q: nyx.to_numpy(float) + residual[:, i]
                          for i, q in enumerate(("q10", "q50", "q90"))}, index=matrix.index)
    result["spike_probability"] = probability
    require(np.isfinite(result.to_numpy(float)).all()
            and (np.diff(result[["test2__q10", "test2__q50", "test2__q90"]].to_numpy(float), axis=1) >= 0).all(),
            "Invalid frozen Test2 quantiles")
    return result


def frozen_reference(seed_bundle, first_day, cutoff, built, quantiles):
    """Reload receipted weekly HGB/Test2 and the already selected prior90 policy."""
    import joblib
    from threadpoolctl import threadpool_limits
    from chronos2_hourly.nyx_historical_hgb_reference import VARIANTS
    from chronos2_hourly.nyx_annual_test2_cpu import PAIRS, DAILY_PEAK_COLUMNS
    from chronos2_hourly.nyx_annual_prior90_cpu import FEATURES, RULES, _mask
    from chronos2_hourly.nyx_annual_equal_ensemble import combine_equal_ensemble
    from chronos2_hourly.nyx_annual_reference_pair import PARTNER, SIGNALS, build_confirmed_pair
    seed_bundle = Path(seed_bundle).resolve()
    receipt = json.loads((seed_bundle / "source_receipts/scarcity_confirmed_pair.json").read_text(encoding="utf-8"))
    require(receipt.get("source_group") == "scarcity_confirmed_pair", "Saved reference receipt required")
    hashes, current = receipt["artifact_sha256"], grid(first_day)
    paths = [name for name in hashes if name.startswith("reference_models/") and name.endswith(".joblib")]
    require(paths, "Saved weekly reference model artifacts missing")
    origin = max(Path(name).parts[1] for name in paths)
    require(pd.Timestamp(origin).date() <= pd.Timestamp(first_day).date(), "Saved reference was fitted after forecast delivery")
    hgb = {}
    for zone in ZONES:
        hgb[zone] = {}
        for variant in VARIANTS:
            name = f"reference_models/{origin}/{zone}_{variant}.joblib"
            fitted, audit = joblib.load(_bound_path(seed_bundle, name, hashes))
            require(fitted.zone == zone and fitted.variant == variant and fitted.origin_day == origin,
                    "Frozen HGB identity differs")
            hgb[zone][variant] = fitted
    test2 = {}
    for pair in PAIRS:
        name = f"reference_models/{origin}/Test2_{'_'.join(pair)}.joblib"
        fitted, audit = joblib.load(_bound_path(seed_bundle, name, hashes))
        require(tuple(fitted.pair) == tuple(pair) and fitted.origin_day == origin, "Frozen Test2 identity differs")
        test2[pair] = fitted
    prepared = {}
    for zone in ZONES:
        prepared[zone] = built.test2_features[zone].drop(columns=list(DAILY_PEAK_COLUMNS), errors="ignore").copy()
        prepared[zone]["baseline_p50"] = quantiles[zone].loc[current, "q50"]
    references, signals, distributions = {}, {}, {}
    for zone in ZONES:
        nyx = quantiles[zone].loc[current, "q50"]
        pair = next(pair for pair in PAIRS if zone in pair)
        distribution = _frozen_test2(test2[pair], prepared[zone], nyx)
        points = {}
        for variant in VARIANTS:
            matrix = built.augmented334[zone] if VARIANTS[variant][0] == "augmented" else built.base292[zone]
            fitted = hgb[zone][variant]
            require(tuple(matrix.columns) == fitted.feature_columns and matrix.index.equals(current)
                    and not np.isinf(matrix.to_numpy(float)).any(), "Frozen HGB input schema/grid differs")
            with threadpool_limits(limits=1):
                raw = np.asarray(fitted.model.predict(matrix.astype(float)), dtype=float)
            point = nyx.to_numpy(float) + np.clip(raw, -60., 60.) if VARIANTS[variant][1] == "residual" else raw
            points[variant] = pd.Series(point, index=current)
        ensemble = combine_equal_ensemble(nyx=nyx, test2=distribution.test2__q50,
            price_residual=points["hist_residual_400"], price_absolute=points["hist_absolute_400"],
            augmented_residual=points["augmented_hist_residual_400"])
        local_days = current.tz_convert("Europe/Paris").date
        peak_gap = nyx.groupby(local_days).transform("max") - nyx
        policy_features = pd.DataFrame({"ensemble__q50": ensemble, "nyx__q50": nyx,
            "test2__q50": distribution.test2__q50,
            "own_joint_deficit": prepared[zone].own_joint_deficit,
            "own_residual_stress": prepared[zone].own_residual_stress,
            "nyx_daily_peak_gap": peak_gap, "spike_probability": distribution.spike_probability}, index=current).loc[:, list(FEATURES)]
        policy_path = _bound_path(seed_bundle, f"reference_policies/{zone}.json", hashes)
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
        require(policy.get("zone") == zone and policy.get("origin_day") == origin
                and policy.get("protocol") == "nyx_test2_spike_gate_prior90_v1", "Frozen policy identity differs")
        number = policy.get("selected_rule_index")
        if number is None:
            require(policy.get("selected_rule") is None, "Frozen fallback policy contains a rule")
            active = np.zeros(len(current), dtype=bool)
        else:
            require(type(number) is int and 0 <= number < len(RULES)
                    and policy.get("selected_rule") == RULES[number], "Frozen prior90 rule differs")
            active = _mask(policy_features, RULES[number])
        zone_signals = policy_features[["ensemble__q50", "nyx__q50", "test2__q50", "spike_probability"]].copy()
        zone_signals["prior90_active"] = active
        zone_signals["partner_nyx__q50"] = quantiles[PARTNER[zone]].loc[current, "q50"]
        zone_signals = zone_signals.loc[:, list(SIGNALS)]
        result = build_confirmed_pair(zone_signals, zone=zone, partner_zone=PARTNER[zone])
        references[zone] = result.scarcity_confirmed_pair.rename("reference")
        signals[zone], distributions[zone] = zone_signals, distribution
    return references, distributions, {"reference_models_fitted": 0, "frozen_reference_origin_day": origin,
        "original_weekly_fit_schedule_preserved": False,
        "frozen_reference_outside_original_week": any(pd.Timestamp(day).date() >= pd.Timestamp(origin).date() + timedelta(days=7)
            for day in pd.Index(current.tz_convert("Europe/Paris").date).unique()),
        "frozen_prior90_policy_reselected": False, "future_labels_used": False}


def _pooled_future_matrix(features, current, ordered_columns):
    require(set(features) == set(ZONES), "Four-country future features required")
    union = sorted({name for frame in features.values() for name in frame.columns})
    expected = union + ["country_" + zone for zone in ZONES]
    require(expected == list(ordered_columns), "Pooled model ordered feature schema differs")
    countries = {}
    for zone in ZONES:
        frame = features[zone]
        require(frame.index.equals(current) and frame.columns.is_unique, "Saved pooled prediction grid differs")
        block = frame.reindex(columns=union).astype(float)
        for name in set(union) - set(frame.columns):
            if name.endswith("__available"):
                block[name] = 0.
        require(not np.isinf(block.to_numpy()).any(), "Infinite pooled feature")
        flags = [name for name in union if name.endswith("__available")]
        require(block[flags].isin((0., 1.)).all().all(), "Invalid pooled availability flags")
        for country in ZONES:
            block["country_" + country] = float(country == zone)
        countries[zone] = block
    index = pd.MultiIndex.from_product([current, ZONES], names=["timestamp_utc", "zone"])
    return pd.concat(countries, names=["zone", "timestamp_utc"]).swaplevel().reindex(index)


def infer_saved_horizon(seed_bundle, seed_output, first_day, cutoff, built, quantiles):
    """Infer seven final saved models, preserving saved calibration coefficients."""
    from catboost import CatBoostClassifier, CatBoostRegressor
    from scipy.special import expit
    from chronos2_hourly.nyx_annual_cpu_live import compose_price
    seed_output = Path(seed_output).resolve()
    receipt = json.loads((seed_output / "receipt.json").read_text(encoding="utf-8"))
    require(receipt.get("status") == "REPORTS_COMPLETE" and receipt.get("future_labels_used") is False,
            "A completed label-free annual report run is required")
    require(pd.Timestamp(receipt["delivery_day"]).date() <= pd.Timestamp(first_day).date(),
            "Saved final models are from a later forecast delivery")
    audits, current = receipt["model_audit"], grid(first_day)
    references, test2_distributions, reference_audit = frozen_reference(seed_bundle, first_day, cutoff, built, quantiles)
    expert_points, model_hashes, expected_hashes = {}, {}, {}
    for family in PRICE_EXPERTS:
        audit = audits["price_experts"][family]
        relative = f"models/{family}.cbm"
        hashes = {relative: audit["model"]["sha256"]}
        expected_hashes.update(hashes)
        path = _bound_path(seed_output, relative, hashes)
        model = CatBoostRegressor()
        model.load_model(str(path), format="cbm")
        matrix = _pooled_future_matrix(built.features[family], current, audit["feature_columns"])
        require(model.feature_names_ == list(matrix.columns) and int(model.tree_count_) == audit["tree_count"],
                "Saved price estimator schema/tree count differs")
        raw = np.asarray(model.predict(matrix, task_type="CPU", thread_count=2), dtype=float)
        require(raw.shape == (len(matrix),) and np.isfinite(raw).all(), "Invalid saved price predictions")
        pooled = pd.Series(raw, index=matrix.index)
        points = {zone: pooled.xs(zone, level="zone").copy() for zone in ZONES}
        if audit["config"]["target_mode"] == "residual":
            points = {zone: points[zone] + quantiles[zone].loc[current, "q50"] for zone in ZONES}
        else:
            require(audit["config"]["target_mode"] == "absolute", "Unknown saved price target mode")
        expert_points[family], model_hashes[relative] = points, digest(path)
    frames = {}
    for zone in ZONES:
        audit = audits["negative_countries"][zone]
        relative = f"models/negative_{zone}.cbm"
        path = _bound_path(seed_output, relative, {relative: audit["model"]["sha256"]})
        expected_hashes[relative] = audit["model"]["sha256"]
        model = CatBoostClassifier()
        model.load_model(str(path), format="cbm")
        matrix = built.features["cwe_residual_2000"][zone]
        require(list(matrix.columns) == audit["feature_columns"] == model.feature_names_
                and int(model.tree_count_) == audit["tree_count"] == 120,
                "Saved negative-price estimator schema/tree count differs")
        raw = np.asarray(model.predict_proba(matrix, thread_count=2), dtype=float)[:, 1]
        if audit["calibration_status"] == "platt":
            slope, intercept = audit["calibration_slope"], audit["calibration_intercept"]
            require(np.isfinite([slope, intercept]).all(), "Saved Platt coefficients missing")
            margins = np.asarray(model.predict(matrix, prediction_type="RawFormulaVal", thread_count=2), dtype=float)
            probability = expit(slope * margins + intercept)
        else:
            require(audit["calibration_status"] == "raw_insufficient_calibration_classes", "Unsupported saved negative calibration")
            probability = raw.copy()
        require(np.isfinite(raw).all() and np.isfinite(probability).all()
                and ((probability >= 0) & (probability <= 1)).all(), "Invalid negative-price probability")
        points = {family: expert_points[family][zone] for family in PRICE_EXPERTS}
        price = compose_price(zone, points, references[zone])
        frame = pd.DataFrame({"price_eur_mwh": price, "p_negative": probability,
            "p_negative_raw": raw, "is_negative_predicted": probability >= .5,
            "reference_eur_mwh": references[zone],
            **{family + "_eur_mwh": point for family, point in points.items()},
            **{"nyx__" + q: quantiles[zone].loc[current, q] for q in ("q10", "q50", "q90")},
            "test2__q10": test2_distributions[zone].test2__q10,
            "test2__q50": test2_distributions[zone].test2__q50,
            "test2__q90": test2_distributions[zone].test2__q90,
            "spike_probability": test2_distributions[zone].spike_probability,
            "forecast_origin_utc": pd.Timestamp(cutoff).tz_convert("UTC"),
            "horizon_day": [(day - pd.Timestamp(first_day).date()).days + 1 for day in current.tz_convert("Europe/Paris").date],
            "timestamp_local": [stamp.isoformat() for stamp in current.tz_convert("Europe/Paris")]}, index=current)
        frame.index.name = "timestamp_utc"
        frames[zone], model_hashes[relative] = frame, digest(path)
    require(model_hashes == expected_hashes, "Saved final models changed during prediction")
    return frames, {"protocol": PROTOCOL, "first_delivery_day": first_day, "forecast_days": 2,
        "shared_cutoff_utc": pd.Timestamp(cutoff).tz_convert("UTC").isoformat(),
        "final_models_fitted": 0, "full_835_day_replay": False, "future_labels_used": False,
        "qualified_for_production": False, "seed_report_receipt_sha256": digest(seed_output / "receipt.json"),
        "saved_models_sha256": model_hashes, "reference": reference_audit,
        "feature_assembly": built.audits, "negative_calibration": "original saved Platt coefficients or original raw mode"}


def _produce_horizon(repo, seed_bundle, seed_output, first_day, forecast_issued_at_utc,
                     output_directory, context=None, *, carry_state=None,
                     cache_root=None, cache_contract=None):
    """Collect new sources, rebuild only the last-origin baseline, infer J+1/J+2.

    The immutable seed is read only. The new output directory is supplied by the
    launcher and stores all new source and bounded fitting evidence separately.
    ``forecast_issued_at_utc`` is the single new operational information cutoff,
    not the wall-clock report generation time.
    """
    from chronos2_hourly.nyx_saved_horizon_sources import collect_horizon_sources
    from chronos2_hourly.nyx_daily_two_day_baseline import build_two_day_baseline
    from chronos2_hourly.nyx_annual_live_preflight import ROOT as installed_repository_root
    require(Path(repo).resolve() == Path(installed_repository_root).resolve(),
            "Horizon repository differs from the installed annual pipeline")
    seed_bundle, seed_output = Path(seed_bundle).resolve(), Path(seed_output).resolve()
    output_directory = Path(output_directory).resolve()
    require(not output_directory.is_relative_to(seed_bundle)
            and not output_directory.is_relative_to(seed_output), "New horizon output cannot overwrite the completed run")
    output_directory.mkdir(parents=True, exist_ok=True)
    cutoff = pd.Timestamp(forecast_issued_at_utc)
    require(cutoff.tzinfo is not None, "Timezone-aware operational cutoff required")
    cutoff = cutoff.tz_convert("UTC")
    seed_receipt = json.loads((seed_output / "receipt.json").read_text(encoding="utf-8"))
    require(seed_receipt.get("status") == "REPORTS_COMPLETE", "Completed seed report receipt required")
    bound_inputs = seed_receipt.get("input_sha256")
    require(isinstance(bound_inputs, dict) and bound_inputs, "Completed seed source hashes missing")
    for relative in bound_inputs:
        _bound_path(seed_bundle, relative, bound_inputs)
    before = {str(path.relative_to(seed_bundle)): digest(path)
              for path in seed_bundle.rglob("*") if path.is_file()}
    source = collect_horizon_sources(seed_bundle, first_day, cutoff, forecast_days=2,
        retained_covariates=carry_state["covariates"] if carry_state else None,
        retained_covariates_evidence=carry_state["covariates_evidence"] if carry_state else None)
    produced = build_two_day_baseline(seed_bundle, targets_by_zone=source["prices"],
        covariates=source["covariates"], first_delivery_day=first_day,
        asof_cutoff_utc=cutoff, output_dir=output_directory / "baseline_extension", threads=2,
        source_audit=source["audit"], carry_state=carry_state)
    quantiles, origins, baselines = {}, {}, {}
    current = grid(first_day)
    covariates = source["covariates"]
    from chronos2_hourly.nyx_local_extra_features import cutoff_times
    logical = cutoff_times(covariates.index, "Europe/Paris")
    donor_origins = pd.read_parquet(seed_bundle / "source_artifacts/saturn/origins.parquet")
    require("forecast_origin_utc" in donor_origins, "Saved logical source origin history missing")
    retained = logical.index.intersection(donor_origins.index)
    logical.loc[retained] = pd.to_datetime(donor_origins.loc[retained, "forecast_origin_utc"], utc=True)
    logical.loc[current] = cutoff
    for zone in ZONES:
        history = produced["zones"][zone]["history"].copy()
        future = produced["zones"][zone]["baseline"].copy()
        # A baseline producer may include the two future days in its extended
        # history. Keep one exact row per physical hour and prefer its fresh
        # baseline values for the requested horizon.
        combined = pd.concat([history.loc[~history.index.isin(current)], future.loc[current]]).sort_index()
        require(combined.index.is_unique and {"nyx__q10", "nyx__q50", "nyx__q90"} <= set(combined),
                f"{zone}: bounded producer baseline schema differs")
        combined["actual"] = source["prices"][zone].reindex(combined.index)
        combined.loc[current, "actual"] = np.nan
        require(np.isnan(combined.loc[current, "actual"].to_numpy(float)).all(), "Future labels entered baseline output")
        quantiles[zone] = combined[["nyx__q10", "nyx__q50", "nyx__q90"]].rename(columns=lambda name: name[5:])
        origins[zone] = logical.copy()
        require("forecast_origin_utc" in combined, f"{zone}: retained prediction origin history missing")
        retained_prediction_origins = pd.to_datetime(combined.forecast_origin_utc, utc=True)
        require(retained_prediction_origins.notna().all() and (retained_prediction_origins <= cutoff).all(),
                f"{zone}: retained prediction was issued after the new source cutoff")
        combined["forecast_origin_utc"] = retained_prediction_origins
        combined.loc[current, "forecast_origin_utc"] = cutoff
        first_history = pd.Timestamp(pd.Timestamp(first_day).date() - timedelta(days=365), tz="Europe/Paris").tz_convert("UTC")
        baselines[zone] = combined.loc[combined.index >= first_history]
    built = assemble_horizon_features(first_day, cutoff, sources=source, quantiles=quantiles, origins=origins)
    forecasts, model_audit = infer_saved_horizon(seed_bundle, seed_output, first_day, cutoff, built, quantiles)
    model_audit["bounded_baseline"] = produced["audit"]
    model_audit["new_operational_context"] = context
    saved = {}
    for group in ("covariates", "fuel", "hydro_features", "exchange_features", "jao_original", "jao_refreshed"):
        path = output_directory / "horizon_sources" / (group + ".parquet")
        path.parent.mkdir(parents=True, exist_ok=True)
        source[group].to_parquet(path)
        saved[str(path.relative_to(output_directory))] = digest(path)
    for group in ("prices", "thermal_features"):
        for zone, frame in source[group].items():
            path = output_directory / "horizon_sources" / group / (zone + ".parquet")
            path.parent.mkdir(parents=True, exist_ok=True)
            (frame.to_frame() if isinstance(frame, pd.Series) else frame).to_parquet(path)
            saved[str(path.relative_to(output_directory))] = digest(path)
    source_audit = {**source["audit"], "source_artifact_sha256": saved}
    (output_directory / "horizon_sources" / "receipt.json").write_text(
        json.dumps(source_audit, indent=2, ensure_ascii=False, allow_nan=False, default=str) + "\n", encoding="utf-8")
    after = {str(path.relative_to(seed_bundle)): digest(path)
             for path in seed_bundle.rglob("*") if path.is_file()}
    require(before == after, "Completed source bundle changed during horizon extension")
    from chronos2_hourly.nyx_daily_state_cache import publish_state
    model_audit["rolling_daily_state"] = publish_state(cache_root, cache_contract,
        produced=produced, covariates=covariates, cutoff=cutoff)
    return {"forecasts": forecasts, "baseline": baselines, "covariates": covariates,
            "model_audit": model_audit, "sources_audit": source_audit,
            "cutoff": cutoff, "first_delivery_day": first_day, "forecast_days": 2}


def produce_horizon(repo, seed_bundle, seed_output, first_day, forecast_issued_at_utc,
                    output_directory, context=None):
    """Serialize owned state updates and retain earlier genuine daily predictions."""
    from chronos2_hourly.nyx_daily_state_cache import build_contract, load_state
    from chronos2_hourly.process_lock import exclusive_process_lock
    contract = build_contract(seed_bundle, seed_output, threads=2)
    cache_root = Path(repo).resolve() / "runs" / "nyx_daily_cpu_states" / contract["seed_report_receipt_sha256"]
    cache_root.mkdir(parents=True, exist_ok=True)
    with exclusive_process_lock(cache_root / ".daily_state.lock"):
        carry_state = load_state(cache_root, contract, first_delivery_day=first_day,
                                 cutoff=forecast_issued_at_utc)
        from chronos2_hourly.nyx_daily_two_day_baseline import MAX_BRIDGE_DAYS
        seed_day = json.loads((Path(seed_bundle) / "source_receipts/nyx_quantiles.json").read_text(encoding="utf-8"))["delivery_day"]
        anchor = carry_state["last_history_day"] if carry_state else seed_day
        gap = (pd.Timestamp(first_day).date() - pd.Timestamp(anchor).date()).days - 1
        require(0 <= gap <= MAX_BRIDGE_DAYS,
                "Daily continuation requires a compatible retained state with at most seven missing recent days")
        return _produce_horizon(repo, seed_bundle, seed_output, first_day, forecast_issued_at_utc,
            output_directory, context, carry_state=carry_state, cache_root=cache_root,
            cache_contract=contract)
