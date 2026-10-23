"""Append a two-day CPU NYX forecast to a verified annual seed.

Only missing recent dates and the final 365-day residual/Kalman window are
reconstructed.  The annual replay and the fitted annual experts are untouched.
Both forecast days use one recorded issue cutoff; future prices are never
training labels or Kalman observations.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from datetime import timedelta
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from chronos2_hourly import nyx_annual_cpu_baseline as annual
from chronos2_hourly import kalman_residual as kalman
from chronos2_hourly.features import build_history_future_feature_matrix, validate_utc_hourly_index
from chronos2_hourly.models.residual_corrector import ResidualCorrector, ResidualMetaFeatureBuilder
from chronos2_hourly.nyx_live_baseline import make_residual_factory
from chronos2_hourly.nyx_live_hybrid import build_pair_interaction
from chronos2_hourly.nyx_annual_epex_prices import EPEX_SERIES, validate_price_identity
from chronos2_hourly.process_lock import exclusive_process_lock
from chronos2_hourly.solar_wind_forecast import INPUT_ALIASES, solar_wind_kalman_covariate_config


PROTOCOL = "nyx_daily_two_day_baseline_v1"
MAX_BRIDGE_DAYS = 7


def _grid(first, stop):
    return annual._grid(first, stop)


def _finite(frame, name):
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError(f"{name}: incomplete or nonfinite physical-hour values")


def predict_chronos_horizon(*, target, covariates, zone, first_delivery_day,
                            asof_cutoff_utc, pipeline, context_length=2048):
    """One native call for both complete civil days (47/48/49 physical hours)."""
    day = pd.Timestamp(first_delivery_day).date()
    future_index = _grid(day, day + timedelta(days=2))
    validate_utc_hourly_index(target.index, name="two-day observed target")
    history_index = pd.date_range(end=future_index[0] - pd.Timedelta(hours=1),
                                  periods=context_length, freq="h", tz="UTC")
    labels = target.reindex(history_index)
    _finite(labels, f"{zone} Chronos observed context")
    expanded = covariates.reindex(covariates.index.union(history_index))
    context, future = annual.canonical_covariates(expanded, zone=zone)
    known = future.reindex(future_index)
    _finite(known, f"{zone} two-day Chronos future covariates")
    past = context.reindex(history_index)
    inputs = [{"target": labels.to_numpy(dtype=np.float32),
               "past_covariates": {c: past[c].to_numpy(dtype=np.float32) for c in past},
               "future_covariates": {c: known[c].to_numpy(dtype=np.float32) for c in known}}]
    predictions, _ = pipeline.predict_quantiles(
        inputs, prediction_length=len(known), quantile_levels=[.1, .5, .9],
        batch_size=max(128, len(context.columns) + 1), context_length=context_length,
        cross_learning=False)
    values = predictions[0]
    if hasattr(values, "detach"):
        values = values.detach().cpu().float().numpy()
    values = np.asarray(values)
    if (values.shape != (1, len(known), 3) or not np.isfinite(values).all()
            or (np.diff(values, axis=2) < 0).any()):
        raise ValueError("Invalid native two-day Chronos quantiles")
    result = pd.DataFrame(values[0].astype(np.float64), index=future_index,
                          columns=annual.QUANTILES)
    result["forecast_origin_utc"] = pd.Timestamp(asof_cutoff_utc).tz_convert("UTC")
    return result


def predict_residual_horizon(*, raw_history, raw_future, covariates, zone,
                            configuration, interaction):
    """Fit once on 365 real-label days; accept genuine shared-cutoff origins."""
    timezone = annual.residual.ZONE_TIMEZONES[zone]
    validate_utc_hourly_index(raw_history.index, name="bounded corrector historical index")
    validate_utc_hourly_index(raw_future.index, name="bounded corrector future index")
    future_days = tuple(pd.Index(raw_future.index.tz_convert(timezone).date).unique())
    if len(future_days) not in (1, 2):
        raise ValueError("Bounded corrector predicts one or two complete civil days")
    first_day = future_days[0]
    expected_future = pd.date_range(str(first_day), str(first_day + timedelta(days=len(future_days))),
        tz=timezone, freq="h", inclusive="left").tz_convert("UTC")
    expected_history = pd.date_range(str(first_day - timedelta(days=365)), str(first_day),
        tz=timezone, freq="h", inclusive="left").tz_convert("UTC")
    if not raw_future.index.equals(expected_future) or not raw_history.index.equals(expected_history):
        raise ValueError("Bounded corrector requires exactly 365 complete days strictly before its future interval")
    if "actual" not in raw_history:
        raise ValueError("Bounded corrector requires real historical labels")
    _finite(raw_history.actual, f"{zone} strictly historical corrector labels")
    if "actual" in raw_future and raw_future.actual.notna().any():
        raise ValueError("Future actual values must never enter bounded corrector inference")
    for label, frame in (("history", raw_history), ("future", raw_future)):
        if "forecast_origin_utc" not in frame:
            raise ValueError(f"Bounded corrector {label} lacks genuine forecast origins")
        stamps = [pd.Timestamp(value) for value in frame.forecast_origin_utc]
        if any(pd.isna(stamp) or stamp.tzinfo is None for stamp in stamps):
            raise ValueError("Bounded corrector forecast origins must be explicit timezone-aware timestamps")
    future_origins = pd.to_datetime(raw_future.forecast_origin_utc, utc=True)
    history_origins = pd.to_datetime(raw_history.forecast_origin_utc, utc=True)
    if future_origins.nunique() != 1 or (history_origins > future_origins.iloc[0]).any():
        raise ValueError("Bounded corrector historical origins exceed the shared information cutoff")
    factory = make_residual_factory(configuration, interaction)
    fitted = factory()
    builder = fitted.feature_builder or ResidualMetaFeatureBuilder(**fitted.feature_builder_options)
    # The pinned annual recipe excludes price-history features.  A changed
    # recipe must introduce its own explicit recursive-feature policy.
    if not builder.exclude_historical_prices:
        raise ValueError("Two-day baseline requires the pinned residual price-feature exclusion")
    _, known = annual.canonical_covariates(covariates, zone=zone)
    options = configuration["hourly"]["feature_engineering"]
    features = build_history_future_feature_matrix(
        raw_history.actual, known.reindex(raw_history.index), known.reindex(raw_future.index),
        price_lags=options.get("target_lags", (24, 48, 168)),
        rolling_windows=options.get("target_rolling_windows", (24, 168)),
        timezone=annual.residual.ZONE_TIMEZONES[zone])
    features = features.loc[:, [c for c in features if not builder._is_excluded(c)]]
    _finite(features, f"{zone} retained residual features")
    train_base = raw_history.loc[:, list(annual.QUANTILES)]
    future_base = raw_future.loc[:, list(annual.QUANTILES)]
    train_expert = train_base.rename(columns=lambda c: "chronos2__" + c)
    future_expert = future_base.rename(columns=lambda c: "chronos2__" + c)
    fitted.fit(features.loc[raw_history.index], raw_history.actual, train_base, train_expert)
    corrected = fitted.predict(features.loc[raw_future.index], future_base, future_expert)
    result = raw_future.copy()
    for q in annual.QUANTILES:
        result[f"chronos2__{q}"] = raw_future[q]
        result[f"residual_corrected__{q}"] = corrected[q]
    result["residual_correction"] = corrected.q50 - raw_future.q50
    result["actual"] = np.nan
    # Persist a normal importable class instead of the local factory subclass.
    # Inference adds the independently frozen interaction explicitly.
    portable = ResidualCorrector()
    portable.__dict__ = deepcopy(fitted.__dict__)
    audit = {"generation_source": "one_bounded_365day_corrector_fit",
             "training_rows": int(fitted.n_training_rows_),
             "training_start_utc": raw_history.index[0].isoformat(),
             "training_end_utc": raw_history.index[-1].isoformat(),
             "forecast_hours": len(raw_future), "future_labels_used": False,
             "training_days": 365, "training_labels_strictly_before_future": True,
             "source_asof_cutoff_utc": future_origins.iloc[0].isoformat(),
             "forecast_origins_rewritten_to_nominal_dates": False,
             "future_price_features_excluded": True,
             "saved_inference_interaction_column": str(interaction.columns[0])}
    return result, audit, portable


def _fit_kalman_horizon(*, past, future_blocks, training_days, columns,
                       candidate_columns, config):
    """Use the pinned rolling primitives, then transition without assimilation."""
    feature_frame = pd.concat([past, *future_blocks], axis=0)
    market, scalers = kalman._market_features(
        feature_frame, covariate_columns=columns, calibration_days=set(training_days),
        clip=config.market_feature_clip)
    variance = kalman._observation_variance(past, calibration_days=set(training_days))
    candidates = {kind: kalman._Candidate(kind=kind, config=config,
        observation_variance=variance, market=market,
        market_feature_columns=candidate_columns[kind]) for kind in config.candidate_kinds}
    governance_days = set(training_days[-config.governance_lookback_days:])
    realised_parts = []
    for day, block in past.groupby("_local_day", sort=True):
        block = block.sort_index()
        for candidate in candidates.values():
            candidate.transition_day()
        if day in governance_days:
            shifts = {kind: np.clip(candidate.raw_correction(block),
                      -config.shift_clip_eur_mwh, config.shift_clip_eur_mwh)
                      for kind, candidate in candidates.items()}
            realised_parts.append(pd.DataFrame({"local_day": day,
                "actual": block.actual.to_numpy(float),
                "base": block["residual_corrected__q50"].to_numpy(float),
                **{f"raw::{kind}": shift for kind, shift in shifts.items()}}))
        for _, row in block.iterrows():
            for candidate in candidates.values():
                candidate.update_hour_rolling(row)
    realised = pd.concat(realised_parts, ignore_index=True) if realised_parts else pd.DataFrame()
    selected, weight, losses = kalman._governance_choice(
        realised, candidate_kinds=tuple(candidates), config=config)
    outputs, daily, state_rows = [], [], []
    for horizon, block in enumerate(future_blocks, start=1):
        for candidate in candidates.values():
            candidate.transition_day()
        shifts = {kind: np.clip(candidate.raw_correction(block),
                  -config.shift_clip_eur_mwh, config.shift_clip_eur_mwh)
                  for kind, candidate in candidates.items()}
        raw_shift = np.zeros(len(block)) if selected == "identity" else shifts[selected]
        applied = weight * raw_shift
        result = pd.DataFrame(index=block.index)
        for q in annual.QUANTILES:
            result[f"nyx__{q}"] = block[f"residual_corrected__{q}"].to_numpy(float) + applied
        result["kalman_raw_correction"] = raw_shift
        result["kalman_weight"] = weight
        result["kalman_correction"] = applied
        result["kalman_selected_filter"] = selected
        outputs.append(result)
        daily.append({"local_day": str(block["_local_day"].iloc[0]),
            "horizon_day": horizon, "hours": len(block), "selected_filter": selected,
            "selected_weight": float(weight), "training_window_days": len(training_days),
            "training_window_hours": len(past), "target_observations_assimilated": 0,
            "training_end_utc": past.index[-1].isoformat(),
            "applied_correction_abs_max": float(np.abs(applied).max())})
        state_rows.append({kind: {"mean": candidate.mean.tolist(),
            "covariance": candidate.covariance.tolist(),
            "feature_names": list(candidate.feature_names),
            "transition": candidate.transition.tolist(), "q": candidate.q.tolist(),
            "observation_variance": candidate.r,
            "market_feature_columns": list(candidate.market_feature_columns)}
            for kind, candidate in candidates.items()})
    state = {"protocol": PROTOCOL, "config": asdict(config), "market_scalers": scalers,
             "selected_filter": selected, "selected_weight": float(weight),
             "state_at_each_forecast": state_rows, "future_observations_assimilated": 0}
    return pd.concat(outputs), {"daily": daily, "governance_losses": losses,
        "last_observation_used_utc": past.index[-1].isoformat(),
        "training_window_days": len(training_days), "future_labels_used": False}, state


def predict_kalman_horizon(*, history, future, covariates, zone, configuration):
    config, _ = annual.residual._kalman_filter_configuration(configuration)
    cov_config = solar_wind_kalman_covariate_config()
    timezone = annual.residual.ZONE_TIMEZONES[zone]
    training_days = tuple(pd.Index(history.index.tz_convert(timezone).date).unique())
    future_days = tuple(pd.Index(future.index.tz_convert(timezone).date).unique())
    if (len(training_days) != 365 or len(future_days) != 2
            or training_days[-1] != future_days[0] - timedelta(days=1)
            or not history.index.equals(_grid(future_days[0] - timedelta(days=365), future_days[0]))
            or not future.index.equals(_grid(future_days[0], future_days[0] + timedelta(days=2)))):
        raise ValueError("Kalman requires 365 preceding civil days and two future days")
    if "actual" in future and future.actual.notna().any():
        raise ValueError("Future actuals forbidden in two-day Kalman inference")
    past, columns, _ = kalman._normalise_input(
        history.rename_axis("delivery_start_utc").reset_index(),
        upstream_model="residual_corrected", timezone=timezone,
        covariates=covariates.loc[history.index, list(INPUT_ALIASES)]
        .rename_axis("timestamp").reset_index(), covariate_config=cov_config)
    if not past.index.equals(history.index):
        raise ValueError("Kalman normalization dropped a historical training hour")
    blocks = []
    for day in future_days:
        index = kalman._expected_local_day_index(day, timezone=timezone)
        blocks.append(kalman._normalise_future_input(
            future.loc[index].drop(columns="actual", errors="ignore")
            .rename_axis("delivery_start_utc").reset_index(),
            upstream_model="residual_corrected", timezone=timezone,
            covariates=covariates.loc[index, list(INPUT_ALIASES)]
            .rename_axis("timestamp").reset_index(), covariate_columns=columns,
            covariate_config=cov_config))
    candidate_columns = {kind: kalman._candidate_market_feature_columns(kind,
        covariate_config=cov_config, covariate_columns=columns) for kind in config.candidate_kinds}
    return _fit_kalman_horizon(past=past, future_blocks=blocks, training_days=training_days,
        columns=columns, candidate_columns=candidate_columns, config=config)


def _bound_frame(bundle, receipt, relative):
    path = bundle / relative
    expected = receipt["artifact_sha256"].get(relative)
    if not expected or not path.is_file() or annual.sha256(path) != expected:
        raise ValueError(f"Annual seed artifact changed or unbound: {relative}")
    return pd.read_parquet(path)


def _checkpoint(bundle, receipt, zone, day):
    directory = f"baseline_checkpoints/{zone}/{day}"
    relative = directory + "/receipt.json"
    path = bundle / relative
    expected = receipt["artifact_sha256"].get(relative)
    if not expected or not path.is_file() or annual.sha256(path) != expected:
        raise ValueError(f"Annual seed checkpoint changed or unbound: {relative}")
    record = json.loads(path.read_text(encoding="utf-8"))
    output = []
    for name in ("chronos.parquet", "residual.parquet"):
        relative = directory + "/" + name
        if record["files"].get(name) != receipt["artifact_sha256"].get(relative):
            raise ValueError(f"Seed checkpoint graph differs: {relative}")
        output.append(_bound_frame(bundle, receipt, relative))
    return (*output, record)


def build_two_day_baseline(seed_bundle, *, targets_by_zone, covariates,
                           first_delivery_day, asof_cutoff_utc, output_dir,
                           threads=2, pipeline=None, source_audit=None, carry_state=None):
    """Return real two-day forecasts and frozen-seed history; write isolated evidence.

    ``covariates`` retains old frozen vintage rows and appends new rows captured
    at ``asof_cutoff_utc``.  The source-capture caller owns provider provenance;
    numeric inputs and its supplied audit are bound by this module's receipt.
    """
    seed = Path(seed_bundle).resolve()
    out = Path(output_dir).resolve()
    if out == seed or out.is_relative_to(seed):
        raise ValueError("Two-day outputs must be outside the sealed annual seed")
    day = pd.Timestamp(first_delivery_day).date()
    cutoff = pd.Timestamp(asof_cutoff_utc)
    if pd.isna(cutoff) or cutoff.tzinfo is None:
        raise ValueError("Explicit timezone-aware issue cutoff required")
    cutoff = cutoff.tz_convert("UTC")
    required_cutoff = pd.Timestamp(str(day - timedelta(days=1)) + " 08:00").tz_localize("Europe/Paris")
    if cutoff != required_cutoff.tz_convert("UTC"):
        raise ValueError("Both forecast days require the first day's D-1 08:00 issue cutoff")
    receipt_path = seed / "source_receipts/nyx_quantiles.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    saturn_path = seed / "source_receipts/saturn.json"
    if receipt.get("source_receipts_sha256", {}).get("saturn") != annual.sha256(saturn_path):
        raise ValueError("Annual NYX seed is not bound to its Saturn source receipt")
    saturn = json.loads(saturn_path.read_text(encoding="utf-8"))
    validate_price_identity(saturn)
    if saturn.get("price_series") != dict(EPEX_SERIES):
        raise ValueError("Annual seed must use the four selected EPEX price series")
    seed_day = pd.Timestamp(receipt["delivery_day"]).date()
    anchor_day = seed_day
    if carry_state is not None:
        carry_cutoff = pd.Timestamp(carry_state["asof_cutoff_utc"])
        contract = carry_state["contract"]
        anchor_day = pd.Timestamp(carry_state["last_history_day"]).date()
        if (carry_cutoff.tzinfo is None or carry_cutoff.tz_convert("UTC") > cutoff
                or contract.get("seed_baseline_receipt_sha256") != annual.sha256(receipt_path)
                or contract.get("price_policy") != "epex_only_v1"
                or contract.get("price_series") != dict(EPEX_SERIES)
                or not seed_day <= anchor_day < day):
            raise ValueError("Rolling carry state differs from the EPEX seed or current issue")
    if not seed_day < day or not 0 <= (day - anchor_day).days - 1 <= MAX_BRIDGE_DAYS:
        raise ValueError("Seed must precede forecast by at most seven uncomputed bridge days")
    if set(targets_by_zone) != set(annual.ZONES) or not set(annual.RAW_ALIASES).issubset(covariates):
        raise ValueError("Four EPEX targets and fourteen annual covariate aliases required")
    validate_utc_hourly_index(covariates.index, name="two-day source covariates")
    first_read = min(day - timedelta(days=365), anchor_day + timedelta(days=1) - timedelta(days=365))
    finite_cov_index = _grid(first_read - timedelta(days=365), day + timedelta(days=2))
    _finite(covariates.reindex(finite_cov_index).loc[:, list(annual.RAW_ALIASES)],
            "Causal interaction history and two-day future covariates")
    future_index = _grid(day, day + timedelta(days=2))
    label_index = _grid(first_read, day)
    for zone, target in targets_by_zone.items():
        validate_utc_hourly_index(target.index, name=f"{zone} observed EPEX target")
        if (target.index >= future_index[0]).any():
            raise ValueError(f"{zone}: future rows forbidden in observed training targets")
        _finite(target.reindex(label_index), f"{zone} real historical training labels")
    if not isinstance(threads, int) or threads < 1:
        raise ValueError("Positive integer CPU thread count required")
    configurations = {z: annual._configuration(z, threads) for z in annual.ZONES}
    dependency_hashes = {relative: annual.sha256_text(annual.ROOT / relative)
                         for relative in annual.CODE_FILES}
    runtime = {name: annual.importlib.metadata.version(name) for name in
        ("torch", "chronos-forecasting", "numpy", "pandas", "catboost", "pykalman", "scipy", "holidays")}
    if carry_state is not None:
        contract = carry_state["contract"]
        if (contract.get("dependencies") != dependency_hashes
                or contract.get("runtime") != runtime or contract.get("cpu_threads") != threads
                or contract.get("configuration_sha256") != {z: configurations[z][1] for z in annual.ZONES}):
            raise ValueError("Rolling carry baseline implementation, configuration or runtime changed")
    identity = {"protocol": PROTOCOL, "seed_receipt_sha256": annual.sha256(receipt_path),
        "first_delivery_day": str(day), "second_delivery_day": str(day + timedelta(days=1)),
        "asof_cutoff_utc": cutoff.isoformat(), "cpu_threads": threads,
        "targets": {z: annual.frame_digest(targets_by_zone[z]) for z in annual.ZONES},
        "covariates_sha256": annual.frame_digest(covariates), "source_audit": source_audit or {},
        "dependency_sha256": dependency_hashes, "runtime": runtime,
        "configuration_sha256": {z: configurations[z][1] for z in annual.ZONES},
        "rolling_carry_receipt_sha256": carry_state["receipt_sha256"] if carry_state else None,
        "implementation_sha256": annual.sha256_text(Path(__file__))}
    identity = json.loads(annual._json(identity))
    out.mkdir(parents=True, exist_ok=True)
    with exclusive_process_lock(out / ".two_day_baseline.lock"):
        saved_path = out / "receipt.json"
        if saved_path.exists():
            saved = json.loads(saved_path.read_text(encoding="utf-8"))
            if saved["identity"] != identity:
                raise ValueError("Completed two-day output has another immutable input identity")
            for relative, expected in saved["artifact_sha256"].items():
                if annual.sha256(out / relative) != expected:
                    raise ValueError(f"Two-day forecast artifact changed: {relative}")
            return {"zones": {z: {name: pd.read_parquet(out / f"{z}/{name}.parquet")
                    for name in ("chronos", "residual", "baseline", "history", "raw_history", "residual_history")}
                    for z in annual.ZONES}, "audit": saved["audit"], "receipt": saved}
        if pipeline is None:
            pipeline = annual.load_cpu_chronos(allow_download=False)
        model_identity = getattr(pipeline, "_nyx_annual_cpu_identity", None)
        model_hash = annual.chronos_identity_digest(model_identity)
        if carry_state is not None and carry_state["contract"].get("chronos_model_sha256") != model_hash:
            raise ValueError("Rolling carry and live inference use different Chronos weights")
        results, audits, artifacts = {}, {}, {}
        for zone in annual.ZONES:
            raw_parts, residual_parts = [], []
            if carry_state is None:
                for old_day in pd.date_range(str(first_read), str(seed_day), freq="D").date:
                    raw, corrected, record = _checkpoint(seed, receipt, zone, str(old_day))
                    if record["identity"]["model_sha256"] != model_hash:
                        raise ValueError("Seed and two-day inference use different pretrained Chronos weights")
                    raw_parts.append(raw)
                    residual_parts.append(corrected)
                history = _bound_frame(seed, receipt, f"baseline_history/{zone}.parquet")
            else:
                carry = carry_state["zones"][zone]
                wanted = _grid(first_read, anchor_day + timedelta(days=1))
                for name, parts in (("raw_history", raw_parts), ("residual_history", residual_parts)):
                    frame = carry[name]
                    validate_utc_hourly_index(frame.index, name="rolling " + name)
                    if not wanted.isin(frame.index).all() or (frame.index >= future_index[0]).any():
                        raise ValueError("Rolling carry has incomplete or future baseline fitting rows")
                    restored = frame.loc[wanted].copy()
                    restored["actual"] = targets_by_zone[zone].reindex(wanted)
                    _finite(restored.actual, "New real EPEX labels for retained cached predictions")
                    parts.append(restored)
                history = carry["history"].copy()
            configuration, config_hashes = configurations[zone]
            score, interaction_audit = build_pair_interaction(covariates, zone,
                annual.residual.ZONE_TIMEZONES[zone])
            bridge_audits = []
            for bridge_day in pd.date_range(str(anchor_day + timedelta(days=1)),
                                           str(day - timedelta(days=1)), freq="D").date:
                print(json.dumps({"stage": "cpu_daily_baseline", "zone": zone,
                    "phase": "bridge_start", "delivery_day": str(bridge_day)}), flush=True)
                print(json.dumps({"stage": "cpu_daily_baseline", "zone": zone,
                    "phase": "bridge_chronos", "delivery_day": str(bridge_day)}), flush=True)
                raw = annual.predict_chronos_day(target=targets_by_zone[zone], covariates=covariates,
                    zone=zone, delivery_day=str(bridge_day), pipeline=pipeline)
                raw["forecast_origin_utc"] = cutoff
                train_index = _grid(bridge_day - timedelta(days=365), bridge_day)
                raw_history = pd.concat(raw_parts).loc[train_index].copy()
                raw_history["actual"] = targets_by_zone[zone].reindex(train_index)
                print(json.dumps({"stage": "cpu_daily_baseline", "zone": zone,
                    "phase": "bridge_residual_fit", "delivery_day": str(bridge_day)}), flush=True)
                corrected, corrector_audit, _ = predict_residual_horizon(raw_history=raw_history,
                    raw_future=raw, covariates=covariates, zone=zone,
                    configuration=configuration, interaction=score)
                for q in annual.QUANTILES:
                    corrected[q] = corrected[f"residual_corrected__{q}"]
                corrected["forecast_origin_utc"] = cutoff
                corrected_history = pd.concat(residual_parts).loc[train_index].copy()
                corrected_history["actual"] = targets_by_zone[zone].reindex(train_index)
                print(json.dumps({"stage": "cpu_daily_baseline", "zone": zone,
                    "phase": "bridge_kalman", "delivery_day": str(bridge_day)}), flush=True)
                bridge, kalman_audit = annual.predict_kalman_day(history=corrected_history,
                    future=corrected, covariates=covariates, zone=zone,
                    delivery_day=str(bridge_day), configuration=configuration)
                bridge["forecast_origin_utc"] = cutoff
                bridge["actual"] = targets_by_zone[zone].reindex(bridge.index)
                history = pd.concat([history, bridge])
                raw_parts.append(raw)
                residual_parts.append(corrected)
                bridge_audits.append({"day": str(bridge_day),
                    "generation_source": "new_cutoff_reconstruction_for_training_inputs",
                    "source_asof_cutoff_utc": cutoff.isoformat(),
                    "historical_original_origin_verified": False,
                    "residual": corrector_audit, "kalman": kalman_audit})
                print(json.dumps({"stage": "cpu_daily_baseline", "zone": zone,
                    "phase": "bridge_complete", "delivery_day": str(bridge_day)}), flush=True)
            train_index = _grid(day - timedelta(days=365), day)
            raw_history = pd.concat(raw_parts).loc[train_index].copy()
            raw_history["actual"] = targets_by_zone[zone].reindex(train_index)
            corrected_history = pd.concat(residual_parts).loc[train_index].copy()
            corrected_history["actual"] = targets_by_zone[zone].reindex(train_index)
            print(json.dumps({"stage": "cpu_daily_baseline", "zone": zone,
                "phase": "two_day_start", "first_delivery_day": str(day)}), flush=True)
            print(json.dumps({"stage": "cpu_daily_baseline", "zone": zone,
                "phase": "two_day_chronos", "first_delivery_day": str(day)}), flush=True)
            raw = predict_chronos_horizon(target=targets_by_zone[zone], covariates=covariates,
                zone=zone, first_delivery_day=str(day), asof_cutoff_utc=cutoff,
                pipeline=pipeline)
            print(json.dumps({"stage": "cpu_daily_baseline", "zone": zone,
                "phase": "two_day_residual_fit", "first_delivery_day": str(day)}), flush=True)
            corrected, residual_audit, saved_corrector = predict_residual_horizon(
                raw_history=raw_history, raw_future=raw, covariates=covariates,
                zone=zone, configuration=configuration, interaction=score)
            print(json.dumps({"stage": "cpu_daily_baseline", "zone": zone,
                "phase": "two_day_kalman", "first_delivery_day": str(day)}), flush=True)
            baseline, kalman_audit, state = predict_kalman_horizon(history=corrected_history,
                future=corrected, covariates=covariates, zone=zone, configuration=configuration)
            baseline["forecast_origin_utc"] = cutoff
            baseline["actual"] = np.nan
            # Historical scoring labels are the explicitly recorded new-cutoff
            # real prices. Quantile outputs retain their own seed lineage.
            history = history.loc[history.index < future_index[0]].copy()
            history["actual"] = targets_by_zone[zone].reindex(history.index)
            history = pd.concat([history, baseline]).sort_index()
            results[zone] = {"chronos": raw, "residual": corrected,
                             "baseline": baseline, "history": history,
                             "raw_history": pd.concat([raw_history, raw]).sort_index(),
                             "residual_history": pd.concat([corrected_history, corrected]).sort_index()}
            audits[zone] = {"bridge": bridge_audits, "residual": residual_audit,
                "kalman": kalman_audit, "configuration_sha256": config_hashes,
                "chronos_model_sha256": model_hash,
                "interaction": {k: v for k, v in interaction_audit.items() if k != "normalizations"}}
            for name, frame in results[zone].items():
                relative = f"{zone}/{name}.parquet"
                annual.write_frame(out / relative, frame)
                artifacts[relative] = annual.sha256(out / relative)
            relative = f"{zone}/kalman_inference_state.json"
            annual.write_json(out / relative, state)
            artifacts[relative] = annual.sha256(out / relative)
            import joblib
            relative = f"{zone}/residual_corrector.joblib"
            joblib.dump(saved_corrector, out / relative)
            artifacts[relative] = annual.sha256(out / relative)
            relative = f"{zone}/future_interaction.parquet"
            annual.write_frame(out / relative, score.loc[future_index])
            artifacts[relative] = annual.sha256(out / relative)
            print(json.dumps({"stage": "cpu_daily_baseline", "zone": zone,
                "phase": "two_day_complete", "first_delivery_day": str(day)}), flush=True)
        audit = {"protocol": PROTOCOL, "future_labels_used": False,
            "annual_835_day_replay_run": False, "annual_expert_models_refitted": False,
            "bounded_corrector_fit_per_zone": True, "two_day_cutoff_shared": True,
            "source_asof_cutoff_utc": cutoff.isoformat(), "zones": audits,
            "rolling_carry_state_used": carry_state is not None,
            "rolling_carry_receipt_sha256": carry_state["receipt_sha256"] if carry_state else None,
            "last_retained_prediction_day": str(anchor_day), "cached_labels_used_for_training": False,
            "historical_metrics_complete_new_full_model_oos": False}
        saved = {"identity": identity, "artifact_sha256": artifacts, "audit": audit}
        annual.write_json(saved_path, saved)
        return {"zones": results, "audit": audit, "receipt": saved}


__all__ = ["build_two_day_baseline", "predict_chronos_horizon",
           "predict_residual_horizon", "predict_kalman_horizon"]
