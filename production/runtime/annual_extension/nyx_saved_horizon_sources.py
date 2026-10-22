"""Fresh J+1/J+2 source inputs for saved NYX model inference.

All newly queried Saturn series share one immutable revision ceiling. Public
sources without an actual pre-ceiling capture remain missing with their
existing availability flags; a retrospective live response is never adopted.
This module fits no model and never changes the completed donor bundle.
"""
from __future__ import annotations

from datetime import date, timedelta
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


ZONES = ("FR", "DE", "BE", "NL")
PROTOCOL = "nyx_saved_horizon_sources_v1"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def _utc(value, name):
    stamp = pd.Timestamp(value)
    require(not pd.isna(stamp) and stamp.tzinfo is not None,
            f"{name}: explicit timezone required")
    return stamp.tz_convert("UTC")


def _day(value):
    day = date.fromisoformat(str(value))
    require(day.isoformat() == str(value), "Expected YYYY-MM-DD")
    return day


def _grid(first, days):
    require(type(days) is int and days in (1, 2), "forecast_days must be 1 or 2")
    first = _day(first)
    return pd.date_range(pd.Timestamp(first, tz="Europe/Paris"),
                         pd.Timestamp(first + timedelta(days=days), tz="Europe/Paris"),
                         freq="h", inclusive="left").tz_convert("UTC").rename("timestamp_utc")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _bound_frame(bundle, group, relative):
    """Read only bytes bound by the completed donor source receipt."""
    bundle = Path(bundle).resolve()
    receipt_path = bundle / "source_receipts" / (group + ".json")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    require(receipt.get("source_group") == group and receipt.get("state") == "COMPLETE",
            f"Donor receipt group differs: {group}")
    target = (bundle / relative).resolve()
    require(target.is_relative_to(bundle), "Donor source escaped bundle")
    expected = receipt.get("artifact_sha256", {}).get(relative)
    require(isinstance(expected, str) and _sha(target) == expected,
            f"Unbound or modified donor source: {relative}")
    frame = pd.read_parquet(target)
    require(_sha(target) == expected, f"Donor source changed during read: {relative}")
    return frame, {"path": relative, "sha256": expected,
                   "receipt_sha256": _sha(receipt_path)}


def _hourly(frame, label, *, finite=False):
    index = frame.index
    require(isinstance(index, pd.DatetimeIndex) and str(index.tz) == "UTC"
            and len(index) and not index.hasnans and index.is_unique
            and index.is_monotonic_increasing and index.equals(index.floor("h")),
            f"{label}: ordered unique UTC hourly index required")
    if finite:
        require(np.isfinite(frame.to_numpy(dtype=float)).all(),
                f"{label}: finite values required")


def _missing(index, columns):
    """Retain exact trained column names and explicit missing-state semantics."""
    frame = pd.DataFrame(np.nan, index=index.copy(), columns=list(columns))
    for column in columns:
        if column.endswith("__available"):
            frame[column] = 0.0
    return frame


def _profiles(client, covariates, future_index, cutoff, *, retained_covariates=None,
              retained_covariates_evidence=None):
    from chronos2_modular.saturn import fetch_saturn_series_from_client
    from chronos2_hourly.nyx_annual_saturn_source import ALIASES, specs

    _hourly(covariates, "Donor covariates", finite=True)
    require(set(ALIASES) == set(covariates.columns), "Donor profile aliases differ")
    cache_audit = None
    if retained_covariates is not None:
        require(isinstance(retained_covariates, pd.DataFrame), "Retained profile DataFrame required")
        require(isinstance(retained_covariates_evidence, dict) and retained_covariates_evidence,
                "Verified rolling profile cache evidence required")
        for key in ("sha256", "receipt_sha256", "seed_receipt_sha256"):
            digest = retained_covariates_evidence.get(key)
            require(isinstance(digest, str) and len(digest) == 64
                    and all(c in "0123456789abcdef" for c in digest),
                    f"Verified rolling profile cache evidence lacks {key}")
        require(isinstance(retained_covariates_evidence.get("path"), str)
                and retained_covariates_evidence["path"], "Verified profile cache path identity required")
        _hourly(retained_covariates, "Retained covariates", finite=True)
        require(set(retained_covariates.columns) == set(ALIASES), "Retained profile aliases differ")
        require(retained_covariates.index[0] == covariates.index[0]
                and covariates.index.isin(retained_covariates.index).all(),
                "Retained profiles must preserve the complete original donor prefix")
        donor_values = covariates.loc[:, list(ALIASES)].to_numpy(dtype=np.float64)
        cache_values = retained_covariates.loc[covariates.index, list(ALIASES)].to_numpy(dtype=np.float64)
        require(np.array_equal(donor_values.view(np.uint64), cache_values.view(np.uint64)),
                "Retained profile cache changed the original donor values")
        require(retained_covariates.index.equals(pd.date_range(retained_covariates.index[0],
                retained_covariates.index[-1], freq="h")), "Retained covariate cache contains a gap")
        covariates = retained_covariates.loc[:, list(ALIASES)].copy()
        preceding = covariates.index[covariates.index < future_index[0]]
        cache_audit = {"evidence": retained_covariates_evidence,
                       "retained_rows_used": len(preceding),
                       "first_retained_utc": preceding[0].isoformat(),
                       "last_retained_utc": preceding[-1].isoformat(),
                       "future_cached_rows_reused": False,
                       "original_donor_prefix_unchanged": True}
    else:
        require(retained_covariates_evidence is None,
                "Rolling profile evidence supplied without a retained frame")
    # Retain all preceding donor rows. Replace any donor prediction rows from
    # the new first delivery onward with a genuinely new source state.
    prior = covariates.loc[covariates.index < future_index[0], list(ALIASES)].copy()
    require(len(prior), "Donor lacks preceding covariate history")
    start = min(prior.index[-1] + pd.Timedelta(hours=1), future_index[0])
    needed = pd.date_range(start, future_index[-1], freq="h").rename("timestamp_utc")
    fresh = pd.DataFrame(index=needed)
    evidence = {}
    for alias, spec in specs().items():
        raw = fetch_saturn_series_from_client(client, spec["series"],
            start=needed[0] - pd.Timedelta(hours=3),
            end=needed[-1] + pd.Timedelta(hours=3), timezone="UTC", revision_date=cutoff,
            naive_timezone=spec["naive_timezone"],
            incomplete_dst_policy=spec["dst_policy"], nocache=True)
        require(isinstance(raw, pd.Series) and isinstance(raw.index, pd.DatetimeIndex)
                and raw.index.is_unique and not raw.index.hasnans,
                f"{alias}: invalid normalized source grid")
        raw.index = raw.index.tz_convert("UTC")
        values = pd.to_numeric(raw, errors="coerce").reindex(needed)
        require(np.isfinite(values.to_numpy(dtype=float)).all(),
                f"{alias}: missing forecast hours at shared cutoff {cutoff.isoformat()}")
        fresh[alias] = values.to_numpy(dtype=float)
        evidence[alias] = {"series": spec["series"], "revision_utc": cutoff.isoformat(),
                           "rows": len(needed), "nocache": True,
                           "source_naive_timezone": spec["naive_timezone"],
                           "incomplete_dst_policy": spec["dst_policy"],
                           "dst_repairs": list(raw.attrs.get("dst_repairs", []))}
    combined = pd.concat([prior, fresh])
    _hourly(combined, "Combined covariates", finite=True)
    require(combined.index.equals(pd.date_range(combined.index[0], future_index[-1], freq="h")),
            "Combined covariate history has a gap")
    return combined, needed, evidence, cache_audit


def _prices(client, bundle, first_day, cutoff):
    from chronos2_hourly.nyx_annual_epex_prices import fetch_epex_prices, price_identity

    stop = pd.Timestamp(_day(first_day), tz="Europe/Paris").tz_convert("UTC")
    prices, evidence = {}, {}
    for zone in ZONES:
        donor, bound = _bound_frame(bundle, "auction_prices",
                                  f"source_artifacts/auction_prices/{zone}.parquet")
        _hourly(donor, f"Donor prices/{zone}")
        require(list(donor.columns) == ["price_eur_mwh"], "Donor price columns differ")
        expected = pd.date_range(donor.index[0], stop, freq="h", inclusive="left")
        prices[zone] = fetch_epex_prices(client, zone, expected, cutoff).rename("price_eur_mwh")
        require(not (prices[zone].index >= stop).any(), "Realized forecast labels forbidden")
        evidence[zone] = {**bound, **price_identity(), "revision_utc": cutoff.isoformat(),
                          "first_utc": expected[0].isoformat(),
                          "last_utc": expected[-1].isoformat(), "rows": len(expected)}
    return prices, evidence


def _fuel(client, donor, first_day, future_index, cutoff):
    """Compute the new known market state, then broadcast it to both days.

    The second forecast day has no future quote or hypothetical price change.
    Its features describe exactly the same observed market state as day one.
    Any bridge days after the donor are explicitly current-fit reconstructions.
    """
    import materialize_saturn_kalman_fuel as fuel

    require(set(donor.columns) == set(fuel.OUTPUT_COLUMNS), "Donor fuel schema differs")
    indexed = donor.copy()
    indexed.index = pd.DatetimeIndex(pd.to_datetime(indexed.value_time_utc, utc=True))
    _hourly(indexed, "Donor fuel")
    first = pd.Timestamp(_day(first_day))
    daily = fuel._recover_daily(indexed)
    daily = daily.loc[daily.index < first]
    require(len(daily) >= fuel.WARMUP_DAYS + 1, "Donor fuel warmup is too short")
    raw = pd.DataFrame(index=daily.index)
    for spec in fuel.MARKET_SERIES:
        raw[spec.alias] = daily[spec.alias]
        source = "ttf_source_value_time_utc" if spec.alias.startswith("ttf") else "eua_source_value_time_utc"
        raw[spec.alias + "__source_value_time_utc"] = daily[source]
    raw["cutoff_time_utc"] = pd.to_datetime(daily.snapshot_time_utc, utc=True)
    append_days = pd.date_range(daily.index[-1] + pd.Timedelta(days=1), first, freq="D")
    market_queries = {}
    for spec in fuel.MARKET_SERIES:
        state = client.get(spec.series,
            from_value_date=cutoff.tz_convert("Europe/Paris") - pd.Timedelta(days=spec.lookback_days),
            to_value_date=cutoff.tz_convert("Europe/Paris"), revision_date=cutoff,
            nocache=True)
        values = fuel._normalise_market_series(state, series=spec.series)
        for delivery in append_days:
            selection_ceiling = min(fuel._civil_cutoff(delivery).tz_convert("UTC"), cutoff)
            selected = values.loc[values.index.tz_convert("UTC") <= selection_ceiling]
            require(len(selected), f"{spec.alias}: no quote known before {selection_ceiling}")
            raw.loc[delivery, spec.alias] = float(selected.iloc[-1])
            raw.loc[delivery, spec.alias + "__source_value_time_utc"] = selected.index[-1].tz_convert("UTC")
            raw.loc[delivery, "cutoff_time_utc"] = cutoff
        market_queries[spec.alias] = {"series": spec.series, "revision_utc": cutoff.isoformat(),
                                     "selection": "latest finite quote before min(logical origin, shared cutoff)"}
    derived = fuel._derive_daily_features(raw.sort_index(), ccgt_efficiency=0.58,
        ccgt_emission_tco2_mwh=0.36, ccgt_vom_eur_mwh=3.0)
    last = derived.loc[first]
    require(np.isfinite(last.loc[list(fuel.FEATURE_COLUMNS)].to_numpy(dtype=float)).all(),
            "New fuel state has nonfinite derived features")
    prior = indexed.loc[indexed.index < future_index[0]].copy()
    bridge_first = prior.index[-1] + pd.Timedelta(hours=1)
    blocks = []
    if bridge_first < future_index[0]:
        for delivery in append_days[append_days < first]:
            row = derived.loc[delivery]
            idx = fuel._physical_utc_index(delivery)
            block = pd.DataFrame(index=idx)
            block["value_time_utc"] = idx
            block["snapshot_time_utc"] = cutoff
            block["revision_time_utc"] = cutoff
            for column in (*fuel.SOURCE_TIME_COLUMNS, *fuel.FEATURE_COLUMNS):
                block[column] = row[column]
            blocks.append(block.loc[:, list(fuel.OUTPUT_COLUMNS)])
    current = pd.DataFrame(index=future_index.copy())
    current["value_time_utc"] = future_index
    current["snapshot_time_utc"] = cutoff
    current["revision_time_utc"] = cutoff
    for column in (*fuel.SOURCE_TIME_COLUMNS, *fuel.FEATURE_COLUMNS):
        current[column] = last[column]
    combined = pd.concat([prior, *blocks, current.loc[:, list(fuel.OUTPUT_COLUMNS)]])
    _hourly(combined, "Combined fuel")
    require(combined.index.is_unique, "Fuel bridge duplicates donor rows")
    return combined, {"market_queries": market_queries,
        "future_state_policy": "same known market state broadcast across both delivery days",
        "future_quotes_used": False, "market_prices_forecast": False,
        "bridge_current_fit_reconstruction": [str(d.date()) for d in append_days if d < first],
        "provider_publication_timestamp_verified": False}


def _thermal(client, index, cutoff):
    from marginal_cost_expert.sources import capacity_catalog, _daily_values
    from chronos2_hourly.nyx_thermal_capacity_features import SOURCES, TECHNOLOGIES, COLUMNS

    catalogue = capacity_catalog(list(ZONES))
    days = pd.DatetimeIndex(sorted(set(index.tz_convert("Europe/Paris").date)))
    source_values, audit = {}, {}
    local_days = index.tz_convert("Europe/Paris").tz_localize(None).normalize()
    for source in SOURCES:
        spec = catalogue[source + "_available_gw"]
        response = client.get(spec["series"], from_value_date=days[0] - pd.Timedelta(days=1),
            to_value_date=days[-1] + pd.Timedelta(days=1), revision_date=cutoff, nocache=True)
        values = _daily_values(response).reindex(days)
        numeric = values.to_numpy(dtype=float)
        require(not np.isinf(numeric).any() and not ((np.isfinite(numeric)) & (numeric < 0)).any(),
                f"{source}: negative or infinite thermal capacity")
        source_values[source] = values.reindex(local_days).to_numpy(dtype=float)
        audit[source] = {"series": spec["series"], "revision_utc": cutoff.isoformat(),
                         "available_days": int(values.notna().sum()), "days": len(days)}
    result = {}
    for zone in ZONES:
        data = {}
        for source in SOURCES:
            name = "thermal__" + source + "_pmax_gw"
            data[name] = source_values[source]
            data[name + "__available"] = np.isfinite(source_values[source]).astype(float)
        for technology in TECHNOLOGIES:
            name = "thermal__own_" + technology + "_pmax_gw"
            values = source_values.get(zone.lower() + "_" + technology, np.full(len(index), np.nan))
            data[name] = values
            data[name + "__available"] = np.isfinite(values).astype(float)
        result[zone] = pd.DataFrame(data, index=index.copy()).loc[:, list(COLUMNS)]
    return result, audit


def _public_captures(index, cutoff, columns, *, archive, verifier, group, delay_hours):
    output = _missing(index, columns)
    audit = {"group": group, "actual_pre_cutoff_capture_required": True,
             "adopted": [], "missing": [], "retrospective_live_fetch": False}
    for delivery in sorted(set(index.tz_convert("Europe/Paris").date)):
        day = delivery.isoformat()
        mask = index.tz_convert("Europe/Paris").date == delivery
        directory = archive / day
        if not (directory / "capture.json").is_file():
            audit["missing"].append({"day": day, "reason": "no immutable source capture"})
            continue
        receipt = json.loads((directory / "capture.json").read_text(encoding="utf-8"))
        retrieved = _utc(receipt.get("retrieved_at_utc"), f"{group} capture")
        if retrieved > cutoff:
            audit["missing"].append({"day": day, "reason": "capture after shared cutoff"})
            continue
        frame, _ = verifier(directory, day)
        require(tuple(frame.columns) == tuple(columns), f"{group}: source feature columns differ")
        expected = index[mask]
        require(frame.index.equals(expected), f"{group}: source delivery grid differs")
        frame = frame.copy()
        eligible = expected - pd.Timedelta(hours=48) + pd.Timedelta(hours=delay_hours) <= cutoff
        frame.loc[~eligible, :] = _missing(expected[~eligible], columns)
        if (~eligible).any():
            audit["missing"].append({"day": day, "reason": "source intervals after shared cutoff",
                                      "hours": int((~eligible).sum())})
        output.loc[expected, list(columns)] = frame.to_numpy()
        audit["adopted"].append({"day": day, "retrieved_at_utc": retrieved.isoformat(),
                                  "capture_sha256": _sha(directory / "capture.json"),
                                  "hours_rejected_at_shared_cutoff": int((~eligible).sum())})
    audit["available_hours"] = {c: int(output[c].sum()) for c in columns if c.endswith("__available")}
    return output, audit


def _jao(index, cutoff, root, columns):
    from chronos2_hourly.nyx_annual_jao_source import verify_daily_capture, build_strict_history_features

    output = _missing(index, columns)
    audit = {"adopted": [], "missing": [], "actual_pre_cutoff_capture_required": True,
             "retrospective_live_fetch": False, "branches_same_prospective_vintage": True}
    for day in sorted(set(index.tz_convert("Europe/Paris").date)):
        wanted = index[index.tz_convert("Europe/Paris").date == day]
        # Missing capture is a declared feature state; malformed existing
        # archives still raise rather than hiding corruption as missing.
        raw = root / "raw/initialComputation" / (day.isoformat() + ".json.gz")
        if not raw.exists():
            audit["missing"].append({"day": str(day), "reason": "no immutable initial-computation capture"})
            continue
        verified = verify_daily_capture(root, day)
        retrieved = _utc(verified["retrieved_at_utc"], "JAO capture")
        modified = _utc(verified["last_modified_utc"], "JAO watermark")
        if max(retrieved, modified) > cutoff:
            audit["missing"].append({"day": str(day), "reason": "capture or modification after shared cutoff"})
            continue
        frame, detail = build_strict_history_features(root, wanted)
        require(set(frame.columns) == set(columns), "JAO trained feature names differ")
        output.loc[wanted, list(columns)] = frame.loc[:, list(columns)].to_numpy()
        audit["adopted"].append({"day": str(day), **verified, "features": detail})
    audit["available_hours"] = int(output["extra_jao__available"].sum())
    return output, output.copy(deep=True), audit


def collect_horizon_sources(seed_bundle, first_day, cutoff, forecast_days=2,
                            client=None, now_utc=None, *, retained_covariates=None,
                            retained_covariates_evidence=None):
    """Return all fresh source blocks and untouched preceding donor history.

    Forecast origins exposed here are actual shared revision ceilings. Any
    logical replay origin needed by older feature helpers must be carried
    separately by the caller and must not be represented as a source vintage.
    """
    from chronos2_modular.saturn import create_saturn_client
    from run_nyx_annual_auction_prices_source import load_plan
    from run_nyx_annual_hydro_source import verify_capture as hydro_verify
    from run_nyx_annual_exchange_source import _verify_capture as exchange_verify
    from chronos2_hourly.nyx_annual_cpu_bundle_builder import JAO_COLUMNS
    from chronos2_hourly.nyx_annual_live_preflight import load_schema, ROOT as repository_root
    from chronos2_hourly.nyx_annual_feature_projection import POOLED
    from chronos2_hourly.nyx_lagged_exchange_features import COLUMNS as EXCHANGE_COLUMNS

    future = _grid(first_day, forecast_days)
    ceiling = _utc(cutoff, "Shared cutoff")
    now = pd.Timestamp.now(tz="UTC") if now_utc is None else _utc(now_utc, "Now")
    require(ceiling <= now, "Shared cutoff has not occurred")
    require(ceiling < future[0], "Shared cutoff must precede first forecast delivery")
    own_origin = pd.Timestamp(f"{_day(first_day) - timedelta(days=1)} 08:00", tz="Europe/Paris")
    require(ceiling <= own_origin.tz_convert("UTC"),
            "Shared cutoff exceeds first day-ahead 08:00 contract")
    bundle = Path(seed_bundle).resolve()
    cov, cov_bound = _bound_frame(bundle, "saturn", "source_artifacts/saturn/covariates.parquet")
    donor_fuel, fuel_bound = _bound_frame(bundle, "fuel", "source_artifacts/fuel/market_fuel_features.parquet")
    repository_root = Path(repository_root).resolve()
    plan = load_plan(repository_root)
    if client is None:
        client = create_saturn_client(plan["saturn_url"], os.getenv("SATURN_AUTHOR") or plan["saturn_author"])
        if hasattr(client, "session"):
            from functools import partial
            client.session.request = partial(client.session.request, timeout=60.0)
    covariates, refreshed, profiles_audit, rolling_cache_audit = _profiles(client, cov, future, ceiling,
        retained_covariates=retained_covariates,
        retained_covariates_evidence=retained_covariates_evidence)
    prices, price_audit = _prices(client, bundle, first_day, ceiling)
    fuel, fuel_audit = _fuel(client, donor_fuel, first_day, future, ceiling)
    thermal, thermal_audit = _thermal(client, future, ceiling)
    schema = load_schema()
    hydro_columns = tuple(c for c in schema["families"][POOLED]["columns"]["FR"] if c.startswith("extra_hydro_"))
    hydro, hydro_audit = _public_captures(future, ceiling, hydro_columns,
        archive=repository_root / "data/pit/nyx_annual_hydro_captures", verifier=hydro_verify,
        group="public_hydro", delay_hours=1)
    exchange, exchange_audit = _public_captures(future, ceiling, EXCHANGE_COLUMNS,
        archive=repository_root / "data/pit/nyx_annual_exchange_captures", verifier=exchange_verify,
        group="lagged_exchange", delay_hours=2)
    jao_root = repository_root / "data/pit/nyx_annual_jao_initial_live"
    original, refreshed_jao, jao_audit = _jao(future, ceiling, jao_root, JAO_COLUMNS)
    actual_revisions = pd.DataFrame(ceiling, index=refreshed, columns=covariates.columns)
    audit = {"protocol": PROTOCOL, "shared_cutoff_utc": ceiling.isoformat(),
        "forecast_first_day": first_day, "forecast_days": forecast_days,
        "forecast_hours": len(future), "generated_at_utc": now.isoformat(),
        "future_labels_used": False, "model_fits": 0, "donor_modified": False,
        "donor_sources": {"covariates": cov_bound, "fuel": fuel_bound},
        "profiles": profiles_audit, "prices": price_audit, "fuel": fuel_audit,
        "rolling_covariate_cache": rolling_cache_audit,
        "thermal": thermal_audit, "public_hydro": hydro_audit,
        "lagged_exchange": exchange_audit, "jao_initial": jao_audit,
        "bridge_profile_current_fit_reconstruction": sorted({str(d) for d in refreshed.tz_convert("Europe/Paris").date if d < _day(first_day)}),
        "provider_publication_timestamp_verified": False,
        "missing_groups": [name for name, item in (("public_hydro", hydro_audit),
            ("lagged_exchange", exchange_audit), ("jao_initial", jao_audit)) if item["missing"]]}
    return {"future_index": future, "refresh_index": refreshed,
        "covariates": covariates, "prices": prices, "fuel": fuel,
        "thermal_features": thermal, "hydro_features": hydro,
        "exchange_features": exchange, "jao_original": original,
        "jao_refreshed": refreshed_jao, "actual_profile_revisions": actual_revisions,
        "logical_origins": pd.Series([min(pd.Timestamp(f"{day - timedelta(days=1)} 08:00", tz="Europe/Paris").tz_convert("UTC"), ceiling) for day in covariates.index.tz_convert("Europe/Paris").date], index=covariates.index, name="forecast_origin_utc"),
        "forecast_origins": {z: pd.Series(ceiling, index=future, name="forecast_origin_utc") for z in ZONES},
        "audit": audit}
