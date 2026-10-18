"""Private sources use a real early revision; historical origins stay fixed."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly import nyx_annual_operational_context as context
from chronos2_hourly import nyx_annual_saturn_source as saturn
from chronos2_hourly import nyx_annual_saturn_archive as archive
from chronos2_hourly import nyx_annual_wind_source as wind
from chronos2_hourly import nyx_local_extra_features as extra
from chronos2_hourly import nyx_local_price_features as prices
import materialize_saturn_kalman_fuel as fuel
import run_nyx_annual_auction_prices_source as auction
import run_nyx_annual_fuel_source as fuel_source
import run_nyx_annual_thermal_source as thermal
from marginal_cost_expert import sources as capacities


DAY = "2026-09-29"
EARLY = pd.Timestamp("2026-09-28T04:45:00Z")  # 06:45 Paris.


@pytest.fixture
def operational(monkeypatch):
    monkeypatch.setattr(context, "_now", lambda: EARLY)
    snapshot = context.make_context(DAY, EARLY)
    monkeypatch.setenv(context.ENV, json.dumps(snapshot))
    return snapshot


def test_private_cutoff_helpers_change_only_delivery_day(operational):
    for day in (DAY, "2026-09-28"):
        target = EARLY if day == DAY else pd.Timestamp("2026-09-27T06:00:00Z")
        stamp = pd.Timestamp(day)
        assert saturn.cutoff(day) == target
        assert archive._cutoff(stamp.date()) == target
        assert wind._origin(day) == target
        assert fuel._civil_cutoff(stamp).tz_convert("UTC") == target
        assert capacities.civil_cutoff(stamp) == target
        grid = saturn.grid(day)
        assert extra.cutoff_times(grid, "Europe/Paris").eq(target).all()
        assert prices._cutoffs(grid, "Europe/Paris").eq(target).all()
        assert fuel._cutoffs_for_value_times(grid).equals(pd.DatetimeIndex([target] * len(grid)))


def test_early_saturn_profile_and_target_queries_bind_actual_revision(operational, monkeypatch, tmp_path):
    calls = []

    def fetch(client, name, start, stop, timezone, **kwargs):
        calls.append(kwargs["revision_date"])
        index = pd.date_range(start, stop, freq="h", tz="UTC")
        return pd.Series(np.arange(len(index), dtype=float), index=index)

    monkeypatch.setattr(saturn, "fetch_saturn_series_from_client", fetch)
    receipt = saturn.capture_day(object(), DAY, tmp_path / "profile", now_utc=EARLY)
    assert receipt["forecast_origin_utc"] == EARLY.isoformat()
    assert all(stamp == EARLY for stamp in calls)
    calls.clear()
    target = saturn.capture_target_snapshot(object(), DAY, "2025-09-20", tmp_path / "targets", now_utc=EARLY)
    assert target["target_revision_utc"] == EARLY.isoformat()
    assert target["target_future_labels_used"] is False
    assert calls == [EARLY] * 4
    frame, _ = saturn.verify_target_snapshot(tmp_path / "targets/targets_current_fit_v1" / DAY,
                                            DAY, "2025-09-20")
    assert frame.index[-1] < saturn.grid(DAY)[0]


def test_early_auction_queries_actual_revision_and_excludes_future_prices(operational, monkeypatch, tmp_path):
    expected, _ = auction.price_grid(DAY)
    calls = []

    def fetch(client, name, start, stop, timezone, **kwargs):
        calls.append(kwargs["revision_date"])
        # Even a provider returning an extra future hour must not admit it.
        index = expected.append(pd.DatetimeIndex([saturn.grid(DAY)[0]]))
        return pd.Series(17.0, index=index)

    monkeypatch.setattr(auction, "fetch_saturn_series_from_client", fetch)
    values, evidence = auction.collect(object(), DAY, now_utc=EARLY)
    assert calls == [EARLY] * 4
    assert evidence["cutoff_utc"] == EARLY.isoformat()
    assert all(frame.index.equals(expected) for frame in values.values())
    receipt_path = auction.publish(tmp_path / "bundle", DAY, values, evidence,
                                   auction.load_plan(), now_utc=EARLY)
    receipt = json.loads(receipt_path.read_text())
    assert receipt["asof_state_utc"] == EARLY.isoformat()
    assert receipt["availability_basis"] == "Saturn revision_date query at recorded operational cutoff_utc"


def test_early_fuel_rejects_quote_after_real_snapshot(operational):
    calls = []

    class Client:
        def get(self, name, **kwargs):
            calls.append(kwargs)
            return pd.Series([31.0, 999.0], index=pd.DatetimeIndex([
                EARLY - pd.Timedelta(hours=12), EARLY + pd.Timedelta(minutes=30)]))

    value, source_time, cutoff = fuel._query_market_value(Client(), fuel.MARKET_SERIES[0], pd.Timestamp(DAY))
    assert value == 31.0 and source_time < EARLY and cutoff == EARLY
    assert calls[0]["revision_date"] == EARLY
    assert calls[0]["to_value_date"] == EARLY


def test_early_fuel_sidecar_and_receipt_show_real_revision(operational, tmp_path):
    start, end = pd.Timestamp("2025-09-29"), pd.Timestamp(DAY)
    first = start - pd.Timedelta(days=fuel.WARMUP_DAYS)
    days = pd.date_range(first, end, freq="D")
    cutoff = pd.DatetimeIndex([fuel._civil_cutoff(day).tz_convert("UTC") for day in days])
    raw = pd.DataFrame({
        "ttf_m1_eur_mwh_th": 35.0 + np.arange(len(days)) / 100.0,
        "eua_first_dec_eur_tco2": 70.0 + np.arange(len(days)) / 200.0,
        "ttf_m1_eur_mwh_th__source_value_time_utc": cutoff - pd.Timedelta(hours=12),
        "eua_first_dec_eur_tco2__source_value_time_utc": cutoff - pd.Timedelta(hours=12),
        "cutoff_time_utc": cutoff,
    }, index=days)
    daily = fuel._derive_daily_features(raw, **fuel_source.ASSUMPTIONS)
    frame = fuel._broadcast_daily_features(daily, start_day=start, end_day=end)
    audit = fuel._audit_payload(frame, start_day=start, end_day=end, requested_start=start,
                               requested_end=end, query_start=first, **fuel_source.ASSUMPTIONS)
    assert audit["cutoff_time"] == "08:00"  # Historical nominal recipe.
    assert audit["operational_asof_utc"] == EARLY.isoformat()
    assert audit["operational_delivery_day"] == DAY
    on_day = pd.DatetimeIndex(frame.value_time_utc).isin(saturn.grid(DAY))
    assert frame.loc[on_day, "snapshot_time_utc"].eq(EARLY).all()
    cache = tmp_path / "cache"
    fuel._write_bundle(frame, cache / fuel.OUTPUT_NAME, cache / (fuel.OUTPUT_NAME + fuel.AUDIT_SUFFIX), audit)
    evidence = fuel_source.verify_fuel_artifact(cache, delivery_day=DAY,
                                               history_start_day=start.date().isoformat(), now_utc=EARLY)
    path = fuel_source.publish_fuel_receipt(tmp_path / "bundle", delivery_day=DAY, evidence=evidence)
    receipt = json.loads(path.read_text())
    assert receipt["asof_state_utc"] == EARLY.isoformat()
    assert receipt["availability_basis"] == "Saturn revision_date query at recorded operational cutoff_utc"
    assert "D-1 08:00" not in receipt["publication_vintage_limit"]


def test_early_price_features_keep_recipe_values_with_real_availability(operational, monkeypatch):
    grid = pd.date_range("2026-09-15", DAY, tz="Europe/Paris", freq="h", inclusive="left").tz_convert("UTC")
    history = pd.Series(np.arange(len(grid), dtype=float), index=grid)
    current = saturn.grid(DAY)
    with monkeypatch.context() as strict:
        strict.delenv(context.ENV)
        expected, _ = prices.build_price_features({"FR": history}, zone="FR", delivery_index=current)
    actual, audit = prices.build_price_features({"FR": history}, zone="FR", delivery_index=current)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    assert audit["operational_price_snapshot_asof_utc"] == EARLY.isoformat()
    assert audit["daily_sources"][0]["forecast_cutoff_utc"] == EARLY.isoformat()
    assert audit["daily_sources"][0]["features"]["price_fr_d1_hour"]["latest_available_by_utc"] == EARLY.isoformat()


def test_early_thermal_never_requests_future_staircase_revision(operational, tmp_path):
    plan = thermal.plan()
    blocks, revisions = [], []

    class Client:
        def block_staircase(self, name, *, from_value_date, to_value_date, **kwargs):
            assert kwargs == thermal.REQUEST
            blocks.append(to_value_date)
            assert to_value_date < pd.Timestamp(DAY)
            days = pd.date_range(from_value_date, to_value_date, freq="D")
            return pd.Series(4.0, index=days)

        def get(self, name, *, from_value_date, to_value_date, revision_date):
            day = from_value_date + pd.Timedelta(days=1)
            revisions.append((day, revision_date))
            expected = EARLY if day == pd.Timestamp(DAY) else capacities.civil_cutoff(day)
            assert revision_date == expected and revision_date <= EARLY
            return pd.Series([9.0 if day == pd.Timestamp(DAY) else 4.0], index=pd.DatetimeIndex([day]))

    daily, evidence = thermal.collect(Client(), DAY, now_utc=EARLY, cache=tmp_path / "cache")
    assert len(blocks) == len(thermal.SOURCES)
    assert all(day == pd.Timestamp("2026-09-28") for day in blocks)
    assert len(revisions) == len(thermal.SOURCES) * 366
    assert all(series.iloc[-1] == 9.0 for series in daily.values())
    assert evidence["delivery_uses_independent_operational_state"] is True
    receipt_path = thermal.publish(tmp_path / "bundle", DAY, daily, evidence, plan, now_utc=EARLY)
    receipt = json.loads(receipt_path.read_text())
    assert receipt["asof_state_utc"] == EARLY.isoformat()
    for name in thermal.SOURCES:
        hourly = pd.read_parquet(tmp_path / "bundle/source_artifacts/thermal_capacity" / f"{name}.parquet")
        assert hourly.loc[saturn.grid(DAY), "asof_query_utc"].eq(EARLY).all()
        assert hourly.loc[saturn.grid(DAY), "pmax_gw"].eq(9.0).all()


def test_default_cutoff_still_refuses_early_private_queries(monkeypatch):
    monkeypatch.delenv(context.ENV, raising=False)
    assert saturn.cutoff(DAY) == pd.Timestamp("2026-09-28T06:00:00Z")
    with pytest.raises(ValueError, match="has not occurred"):
        auction.collect(object(), DAY, now_utc=EARLY)
    with pytest.raises(ValueError, match="has not occurred"):
        thermal.collect(object(), DAY, now_utc=EARLY)


@pytest.mark.parametrize("launch", ["2026-10-24T22:05:00Z", "2026-10-25T20:37:00Z"])
def test_complete_feature_assembly_at_midnight_and_late_on_dst_day(monkeypatch, launch):
    """Exercise all twelve matrices and four upstream assemblies at real origins."""
    from test_nyx_annual_cpu_bundle_builder import inputs
    from chronos2_hourly import nyx_annual_cpu_bundle_builder as builder
    from chronos2_hourly import nyx_annual_cpu_baseline as baseline

    now = pd.Timestamp(launch)
    monkeypatch.setattr(context, "_now", lambda: now)
    snapshot = context.make_context("2026-10-26", now)
    monkeypatch.setenv(context.ENV, json.dumps(snapshot))
    day, sources = inputs()
    # The live private producer supplies normalized pmax/as-of frames. The
    # fixture's legacy research vintage format intentionally retains its
    # strict 08:00 contract and is not the operational source path.
    full = builder.gate.delivery_grid(day)[0]
    source_origins = extra.cutoff_times(full, "Europe/Paris")
    sources["thermal_sources"] = {name: pd.DataFrame({"pmax_gw": 1.0,
        "asof_query_utc": source_origins}, index=full) for name in builder.thermal.SOURCES}
    current = sources["reference_index"].tz_convert("Europe/Paris").date == pd.Timestamp(day).date()
    expected_origin = pd.Timestamp(snapshot["asof_utc"])
    assert sources["forecast_origins"]["FR"].loc[current].eq(expected_origin).all()
    plan = baseline.build_delivery_plan(day)
    assert plan.forecast_origin_utc == expected_origin
    assert plan.horizon == 24 and len(plan.delivery_index_utc) == 24
    previous_plan = baseline.build_delivery_plan("2026-10-25")
    assert previous_plan.forecast_origin_utc == pd.Timestamp("2026-10-24T06:00:00Z")
    assert previous_plan.horizon == 25
    result = builder.build_feature_matrices(day, **sources)
    for zone in builder.gate.ZONES:
        assert result.base292[zone].shape[1] == 292
        assert result.augmented334[zone].shape[1] == 334
        assert result.test2_features[zone].shape[1] == 48
        for family, width in builder.gate.FAMILIES.items():
            frame = result.features[family][zone]
            assert frame.shape == (73, width)
            builder.gate.validate_feature_frame(frame, list(frame), sources["feature_index"], family)
        price_audit = result.audits["countries"][zone]["price"]
        current_audit = next(row for row in price_audit["daily_sources"] if row["delivery_day"] == day)
        assert current_audit["forecast_cutoff_utc"] == expected_origin.isoformat()
