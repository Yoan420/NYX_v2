"""Any-hour source snapshots retain real retrieval and independently rebuilt RAW."""
from datetime import date, timedelta
import json

import pandas as pd
import pytest

import run_nyx_annual_daily_capture as daily
import run_nyx_annual_exchange_source as exchange
import run_nyx_annual_hydro_source as hydro
from chronos2_hourly import jao_flowbased as jao
from chronos2_hourly import nyx_annual_jao_history as jao_history
from chronos2_hourly import nyx_annual_jao_source as jao_source
from chronos2_hourly import nyx_annual_public_history as public_history
from chronos2_hourly.nyx_annual_live_preflight import delivery_grid, validate_source_receipt
from chronos2_hourly.nyx_annual_operational_context import context_scope, make_context
from test_nyx_annual_jao_source import _fetch, FakeClient
from test_nyx_annual_public_history import Session


DAY = "2026-09-30"


def at(hour):
    return pd.Timestamp(f"2026-09-29 {hour}", tz="Europe/Paris").tz_convert("UTC")


@pytest.mark.parametrize("hour", ["00:05", "07:30", "14:00", "23:30"])
def test_target_day_accepts_any_hour_only_in_explicit_context(hour):
    now = at(hour)
    with context_scope(make_context(DAY, now)):
        assert daily.target_day(now) == date.fromisoformat(DAY)
    if hour in ("00:05", "14:00", "23:30"):
        with pytest.raises(ValueError, match="01:15"):
            daily.target_day(now)


@pytest.mark.parametrize("module,verify", [(exchange, exchange._verify_capture),
                                           (hydro, hydro.verify_capture)])
@pytest.mark.parametrize("hour", ["00:05", "23:30"])
def test_any_hour_capture_preserves_actual_time_reuses_and_rejects_strict(tmp_path, module, verify, hour):
    launch = at(hour)
    actual = launch + pd.Timedelta(seconds=5)
    context = make_context(DAY, launch)
    with context_scope(context):
        session = Session(actual)
        path = module.capture(DAY, tmp_path, session=session, now_utc=actual)
        frame, receipt = verify(path.parent, DAY)
        assert receipt["retrieved_at_utc"] == actual.isoformat()
        assert receipt["operational_snapshot"] == context
        assert receipt["asof_cutoff_verified"] is False
        assert receipt["delivery_snapshot_pre_cutoff_verified"] is False
        if module is exchange and hour == "00:05":
            assert receipt["source_end_plus_nominal_hour_before_cutoff"] is False
        assert not frame.empty
        before = path.read_bytes()
        unused = Session(actual + pd.Timedelta(minutes=1))
        module.capture(DAY, tmp_path, session=unused, now_utc=actual + pd.Timedelta(minutes=1))
        assert unused.calls == [] and path.read_bytes() == before
    with pytest.raises(ValueError, match="context differs"):
        verify(path.parent, DAY)
    with context_scope(make_context(DAY, launch + pd.Timedelta(minutes=1))):
        with pytest.raises(ValueError, match="context differs"):
            verify(path.parent, DAY)


def test_operational_missing_quarters_keep_native_nan_flags(tmp_path):
    launch = at("00:05")

    class PartialSession(Session):
        def get(self, *args, **kwargs):
            result = super().get(*args, **kwargs)
            obj = json.loads(result.content)
            obj["data"].pop(1)
            result.content = json.dumps(obj).encode()
            return result

    with context_scope(make_context(DAY, launch)):
        path = exchange.capture(DAY, tmp_path, session=PartialSession(launch), now_utc=launch)
        frame, _ = exchange._verify_capture(path.parent, DAY)
        level = exchange.exchange.LEVELS[0]
        assert pd.isna(frame[level].iloc[0])
        assert frame[level + "__available"].iloc[0] == 0
        (path.parent / "DE.json").write_text("{}", encoding="utf-8")
        with pytest.raises(ValueError, match="bytes changed"):
            exchange._verify_capture(path.parent, DAY)


def test_collection_keeps_launch_target_when_run_crosses_midnight():
    with context_scope(make_context(DAY, at("23:30"))):
        after_midnight = at("23:30") + pd.Timedelta(hours=1)
        assert daily.target_day(after_midnight, date.fromisoformat(DAY)) == date.fromisoformat(DAY)


def test_operational_response_rejects_observation_intervals_from_the_future():
    actual = at("00:00")
    first, stop = actual - pd.Timedelta(hours=1), actual + pd.Timedelta(hours=1)
    result = Session(actual).get(exchange.URLS["FR"],
        params=exchange._params("FR", first, stop), timeout=(10, 45), allow_redirects=False)
    with pytest.raises(ValueError, match="ends after actual retrieval"):
        exchange._response(result.content, "FR", first, stop, actual, actual, operational=True)


@pytest.mark.parametrize("module,verify", [(exchange, exchange._verify_capture),
                                           (hydro, hydro.verify_capture)])
def test_autumn_clock_change_midnight_caps_request_and_retains_unavailable_tail(tmp_path, module, verify):
    day = "2025-10-27"
    actual = pd.Timestamp("2025-10-26 00:05", tz="Europe/Paris").tz_convert("UTC")
    with context_scope(make_context(day, actual)):
        _, _, expected_stop, _ = module._source_window(day) if module is exchange else module.source_window(day)
        assert expected_stop > actual
        session = Session(actual)
        path = module.capture(day, tmp_path, session=session, now_utc=actual)
        frame, receipt = verify(path.parent, day)
        assert frame.filter(like="__available").iloc[-1].eq(0).all()
        for _, params in session.calls:
            assert pd.Timestamp(params["end"]) < actual.floor("h")
        for window in receipt["request_window_utc"].values():
            assert pd.Timestamp(window[1]) == actual.floor("h")


@pytest.mark.parametrize("empty,hour", [(True, "00:05"), (False, "23:30")])
def test_jao_real_empty_or_late_delivery_is_archived_without_fallback(tmp_path, empty, hour):
    launch = at(hour)
    actual = launch + pd.Timedelta(seconds=5)
    result = _fetch(date.fromisoformat(DAY), retrieved=actual,
                    last_modified=actual - pd.Timedelta(seconds=1))
    if empty:
        result = jao.JaoFetchResult(**{**result.__dict__, "rows": (), "total_rows": 0,
                                      "last_modified_utc": None, "page_last_modified_utc": ()})
    context = make_context(DAY, launch)
    with context_scope(context):
        item = jao_source.capture_day(day=date.fromisoformat(DAY), cache_root=tmp_path,
            client=FakeClient(result), now_utc=actual)
        assert item["retrieved_at_utc"] == actual.isoformat()
        assert item["rows"] == (0 if empty else 24)
        audit_path = jao_source._paths(tmp_path, date.fromisoformat(DAY))[1]
        audit = json.loads(audit_path.read_text())
        assert audit["operational_snapshot"] == context
        assert audit["asof_cutoff_verified"] is False
        assert audit.get("fallback_reason") is None
        full, current, _ = delivery_grid(DAY)
        frame, _, _ = jao_history._features({date.fromisoformat(DAY): tmp_path}, current,
                                           date.fromisoformat(DAY))
        if empty:
            assert frame[list(jao_source.VALUE_COLUMNS)].isna().all().all()
            assert frame[jao_source.AVAILABLE].eq(0).all()
        else:
            assert frame[jao_source.AVAILABLE].eq(1).all()
    with pytest.raises(ValueError, match="context differs"):
        jao_source.verify_daily_capture(tmp_path, date.fromisoformat(DAY))


@pytest.mark.parametrize("group", public_history.GROUPS)
def test_operational_public_history_reconstructs_and_cannot_qualify(tmp_path, monkeypatch, group):
    launch, actual = at("23:30"), at("23:31")
    context = make_context(DAY, launch)
    monkeypatch.setattr(public_history, "MIN_REQUEST_INTERVAL_SECONDS", 0)
    with context_scope(context):
        _, current, cutoff = delivery_grid(DAY)
        training = pd.date_range(current[0] - pd.Timedelta(days=2), current[0], freq="h", inclusive="left")
        monkeypatch.setattr(public_history, "delivery_grid", lambda day: (training.append(current), current, cutoff))
        collector = hydro if group == "public_hydro" else exchange
        collector.capture(DAY, tmp_path / "capture", session=Session(actual), now_utc=actual)
        path = public_history.publish_history(group, DAY, tmp_path / "bundle",
            archive=tmp_path / "capture", history_archive=tmp_path / "history",
            session=Session(actual), now_utc=actual)
        receipt = json.loads(path.read_text())
        facts = public_history.verify_bundle_history(tmp_path / "bundle", DAY, receipt)
        assert facts["asof_state_utc"] == actual.isoformat()
        assert facts["asof_cutoff_verified"] is False
        assert facts["delivery_snapshot_pre_cutoff_verified"] is False
        validate_source_receipt(receipt, group=group, day=DAY, bundle=tmp_path / "bundle",
                                cutoff=cutoff, allow_training_bootstrap=True)
        with pytest.raises(ValueError):
            validate_source_receipt(receipt, group=group, day=DAY, bundle=tmp_path / "bundle", cutoff=cutoff)
        relative = f"source_artifacts/{group}/features.parquet"
        frame = pd.read_parquet(tmp_path / "bundle" / relative)
        frame.iloc[0, 0] += 1
        frame.to_parquet(tmp_path / "bundle" / relative)
        receipt["artifact_sha256"][relative] = public_history.sha256(tmp_path / "bundle" / relative)
        with pytest.raises(AssertionError):
            public_history.verify_bundle_history(tmp_path / "bundle", DAY, receipt)


def test_operational_empty_jao_delivery_bootstrap_reconstructs_portable_raw(tmp_path, monkeypatch):
    launch, actual = at("00:05"), at("00:06")
    target = date.fromisoformat(DAY)
    with context_scope(make_context(DAY, launch)):
        _, current, cutoff = delivery_grid(DAY)
        training = pd.date_range(current[0] - pd.Timedelta(days=2), current[0], freq="h", inclusive="left")
        monkeypatch.setattr(jao_source, "delivery_grid", lambda day: (training.append(current), current, cutoff))
        empty = _fetch(target, retrieved=actual)
        empty = jao.JaoFetchResult(**{**empty.__dict__, "rows": (), "total_rows": 0,
                                     "last_modified_utc": None, "page_last_modified_utc": ()})
        jao_source.capture_day(day=target, cache_root=tmp_path / "capture",
                              client=FakeClient(empty), now_utc=actual)

        class HistoryClient:
            def fetch_initial_day(self, day):
                return _fetch(day, retrieved=actual)

        bundle = tmp_path / "bundle"
        jao_history.publish_history(DAY, bundle, live_cache_root=tmp_path / "capture",
            history_cache_root=tmp_path / "history", legacy_cache_root=tmp_path / "legacy",
            client=HistoryClient())
        receipt = json.loads((bundle / "source_receipts/jao_initial.json").read_text())
        verified = jao_history.verify_bundle_history(bundle, DAY, receipt)
        assert verified["daily_raw_captures_recomputed"] == 3
        assert verified["asof_cutoff_verified"] is False
        assert verified["delivery_snapshot_pre_cutoff_verified"] is False
        frame = pd.read_parquet(bundle / jao_source.SOURCE_SUBDIR / jao_source.FEATURE_NAME)
        assert frame.loc[current, list(jao_source.VALUE_COLUMNS)].isna().all().all()
        assert frame.loc[current, jao_source.AVAILABLE].eq(0).all()
        assert frame.loc[training, jao_source.AVAILABLE].eq(1).all()
        validate_source_receipt(receipt, group="jao_initial", day=DAY, bundle=bundle,
                                cutoff=cutoff, allow_training_bootstrap=True)
        with pytest.raises(ValueError):
            validate_source_receipt(receipt, group="jao_initial", day=DAY, bundle=bundle, cutoff=cutoff)
