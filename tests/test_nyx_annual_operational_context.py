"""Operational launch origins stay explicit, immutable, and unqualified."""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly import nyx_annual_live_preflight as gate
from chronos2_hourly import nyx_annual_operational_context as context


@pytest.fixture(autouse=True)
def clean_context(monkeypatch):
    monkeypatch.delenv(context.ENV, raising=False)


@pytest.mark.parametrize("local,day,hour", [
    ("2026-09-28 00:05+02:00", "2026-09-29", "2026-09-27T22:05:00+00:00"),
    ("2026-09-28 07:59+02:00", "2026-09-29", "2026-09-28T05:59:00+00:00"),
    ("2026-09-28 08:01+02:00", "2026-09-29", "2026-09-28T06:00:00+00:00"),
    ("2026-09-28 14:00+02:00", "2026-09-29", "2026-09-28T06:00:00+00:00"),
    ("2026-09-28 23:59+02:00", "2026-09-29", "2026-09-28T06:00:00+00:00"),
])
def test_early_and_late_launches_cap_the_logical_origin(local, day, hour):
    snapshot = context.make_context(day, now_utc=pd.Timestamp(local))
    assert snapshot["asof_utc"] == hour
    assert snapshot["requested_at_utc"] == pd.Timestamp(local).tz_convert("UTC").isoformat()
    with context.context_scope(snapshot):
        assert gate.delivery_grid(day)[2] == pd.Timestamp(hour)
        assert context.metadata(day) == {"operational_snapshot": snapshot}
        assert context.is_operational(day)


@pytest.mark.parametrize("day,local,hours,nominal", [
    ("2026-03-29", "2026-03-28 23:59+01:00", 23, "2026-03-28T07:00:00+00:00"),
    ("2026-10-25", "2026-10-24 23:59+02:00", 25, "2026-10-24T06:00:00+00:00"),
])
def test_snapshot_preserves_physical_dst_grid(monkeypatch, day, local, hours, nominal):
    now = pd.Timestamp(local).tz_convert("UTC")
    monkeypatch.setattr(context, "_now", lambda: now)
    snapshot = context.make_context(day)
    with context.context_scope(snapshot):
        _, current, origin = gate.delivery_grid(day)
        assert len(current) == hours
        assert str(current.tz) == "UTC" and current.is_unique
        assert origin.isoformat() == nominal


def test_context_scope_restores_outer_environment_after_failure():
    outer = context.make_context("2026-09-29", "2026-09-28T10:00:00+00:00")
    inner = context.make_context("2026-09-30", "2026-09-29T03:00:00+00:00")
    with context.context_scope(outer):
        encoded = os.environ[context.ENV]
        with pytest.raises(RuntimeError):
            with context.context_scope(inner):
                assert context.operational_context() == inner
                raise RuntimeError("interrupted")
        assert os.environ[context.ENV] == encoded
    assert context.ENV not in os.environ


def test_only_target_day_uses_effective_origin_and_continuation_keeps_launch_time():
    snapshot = context.make_context("2026-09-29", "2026-09-28T01:00:00+00:00")
    strict = gate.delivery_grid("2026-09-28")[2]
    with context.context_scope(snapshot):
        assert context.operational_context() == snapshot
        assert context.operational_context("2026-09-28") is None
        assert not context.is_operational("2026-09-28")
        assert context.metadata("2026-09-28") == {}
        assert gate.delivery_grid("2026-09-28")[2] == strict


def test_make_context_requires_tomorrow_in_paris():
    # UTC is still September 27, while the user's Paris date is September 28.
    context.make_context("2026-09-29", "2026-09-27T22:05:00+00:00")
    with pytest.raises(ValueError, match="tomorrow"):
        context.make_context("2026-09-28", "2026-09-27T22:05:00+00:00")


@pytest.mark.parametrize("change", [
    {"asof_utc": "2026-09-28T06:00:00+00:00"},
    {"nominal_cutoff_utc": "2026-09-28T07:00:00+00:00"},
    {"requested_at_utc": "2026-09-28T01:00:00"},
    {"delivery_day": "2026-09-30"},
    {"extra": "unapproved"},
])
def test_invalid_or_future_origin_declarations_are_rejected(monkeypatch, change):
    snapshot = context.make_context("2026-09-29", "2026-09-28T01:00:00+00:00")
    snapshot.update(change)
    monkeypatch.setenv(context.ENV, json.dumps(snapshot))
    with pytest.raises(ValueError, match="Operational snapshot"):
        context.operational_context()


def test_environment_rejects_future_requests(monkeypatch):
    launch = pd.Timestamp("2026-09-28T01:00:00+00:00")
    snapshot = context.make_context("2026-09-29", launch)
    monkeypatch.setattr(context, "_now", lambda: launch - pd.Timedelta(seconds=1))
    monkeypatch.setenv(context.ENV, json.dumps(snapshot))
    with pytest.raises(ValueError, match="future"):
        context.operational_context()


def test_invalid_environment_does_not_silently_return_to_strict_mode(monkeypatch):
    monkeypatch.setenv(context.ENV, "not JSON")
    with pytest.raises(ValueError, match="environment context JSON"):
        gate.delivery_grid("2026-09-29")


def _source_receipt(tmp_path, snapshot):
    artifact = tmp_path / "raw.bin"
    artifact.write_bytes(b"actual operational public response")
    cutoff = pd.Timestamp(snapshot["asof_utc"])
    delivery_retrieved = pd.Timestamp(snapshot["requested_at_utc"]) + pd.Timedelta(minutes=1)
    return {"protocol": gate.SOURCE_PROTOCOL, "source_group": "public_hydro",
            "delivery_day": snapshot["delivery_day"], "state": "COMPLETE",
            "history_policy": gate.TRAINING_HISTORY_POLICY,
            "training_snapshot_max_retrieved_at_utc": delivery_retrieved.isoformat(),
            "delivery_snapshot_pre_cutoff_verified": bool(delivery_retrieved <= cutoff),
            "origin_snapshot_capture_verified": False,
            "asof_cutoff_verified": False, "training_window_complete": True,
            "asof_state_utc": delivery_retrieved.isoformat(),
            "artifact_sha256": {"raw.bin": gate.sha256(artifact)},
            "operational_snapshot": snapshot}


@pytest.mark.parametrize("launch", ["2026-09-28T01:00:00+00:00", "2026-09-28T12:00:00+00:00"])
def test_late_response_is_accepted_only_for_exact_operational_preparation(tmp_path, launch):
    day = "2026-09-29"
    snapshot = context.make_context(day, launch)
    with context.context_scope(snapshot):
        receipt = _source_receipt(tmp_path, snapshot)
        cutoff = gate.delivery_grid(day)[2]
        gate.validate_source_receipt(receipt, group="public_hydro", day=day,
            bundle=tmp_path, cutoff=cutoff, allow_training_bootstrap=True)
        with pytest.raises(ValueError, match="exact preparation context"):
            gate.validate_source_receipt(receipt, group="public_hydro", day=day,
                bundle=tmp_path, cutoff=cutoff)
        receipt["asof_cutoff_verified"] = True
        with pytest.raises(ValueError, match="training history/cutoff declaration"):
            gate.validate_source_receipt(receipt, group="public_hydro", day=day,
                bundle=tmp_path, cutoff=cutoff, allow_training_bootstrap=True)
    receipt["asof_cutoff_verified"] = False
    with pytest.raises(ValueError, match="exact preparation context"):
        gate.validate_source_receipt(receipt, group="public_hydro", day=day,
            bundle=tmp_path, cutoff=gate.delivery_grid(day)[2], allow_training_bootstrap=True)


def test_mismatched_context_marker_and_future_retrieval_are_rejected(tmp_path):
    day = "2026-09-29"
    snapshot = context.make_context(day, "2026-09-28T01:00:00+00:00")
    with context.context_scope(snapshot):
        receipt = _source_receipt(tmp_path, snapshot)
        receipt["operational_snapshot"] = {**snapshot, "requested_at_utc": "2026-09-28T02:00:00+00:00"}
        with pytest.raises(ValueError, match="exact preparation context"):
            gate.validate_source_receipt(receipt, group="public_hydro", day=day,
                bundle=tmp_path, cutoff=gate.delivery_grid(day)[2], allow_training_bootstrap=True)
        receipt["operational_snapshot"] = snapshot
        receipt["training_snapshot_max_retrieved_at_utc"] = (pd.Timestamp.now(tz="UTC") + pd.Timedelta(hours=1)).isoformat()
        with pytest.raises(ValueError, match="future"):
            gate.validate_source_receipt(receipt, group="public_hydro", day=day,
                bundle=tmp_path, cutoff=gate.delivery_grid(day)[2], allow_training_bootstrap=True)


@pytest.mark.parametrize("marker", [None, {}, []])
def test_invalid_snapshot_marker_cannot_fall_back_to_strict_receipt(tmp_path, marker):
    day = "2026-09-29"
    snapshot = context.make_context(day, "2026-09-28T01:00:00+00:00")
    receipt = _source_receipt(tmp_path, snapshot)
    receipt["operational_snapshot"] = marker
    with pytest.raises(ValueError, match="exact preparation context"):
        gate.validate_source_receipt(receipt, group="public_hydro", day=day,
            bundle=tmp_path, cutoff=gate.delivery_grid(day)[2], allow_training_bootstrap=True)


def test_operational_baseline_still_refuses_future_labels_and_origins():
    day = "2026-09-29"
    snapshot = context.make_context(day, "2026-09-28T01:00:00+00:00")
    with context.context_scope(snapshot):
        full, current, cutoff = gate.delivery_grid(day)
        frame = pd.DataFrame({"actual": np.zeros(len(full)), "nyx__q50": np.ones(len(full)),
                              "forecast_origin_utc": [cutoff] * len(full)}, index=full)
        frame.loc[current, "actual"] = np.nan
        gate.validate_baseline(frame, full, current, cutoff, "snapshot baseline")
        frame.loc[current[0], "actual"] = 10.
        with pytest.raises(ValueError, match="future labels absent"):
            gate.validate_baseline(frame, full, current, cutoff, "snapshot baseline")
        frame.loc[current, "actual"] = np.nan
        frame.loc[current[0], "forecast_origin_utc"] = cutoff + pd.Timedelta(seconds=1)
        with pytest.raises(ValueError, match="after D-1 08:00"):
            gate.validate_baseline(frame, full, current, cutoff, "snapshot baseline")


def test_default_grid_and_metadata_keep_the_strict_contract():
    assert context.operational_context() is None
    assert not context.is_operational()
    assert context.metadata("2026-09-29") == {}
    assert gate.delivery_grid("2026-09-29")[2] == pd.Timestamp("2026-09-28T06:00:00+00:00")
