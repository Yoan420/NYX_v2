"""Capture window, source failures and persistence across forecast preparation."""
from contextlib import nullcontext
from datetime import date, datetime
import json
from pathlib import Path

import pandas as pd
import pytest

from nyx_runtime import capture, pipeline, runtime
from nyx_runtime.settings import RunSettings
import run_nyx_annual_daily_capture as collector


@pytest.mark.parametrize("clock,delivery", [
    ("2026-03-28T06:00:00+00:00", "2026-03-29"),
    ("2026-03-29T05:00:00+00:00", "2026-03-30"),
    ("2026-10-24T05:00:00+00:00", "2026-10-25"),
    ("2026-10-25T06:00:00+00:00", "2026-10-26"),
])
def test_capture_window_uses_paris_civil_time_across_dst(clock, delivery):
    instant = datetime.fromisoformat(clock)
    assert capture.capture_day(instant) == date.fromisoformat(delivery)
    assert collector.target_day(pd.Timestamp(instant)) == date.fromisoformat(delivery)


@pytest.mark.parametrize("clock", [
    "2026-10-22T06:00:00+00:00",  # exactly 08:00 Paris
    "2026-10-22T00:14:00+02:00",
    "2026-10-22T01:14:59+02:00",
    "2026-10-25T07:00:00+00:00",  # winter cutoff
])
def test_capture_rejects_outside_real_publication_window(clock):
    with pytest.raises(ValueError, match="01:15"):
        capture.capture_day(datetime.fromisoformat(clock))


@pytest.fixture
def capture_environment(tmp_path, monkeypatch):
    clock = datetime.fromisoformat("2026-10-22T05:00:00+00:00")
    inputs, outputs, core = tmp_path / "input", tmp_path / "output", tmp_path / "runtime"
    inputs.mkdir()
    (inputs / "snapshot.json").write_text('{"protocol":"fixture"}', encoding="utf-8")
    monkeypatch.setattr(runtime, "RUNTIME", core)
    monkeypatch.setattr(capture, "_utc_now", lambda: clock)
    monkeypatch.setattr(capture, "activate_runtime", lambda path: None)
    monkeypatch.setattr(collector, "_clock", lambda _: pd.Timestamp(clock))
    monkeypatch.setattr(pipeline.snapshot, "validate_snapshot", lambda _: {"protocol": "fixture"})
    monkeypatch.setattr(collector, "_tls_configuration", lambda _: (True, "fixture"))
    monkeypatch.setattr(collector, "JaoCoreClient", lambda **_: nullcontext(object()))
    monkeypatch.delenv("NYX_ANNUAL_OPERATIONAL_CONTEXT", raising=False)

    def save(directory):
        assert (core / "runs/.nyx_snapshot.json").is_file()
        assert (core / "runs/.nyx_cloud.lock").is_file()
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "capture.json"
        path.write_text(json.dumps({"retrieved_at_utc": clock.isoformat()}), encoding="utf-8")
        return path

    def jao(**options):
        save(options["cache_root"] / options["day"].isoformat())
        return {"retrieved_at_utc": clock.isoformat()}

    monkeypatch.setattr(collector, "capture_day", jao)
    monkeypatch.setattr(collector.exchange_source, "capture",
                        lambda day, archive, **_: save(archive / day))
    monkeypatch.setattr(collector.hydro_source, "capture",
                        lambda day, archive, **_: save(archive / day))
    return RunSettings(inputs, outputs), core


def test_capture_persists_before_next_forecast_preparation(capture_environment, capsys):
    settings, core = capture_environment
    assert capture.execute_capture(settings) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["state"] == "COMPLETE"
    assert set(result["sources"]) == {"jao_initial", "lagged_exchange", "public_hydro"}
    receipts = list(core.glob("data/pit/*/2026-10-23/capture.json"))
    before = {path: path.read_bytes() for path in receipts}
    assert len(before) == 3
    with runtime.runtime_lock():
        pipeline._prepare_runtime(settings)
    assert {path: path.read_bytes() for path in receipts} == before
    assert not (core / "runs/.nyx_cloud.lock").exists()
    assert Path(result["capture_log"]).is_file()
    assert not (settings.output_root / "latest.json").exists()


def test_provider_failure_keeps_other_captures_and_returns_nonzero(
        capture_environment, monkeypatch, capsys):
    settings, core = capture_environment

    def fail(**_):
        raise RuntimeError("upstream unavailable token=private")

    monkeypatch.setattr(collector, "capture_day", fail)
    assert capture.execute_capture(settings) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["sources"]["jao_initial"]["state"] == "ERROR"
    assert "private" not in result["sources"]["jao_initial"]["error"]
    assert result["sources"]["lagged_exchange"]["state"] == "COMPLETE"
    assert result["sources"]["public_hydro"]["state"] == "COMPLETE"
    assert len(list(core.glob("data/pit/*/2026-10-23/capture.json"))) == 2
    assert not (core / "runs/.nyx_cloud.lock").exists()


def test_capture_checks_window_again_after_snapshot_preparation(
        capture_environment, monkeypatch, capsys):
    settings, core = capture_environment
    clocks = iter([datetime.fromisoformat("2026-10-22T05:59:59+00:00"),
                   datetime.fromisoformat("2026-10-22T06:00:00+00:00")])
    monkeypatch.setattr(capture, "_utc_now", lambda: next(clocks))
    assert capture.execute_capture(settings) == 2
    assert "08:00" in json.loads(capsys.readouterr().out)["error"]
    assert not list(core.glob("data/pit/*/*/capture.json"))


def test_capture_refuses_operational_override_and_wrong_requested_day(
        capture_environment, monkeypatch, capsys):
    settings, core = capture_environment
    monkeypatch.setenv("NYX_ANNUAL_OPERATIONAL_CONTEXT", "{}")
    assert capture.execute_capture(settings) == 2
    assert "NYX_ANNUAL_OPERATIONAL_CONTEXT" in json.loads(capsys.readouterr().out)["error"]
    assert not core.exists()
    monkeypatch.delenv("NYX_ANNUAL_OPERATIONAL_CONTEXT")
    assert capture.execute_capture(settings, date(2026, 10, 24)) == 2
    assert "demain" in json.loads(capsys.readouterr().out)["error"]


def test_verify_requires_day_and_never_calls_collectors(
        capture_environment, monkeypatch, capsys):
    settings, core = capture_environment
    assert capture.execute_capture(settings, verify_only=True) == 2
    assert "--delivery-day" in json.loads(capsys.readouterr().out)["error"]
    calls = []

    def verify(day, **paths):
        calls.append((day, paths))
        return {"action": "verify", "state": "COMPLETE", "sources": {}}

    monkeypatch.setattr(collector, "verify_daily", verify)
    monkeypatch.setattr(collector, "capture_daily", lambda **_: pytest.fail("network collector called"))
    monkeypatch.setattr(capture, "_utc_now", lambda: datetime.fromisoformat("2026-10-22T14:00:00+00:00"))
    assert capture.execute_capture(settings, date(2026, 10, 20), verify_only=True) == 0
    assert calls[0][0] == date(2026, 10, 20)
    assert calls[0][1]["jao_cache"] == core / "data/pit/nyx_annual_jao_initial_live"
