"""The API probe diagnoses current responses without manufacturing a vintage."""
from __future__ import annotations

import json

import pandas as pd
import pytest

import run_nyx_annual_exchange_probe as probe
from test_run_nyx_annual_exchange_source import body, Response


class Clock:
    def __init__(self):
        self.now = pd.Timestamp("2026-10-02T09:30:00Z")
        self.waits = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.waits.append(seconds)
        self.now += pd.Timedelta(seconds=seconds)


class Session:
    def __init__(self, de_status=200):
        self.de_status = de_status
        self.calls = []

    def get(self, url, **kwargs):
        zone = "DE" if url.endswith("cbpf") else "FR"
        self.calls.append(zone)
        response = Response(body("2026-10-03", zone))
        response.status_code = self.de_status if zone == "DE" else 200
        return response


def test_probe_after_forecast_cutoff_does_not_create_capture_or_claim_vintage(tmp_path, monkeypatch):
    monkeypatch.setattr(probe.exchange_source, "DEFAULT_ARCHIVE", tmp_path / "captures")
    clock = Clock()
    result = probe.probe("2026-10-03", session=Session(), utc_now=clock, sleep=clock.sleep)
    assert result["state"] == "COMPLETE"
    assert result["capture_written"] is False
    assert result["forecast_enabled"] is False
    assert result["qualified_for_production"] is False
    assert pd.Timestamp(result["started_at_utc"]) > pd.Timestamp(result["forecast_cutoff_utc"])
    assert set(result["sources"]) == {"DE", "FR"}
    assert not list(tmp_path.iterdir())


def test_probe_records_persistent_500_and_still_checks_fr_without_source_writes(tmp_path):
    session, clock = Session(de_status=500), Clock()
    result = probe.probe("2026-10-03", session=session, utc_now=clock, sleep=clock.sleep)
    assert result["state"] == "ERROR"
    assert "HTTP 500 after 3 attempt" in result["sources"]["DE"]["error"]
    assert result["sources"]["FR"]["state"] == "COMPLETE"
    assert session.calls == ["DE", "DE", "DE", "FR"]
    assert clock.waits == [31., 62.]
    assert not list(tmp_path.iterdir())


def test_probe_preserves_diagnostic_when_provider_feature_schema_is_malformed(tmp_path):
    class MalformedSession(Session):
        def get(self, url, **kwargs):
            response = super().get(url, **kwargs)
            if url.endswith("cbpf"):
                payload = json.loads(response.content)
                payload["attributes"] = None
                response.content = json.dumps(payload).encode("utf-8")
            return response

    clock = Clock()
    result = probe.probe("2026-10-03", session=MalformedSession(),
                         utc_now=clock, sleep=clock.sleep)
    assert result["state"] == "ERROR"
    assert result["feature_error"]
    assert set(result["sources"]) == {"DE", "FR"}
    assert result["capture_written"] is False
    assert not list(tmp_path.iterdir())


def test_probe_cli_writes_only_diagnostic(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "ROOT", tmp_path)
    result = {"protocol": probe.PROTOCOL, "state": "ERROR", "capture_written": False,
              "sources": {"DE": {"error": "HTTP 500"}}}
    monkeypatch.setattr(probe, "probe", lambda *_: result)
    output = tmp_path / "runs/diagnostics/probe.json"
    assert probe.main(["--delivery-day", "2026-10-03", "--output", str(output)]) == 2
    assert json.loads(output.read_text())["capture_written"] is False
    assert not (tmp_path / "data").exists()
    assert not (tmp_path / "runs/live").exists()


@pytest.mark.parametrize("folder", ["data/pit/captures", "runs/live/nyx_annual_cpu",
                                   "runs/nyx_annual_cpu_live", "runs/nyx_annual_cpu_reports"])
def test_invalid_diagnostic_path_cannot_issue_requests_or_overwrite_sources(tmp_path, monkeypatch, folder):
    monkeypatch.setattr(probe, "ROOT", tmp_path)
    monkeypatch.setattr(probe, "probe", lambda *_: pytest.fail("Unexpected network probe"))
    output = tmp_path / folder / "capture.json"
    assert probe.main(["--delivery-day", "2026-10-03", "--output", str(output)]) == 2
    assert not output.exists()
