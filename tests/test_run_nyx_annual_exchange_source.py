"""Prospective exchange capture tests use synthetic API bodies only."""
from __future__ import annotations

import json
import zipfile
from email.utils import format_datetime

import pandas as pd
import pytest
import requests

import run_nyx_annual_exchange_source as m


class Response:
    status_code = 200

    def __init__(self, content: bytes):
        self.content = content


class Session:
    def __init__(self, bodies):
        self.bodies = bodies
        self.calls = []

    def get(self, url, *, params, timeout, allow_redirects):
        zone = "DE" if url.endswith("/cbpf") else "FR"
        self.calls.append((zone, params))
        assert timeout == (10, 45) and allow_redirects is False
        return Response(self.bodies[zone])


def body(day: str, zone: str, *, missing_quarter=False,
         generated_after_cutoff=False, null_value=False) -> bytes:
    current, first, stop, cutoff = m._source_window(day)
    index = pd.date_range(first, stop, freq="15min", inclusive="left")
    if missing_quarter:
        index = index.delete(1)
    timezone = "Europe/Berlin" if zone == "DE" else "Europe/Paris"
    fields = ("france", "netherlands", "belgium", "sum") if zone == "DE" else (
        "cross_border_electricity_trading",)
    row = ({"france": -1.0, "netherlands": 2.0, "belgium": 0.0,
            "sum": 3.0} if zone == "DE" else
           {"cross_border_electricity_trading": -1000.0})
    payload = {
        "schema_version": "2.0", "country": zone.lower(),
        "endpoint": "cbpf" if zone == "DE" else "public_power",
        "unit": "GW" if zone == "DE" else "MW",
        "timezone": timezone, "interval_minutes": 15, "resolution": "PT15M",
        "generated_at": (cutoff + pd.Timedelta(minutes=1) if generated_after_cutoff
                         else cutoff - pd.Timedelta(hours=1)).isoformat(),
        "available_from": first.isoformat(),
        "available_until": (stop - pd.Timedelta(minutes=15)).isoformat(),
        "attributes": ({"sign_convention": "positive = import, negative = export"}
                       if zone == "DE" else {}),
        "series": [{"id": name} for name in fields],
        "data": [{"timestamp": t.tz_convert(timezone).isoformat(),
                  "values": row} for t in index],
    }
    if null_value:
        name = fields[0]
        payload["data"][0]["values"] = dict(row, **{name: None})
    assert len(current) in (23, 24, 25)
    return json.dumps(payload, separators=(",", ":")).encode()


@pytest.mark.parametrize("day,expected_hours", [("2026-03-29", 23),
                                                ("2026-10-25", 25)])
def test_daily_capture_is_causal_immutable_and_preserves_signs(tmp_path, day, expected_hours):
    cutoff = m._source_window(day)[-1]
    session = Session({zone: body(day, zone) for zone in ("DE", "FR")})
    receipt_path = m.capture(day, tmp_path, session=session,
                             now_utc=cutoff - pd.Timedelta(minutes=5))
    assert len(session.calls) == 2
    assert all(pd.Timestamp(params["end"]) < cutoff for _, params in session.calls)
    frame, receipt = m._verify_capture(receipt_path.parent, day)
    assert len(frame) == expected_hours
    assert frame[m.exchange.LEVELS[0]].eq(-1.).all()
    assert frame[m.exchange.LEVELS[-1]].eq(-1.).all()
    assert frame.filter(like="__available").eq(1.).all().all()
    assert receipt["provider_first_publication_timestamp_verified"] is False
    before = receipt_path.read_bytes()
    m.capture(day, tmp_path, session=Session({}),
              now_utc=cutoff - pd.Timedelta(minutes=5))
    assert receipt_path.read_bytes() == before


def test_late_or_unavailable_api_never_publishes_capture(tmp_path):
    day = "2026-09-29"
    cutoff = m._source_window(day)[-1]
    valid = {zone: body(day, zone) for zone in ("DE", "FR")}
    late = Session(valid)
    with pytest.raises(ValueError, match="capture starts"):
        m.capture(day, tmp_path, session=late,
                  now_utc=cutoff + pd.Timedelta(minutes=1))
    assert late.calls == []
    after = Session({"DE": body(day, "DE", generated_after_cutoff=True),
                     "FR": valid["FR"]})
    with pytest.raises(ValueError, match="before D-1 08"):
        m.capture(day, tmp_path, session=after,
                  now_utc=cutoff - pd.Timedelta(minutes=5))
    assert not (tmp_path / day / "capture.json").exists()
    incomplete = Session({"DE": body(day, "DE", missing_quarter=True),
                          "FR": valid["FR"]})
    with pytest.raises(ValueError, match="incomplete source interval grid"):
        m.capture(day, tmp_path, session=incomplete,
                  now_utc=cutoff - pd.Timedelta(minutes=5))
    assert not (tmp_path / day / "capture.json").exists()


def test_explicit_null_retains_historical_missing_flag_without_imputation(tmp_path):
    day = "2026-09-29"
    cutoff = m._source_window(day)[-1]
    session = Session({"DE": body(day, "DE"),
                       "FR": body(day, "FR", null_value=True)})
    path = m.capture(day, tmp_path, session=session,
                     now_utc=cutoff - pd.Timedelta(minutes=5))
    frame, receipt = m._verify_capture(path.parent, day)
    level = m.exchange.LEVELS[-1]
    assert pd.isna(frame[level].iloc[0])
    assert frame[level + "__available"].iloc[0] == 0
    assert receipt["missing_hours_by_feature"][level] == 1


def test_annual_assembly_refuses_missing_daily_vintages(tmp_path):
    day = "2026-09-29"
    cutoff = m._source_window(day)[-1]
    session = Session({zone: body(day, zone) for zone in ("DE", "FR")})
    m.capture(day, tmp_path / "archive", session=session,
              now_utc=cutoff - pd.Timedelta(minutes=5))
    bundle = tmp_path / "bundle"
    with pytest.raises(ValueError, match="No PIT exchange capture"):
        m.assemble(day, tmp_path / "archive", bundle)
    assert not (bundle / "source_receipts/lagged_exchange.json").exists()


def test_assembly_binds_verified_captures_to_preflight_receipt(tmp_path, monkeypatch):
    # Use the real daily captures and receipt validator with a two-day fixture.
    # Production's _capture_days remains the full 366 days.
    days = ["2026-09-28", "2026-09-29"]
    archive = tmp_path / "archive"
    for day in days:
        cutoff = m._source_window(day)[-1]
        session = Session({zone: body(day, zone) for zone in ("DE", "FR")})
        m.capture(day, archive, session=session,
                  now_utc=cutoff - pd.Timedelta(minutes=5))
    original_grid = m.delivery_grid

    def two_day_grid(day):
        full, current, cutoff = original_grid(day)
        if day == days[-1]:
            prior = original_grid(days[0])[1]
            return prior.append(current), current, cutoff
        return full, current, cutoff

    monkeypatch.setattr(m, "delivery_grid", two_day_grid)
    monkeypatch.setattr(m, "_capture_days", lambda day: days)
    bundle = tmp_path / "bundle"
    destination = m.assemble(days[-1], archive, bundle)
    receipt = json.loads(destination.read_text(encoding="utf-8"))
    assert receipt["source_group"] == "lagged_exchange"
    assert receipt["origin_snapshot_capture_verified"] is True
    assert receipt["provider_first_publication_timestamp_verified"] is False
    assert receipt["captures"] == 2
    m.validate_source_receipt(receipt, group="lagged_exchange", day=days[-1],
                              bundle=bundle, cutoff=original_grid(days[-1])[-1])
    frame = pd.read_parquet(bundle / "source_artifacts/lagged_exchange/features.parquet")
    assert len(frame) == 48 and list(frame.columns) == list(m.exchange.COLUMNS)
    with zipfile.ZipFile(bundle / "source_artifacts/lagged_exchange/pit_captures.zip") as saved:
        assert len(saved.namelist()) == 2 * len(m.CAPTURE_NAMES)


def test_tampered_raw_body_is_rejected_before_annual_assembly(tmp_path):
    day = "2026-09-29"
    cutoff = m._source_window(day)[-1]
    session = Session({zone: body(day, zone) for zone in ("DE", "FR")})
    receipt = m.capture(day, tmp_path, session=session,
                        now_utc=cutoff - pd.Timedelta(minutes=5))
    (receipt.parent / "FR.json").write_bytes(b"{}")
    with pytest.raises(ValueError, match="bytes changed"):
        m._verify_capture(receipt.parent, day)


class RetrySession:
    """Scripted HTTP responses preserve the actual session.get contract."""

    def __init__(self, day, events):
        self.day = day
        self.events = {zone: list(items) for zone, items in events.items()}
        self.calls = []

    def get(self, url, *, params, timeout, allow_redirects):
        zone = "DE" if url.endswith("/cbpf") else "FR"
        self.calls.append((zone, params))
        assert timeout == (10, 45) and allow_redirects is False
        event = self.events[zone].pop(0)
        if isinstance(event, Exception):
            raise event
        status, headers = event if isinstance(event, tuple) else (event, {})
        response = Response(body(self.day, zone))
        response.status_code, response.headers = status, headers
        return response


def test_transient_503_recovers_once_and_keeps_a_single_immutable_capture(tmp_path):
    day = "2026-10-03"
    cutoff = m._source_window(day)[-1]
    now = cutoff - pd.Timedelta(minutes=5)
    session = RetrySession(day, {"DE": [503, 200], "FR": [200]})
    waits = []
    path = m.capture(day, tmp_path, session=session, now_utc=now, sleep=waits.append)
    assert [zone for zone, _ in session.calls] == ["DE", "DE", "FR"]
    assert waits == [31.0]
    frame, receipt = m._verify_capture(path.parent, day)
    assert receipt["retrieved_at_utc"] == (now + pd.Timedelta(seconds=31)).isoformat()
    assert frame[m.exchange.LEVELS[0]].eq(-1).all()
    assert (path.parent / "DE.json").read_bytes() == body(day, "DE")
    assert sorted(item.name for item in path.parent.iterdir()) == sorted(m.CAPTURE_NAMES)
    saved = {name: (path.parent / name).read_bytes() for name in m.CAPTURE_NAMES}
    m.capture(day, tmp_path, session=Session({}), now_utc=now, sleep=waits.append)
    assert {name: (path.parent / name).read_bytes() for name in m.CAPTURE_NAMES} == saved


def test_three_server_errors_stop_without_publishing_any_capture(tmp_path):
    day = "2026-10-03"
    cutoff = m._source_window(day)[-1]
    session = RetrySession(day, {"DE": [500, 502, 504]})
    waits = []
    with pytest.raises(ValueError, match="HTTP 504 after 3"):
        m.capture(day, tmp_path, session=session,
                  now_utc=cutoff - pd.Timedelta(minutes=5), sleep=waits.append)
    assert len(session.calls) == 3 and waits == [31.0, 62.0]
    assert not (tmp_path / day).exists()


@pytest.mark.parametrize("date_header", [False, True])
def test_retry_after_seconds_and_http_date_are_respected(tmp_path, date_header):
    day = "2026-10-03"
    now = m._source_window(day)[-1] - pd.Timedelta(minutes=20)
    value = (format_datetime((now + pd.Timedelta(seconds=90)).to_pydatetime(), usegmt=True)
             if date_header else "90")
    session = RetrySession(day, {"DE": [(429, {"Retry-After": value}), 200], "FR": [200]})
    waits = []
    path = m.capture(day, tmp_path, session=session, now_utc=now, sleep=waits.append)
    _, receipt = m._verify_capture(path.parent, day)
    assert waits == [90.0]
    assert receipt["retrieved_at_utc"] == (now + pd.Timedelta(seconds=90)).isoformat()


@pytest.mark.parametrize("retry_after", ["0", "5", "invalid", "NaN"])
def test_retry_backoff_is_at_least_31_seconds(tmp_path, retry_after):
    day = "2026-10-03"
    session = RetrySession(day, {"DE": [(429, {"Retry-After": retry_after}), 200], "FR": [200]})
    waits = []
    m.capture(day, tmp_path, session=session,
              now_utc=m._source_window(day)[-1] - pd.Timedelta(minutes=5), sleep=waits.append)
    assert waits == [31.0]


def test_retry_after_exactly_600_seconds_is_allowed_when_cutoff_has_room(tmp_path):
    day = "2026-10-03"
    session = RetrySession(day, {"DE": [(429, {"Retry-After": "600"}), 200], "FR": [200]})
    waits = []
    m.capture(day, tmp_path, session=session,
              now_utc=m._source_window(day)[-1] - pd.Timedelta(minutes=20), sleep=waits.append)
    assert waits == [600.0]


@pytest.mark.parametrize("date_header", [False, True])
def test_retry_after_beyond_bound_never_retries_earlier_than_requested(tmp_path, date_header):
    day = "2026-10-03"
    now = m._source_window(day)[-1] - pd.Timedelta(minutes=20)
    value = (format_datetime((now + pd.Timedelta(seconds=601)).to_pydatetime(), usegmt=True)
             if date_header else "601")
    session = RetrySession(day, {"DE": [(429, {"Retry-After": value})]})
    waits = []
    with pytest.raises(ValueError, match="600-second retry limit"):
        m.capture(day, tmp_path, session=session, now_utc=now, sleep=waits.append)
    assert len(session.calls) == 1 and waits == []
    assert not (tmp_path / day).exists()


def test_wait_reaching_cutoff_stops_before_sleep_or_second_get(tmp_path):
    day = "2026-10-03"
    cutoff = m._source_window(day)[-1]
    session = RetrySession(day, {"DE": [503]})
    waits = []
    with pytest.raises(ValueError, match="retry would reach.*cutoff"):
        m.capture(day, tmp_path, session=session,
                  now_utc=cutoff - pd.Timedelta(seconds=31), sleep=waits.append)
    assert len(session.calls) == 1 and waits == []
    assert not (tmp_path / day).exists()


def test_no_get_starts_at_cutoff():
    day = "2026-10-03"
    _, first, stop, cutoff = m._source_window(day)
    session = Session({})
    with pytest.raises(ValueError, match="request reached.*cutoff"):
        m._get_response(session, "DE", first, stop, cutoff, now_utc=cutoff)
    assert session.calls == []


@pytest.mark.parametrize("error", [requests.Timeout("timeout"), requests.ConnectionError("disconnected")])
def test_transient_network_error_recovers(tmp_path, error):
    day = "2026-10-03"
    session = RetrySession(day, {"DE": [error, 200], "FR": [200]})
    waits = []
    path = m.capture(day, tmp_path, session=session,
                     now_utc=m._source_window(day)[-1] - pd.Timedelta(minutes=5), sleep=waits.append)
    m._verify_capture(path.parent, day)
    assert waits == [31.0] and len(session.calls) == 3


def test_ssl_failure_is_not_retried(tmp_path):
    day = "2026-10-03"
    session = RetrySession(day, {"DE": [requests.exceptions.SSLError("certificate verify failed")]})
    waits = []
    with pytest.raises(requests.exceptions.SSLError):
        m.capture(day, tmp_path, session=session,
                  now_utc=m._source_window(day)[-1] - pd.Timedelta(minutes=5), sleep=waits.append)
    assert len(session.calls) == 1 and waits == []
    assert not (tmp_path / day).exists()


@pytest.mark.parametrize("status", [400, 401, 403, 404, 501])
def test_non_transient_http_error_is_not_retried(tmp_path, status):
    day = "2026-10-03"
    session = RetrySession(day, {"DE": [status]})
    waits = []
    with pytest.raises(ValueError, match=f"HTTP {status} after 1"):
        m.capture(day, tmp_path, session=session,
                  now_utc=m._source_window(day)[-1] - pd.Timedelta(minutes=5), sleep=waits.append)
    assert len(session.calls) == 1 and waits == []
    assert not (tmp_path / day).exists()


def test_invalid_successful_payload_is_not_retried(tmp_path):
    day = "2026-10-03"
    session = Session({"DE": b"not json"})
    waits = []
    with pytest.raises(ValueError, match="invalid Energy-Charts JSON"):
        m.capture(day, tmp_path, session=session,
                  now_utc=m._source_window(day)[-1] - pd.Timedelta(minutes=5), sleep=waits.append)
    assert len(session.calls) == 1 and waits == []


def test_elapsed_waits_carry_from_de_to_fr_with_fixed_test_clock(tmp_path):
    day = "2026-10-03"
    session = RetrySession(day, {"DE": [503, 200], "FR": [503, 503]})
    waits = []
    with pytest.raises(ValueError, match="retry would reach.*cutoff"):
        m.capture(day, tmp_path, session=session,
                  now_utc=m._source_window(day)[-1] - pd.Timedelta(seconds=100), sleep=waits.append)
    assert [zone for zone, _ in session.calls] == ["DE", "DE", "FR", "FR"]
    assert waits == [31.0, 31.0]
    assert not (tmp_path / day).exists()


def test_live_clock_is_rechecked_after_request_returns(tmp_path):
    day = "2026-10-03"
    cutoff = m._source_window(day)[-1]
    current = [cutoff - pd.Timedelta(seconds=1)]
    session = RetrySession(day, {"DE": [200]})
    get = session.get

    def finishes_late(*args, **kwargs):
        response = get(*args, **kwargs)
        current[0] = cutoff
        return response

    session.get = finishes_late
    with pytest.raises(ValueError, match="response reached.*cutoff"):
        m.capture(day, tmp_path, session=session, utc_now=lambda: current[0])
    assert len(session.calls) == 1 and not (tmp_path / day).exists()


def test_retry_progress_is_structured_and_omits_exception_secrets(tmp_path, capsys):
    day = "2026-10-03"
    session = RetrySession(day, {"DE": [requests.ConnectionError("password=private-token"), 200], "FR": [200]})
    m.capture(day, tmp_path, session=session,
              now_utc=m._source_window(day)[-1] - pd.Timedelta(minutes=5), sleep=lambda seconds: None)
    output = capsys.readouterr().out
    assert json.loads(output) == {"state": "RETRYING", "country": "DE", "attempt": 2,
                                  "max_attempts": 3, "wait_seconds": 31.0, "http_status": None}
    assert "private-token" not in output


def test_diagnostic_helper_can_label_its_own_deadline():
    day = "2026-10-03"
    _, first, stop, cutoff = m._source_window(day)
    session = Session({})
    with pytest.raises(ValueError, match="diagnostic time limit") as raised:
        m._get_response(session, "DE", first, stop, cutoff, now_utc=cutoff,
                        deadline_label="diagnostic time limit")
    assert "08:00" not in str(raised.value) and session.calls == []
