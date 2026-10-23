"""Capture PIT lagged exchanges for the annual CWE CPU feature bundle.

Run ``capture`` before each delivery day's D-1 08:00 Paris cutoff.  Two
Energy-Charts v2 responses and their exact ten-column feature projection are
frozen locally.  ``assemble`` requires such a capture for every one of the
365 training days and the delivery day; it never backfills a missing vintage
from the latest API response or the fixed 2024-26 research archive.

The provider does not expose first-publication timestamps for individual
values.  A captured response proves only that its returned state was observed
before the recorded cutoff.  This producer does not construct the 503-column
model matrix or qualify a price model.
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta
from email.utils import parsedate_to_datetime
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import time
import zipfile

import numpy as np
import pandas as pd
import requests

from chronos2_hourly import nyx_lagged_exchange_features as exchange
from chronos2_hourly.nyx_annual_live_preflight import (
    ROOT, SOURCE_PROTOCOL, delivery_grid, sha256, validate_source_receipt,
)
from chronos2_hourly.process_lock import exclusive_process_lock
from chronos2_hourly.nyx_annual_operational_context import is_operational, metadata as snapshot_metadata, operational_context


PROTOCOL = "nyx_annual_lagged_exchange_capture_v1"
URLS = {"DE": "https://api.energy-charts.info/v2/cbpf",
        "FR": "https://api.energy-charts.info/v2/public_power"}
DEFAULT_ARCHIVE = ROOT / "data/pit/nyx_annual_exchange_captures"
CAPTURE_NAMES = ("DE.json", "FR.json", "features.parquet", "capture.json")
RETRYABLE_HTTP = frozenset((429, 500, 502, 503, 504))
MAX_ATTEMPTS = 3
MAX_RETRY_WAIT_SECONDS = 600


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def encoded(value: dict) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _utc(value: object, label: str) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    require(stamp.tzinfo is not None and not pd.isna(stamp),
            f"{label}: timezone-aware timestamp required")
    return stamp.tz_convert("UTC")


def _day(value: str) -> date:
    day = date.fromisoformat(value)
    require(day.isoformat() == value, "Delivery day must be YYYY-MM-DD")
    return day


def _source_window(day: str) -> tuple[pd.DatetimeIndex, pd.Timestamp,
                                      pd.Timestamp, pd.Timestamp]:
    _, current, cutoff = delivery_grid(day)
    first = current[0] - pd.Timedelta(hours=48)
    stop = current[-1] - pd.Timedelta(hours=47)
    require(stop > first and current.equals(pd.date_range(
        current[0], periods=len(current), freq="h")),
        "Delivery UTC grid is discontinuous")
    nominal = pd.Timestamp(operational_context(day)["nominal_cutoff_utc"]) if is_operational(day) else cutoff
    require(stop + pd.Timedelta(hours=1) <= nominal,
            "Lagged source nominal H+1 publication exceeds cutoff")
    return current, first, stop, cutoff


def _params(zone: str, first: pd.Timestamp, stop: pd.Timestamp) -> dict:
    return {"country": zone.lower(), "start": first.isoformat(),
            "end": (stop - pd.Timedelta(minutes=1)).isoformat()}


def _retry_after_seconds(response, now: pd.Timestamp) -> float:
    """Parse either HTTP Retry-After form; malformed headers use our backoff."""
    value = (getattr(response, "headers", None) or {}).get("Retry-After")
    if value is None:
        return 0.0
    try:
        seconds = float(value)
        if np.isfinite(seconds):
            return max(0.0, seconds)
    except (TypeError, ValueError):
        pass
    try:
        stamp = pd.Timestamp(parsedate_to_datetime(str(value)))
        if stamp.tzinfo is None:
            stamp = stamp.tz_localize("UTC")
        return max(0.0, (stamp.tz_convert("UTC") - now).total_seconds())
    except (TypeError, ValueError, OverflowError):
        return 0.0


def _get_response(session, zone: str, first: pd.Timestamp,
                  stop: pd.Timestamp, cutoff: pd.Timestamp, *,
                  now_utc: pd.Timestamp | None = None, utc_now=None,
                  sleep=time.sleep, deadline_label="D-1 08:00 cutoff"):
    """At most three transient-error attempts, all strictly before the cutoff.

    ``now_utc`` is the deterministic test clock used by capture's existing
    fixtures. It advances by every requested wait, even when test sleep returns
    immediately. ``utc_now`` can instead supply a changing test clock. Live
    calls always read the actual UTC time before and after requests and waits.
    """
    require(now_utc is None or utc_now is None, "Choose one exchange capture clock")
    fixed_now = None if now_utc is None else _utc(now_utc, "Capture clock")

    def clock():
        return (fixed_now if fixed_now is not None else
                _utc(utc_now() if utc_now is not None else pd.Timestamp.now(tz="UTC"),
                     "Capture clock"))

    for attempt in range(MAX_ATTEMPTS):
        require(clock() < cutoff, f"{zone}: Energy-Charts request reached {deadline_label}")
        response = None
        try:
            response = session.get(URLS[zone], params=_params(zone, first, stop),
                                   timeout=(10, 45), allow_redirects=False)
        except requests.exceptions.SSLError:
            # SSLError is also a ConnectionError; TLS failures are not retried.
            raise
        except (requests.Timeout, requests.ConnectionError):
            if attempt == MAX_ATTEMPTS - 1:
                raise
        retrieved = clock()
        require(retrieved < cutoff,
                f"{zone}: Energy-Charts response reached {deadline_label}")
        if response is not None:
            if response.status_code == 200:
                return response, retrieved
            require(response.status_code in RETRYABLE_HTTP and attempt < MAX_ATTEMPTS - 1,
                    f"{zone}: Energy-Charts HTTP {response.status_code} after {attempt + 1} attempt(s)")
        delay = max(31.0 * (attempt + 1), _retry_after_seconds(response, retrieved))
        require(delay <= MAX_RETRY_WAIT_SECONDS,
                f"{zone}: Energy-Charts Retry-After exceeds the 600-second retry limit")
        require(retrieved + pd.Timedelta(seconds=delay) < cutoff,
                f"{zone}: Energy-Charts retry would reach {deadline_label}")
        print(json.dumps({"state": "RETRYING", "country": zone,
                          "attempt": attempt + 2, "max_attempts": MAX_ATTEMPTS,
                          "wait_seconds": delay,
                          "http_status": response.status_code if response is not None else None}),
              flush=True)
        sleep(delay)
        if fixed_now is not None:
            fixed_now = retrieved + pd.Timedelta(seconds=delay)
    raise AssertionError("Exchange retry loop exhausted without a result")


def _response(raw: bytes, zone: str, first: pd.Timestamp,
              stop: pd.Timestamp, cutoff: pd.Timestamp,
              retrieved: pd.Timestamp, *, require_generated_before_retrieval: bool = True,
              operational: bool = False) -> tuple[dict, dict]:
    try:
        obj = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{zone}: invalid Energy-Charts JSON") from error
    require(isinstance(obj, dict), f"{zone}: response must be an object")
    generated = _utc(obj.get("generated_at"), f"{zone} generated_at")
    require((operational or retrieved <= cutoff) and (not require_generated_before_retrieval or generated <= retrieved),
            f"{zone}: response was not captured before D-1 08:00")
    available_from = _utc(obj.get("available_from"), f"{zone} available_from")
    available_until = _utc(obj.get("available_until"), f"{zone} available_until")
    require(operational or (available_from <= first and available_until >= stop - pd.Timedelta(hours=1)),
            f"{zone}: provider range does not cover lagged delivery source")
    rows = obj.get("data")
    require(isinstance(rows, list) and (operational or bool(rows)),
            f"{zone}: empty Energy-Charts response")
    stamps = []
    for row in rows:
        stamp = _utc(row["timestamp"], f"{zone} observation")
        require(first <= stamp < stop, f"{zone}: response escaped requested UTC interval")
        stamps.append(stamp)
    index = pd.DatetimeIndex(stamps)
    require(index.is_unique and index.is_monotonic_increasing,
            f"{zone}: unordered or duplicate source interval")
    step = obj.get("interval_minutes")
    require(step in ((15,) if zone == "DE" else (15, 60)),
            f"{zone}: unsupported source interval")
    if operational:
        subhour_blocks = {stamp.floor("h") for stamp in index if stamp != stamp.floor("h")}
        require(all(stamp + pd.Timedelta(minutes=15 if stamp.floor("h") in subhour_blocks else step) <= retrieved
                    for stamp in index),
                f"{zone}: observation interval ends after actual retrieval")
    elif step == 15:
        required = pd.date_range(first, stop, freq="15min", inclusive="left")
        require(index.equals(required), f"{zone}: incomplete source interval grid")
    else:
        for hour in pd.date_range(first, stop, freq="h", inclusive="left"):
            observed = index[(index >= hour) & (index < hour + pd.Timedelta(hours=1))]
            quarters = pd.date_range(hour, periods=4, freq="15min")
            require(observed.equals(pd.DatetimeIndex([hour])) or observed.equals(quarters),
                    f"{zone}: incomplete source interval grid")
    return obj, {"generated_at_utc": generated.isoformat(),
                 "available_from_utc": available_from.isoformat(),
                 "available_until_utc": available_until.isoformat(),
                 "observations": len(rows)}


def _feature_frame(de: dict, fr: dict, current: pd.DatetimeIndex) -> pd.DataFrame:
    hourly = exchange.exchange_hourly([de], [fr])
    country, audit = exchange.build_features(hourly, current)
    frame = country["FR"]
    for zone in exchange.ZONES[1:]:
        pd.testing.assert_frame_equal(frame, country[zone], check_exact=True)
    require(tuple(frame.columns) == exchange.COLUMNS
            and frame.index.equals(current)
            and not np.isinf(frame[list(exchange.LEVELS)].to_numpy(dtype=float)).any(),
            "Lagged exchange feature grid or schema differs")
    for level in exchange.LEVELS:
        values = frame[level].to_numpy(dtype=float)
        flags = frame[level + "__available"].to_numpy(dtype=float)
        require(np.isin(flags, (0., 1.)).all()
                and np.array_equal(flags, np.isfinite(values).astype(float)),
                f"{level}: missing value and availability flag differ")
    require(audit["cutoff_checked_all_zones_and_rows"] is True,
            "Lagged exchange cutoff audit absent")
    frame.index.name = "timestamp_utc"
    return frame


def _atomic_immutable(path: Path, content: bytes) -> str:
    """Publish bytes once; a conflicting rerun is a provenance error."""
    digest = hashlib.sha256(content).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(sha256(path) == digest, f"Immutable exchange artifact differs: {path}")
        return digest
    with tempfile.NamedTemporaryFile(prefix=path.name + ".", suffix=".tmp",
                                     dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(content)
    try:
        require(not path.exists(), f"Concurrent exchange artifact appeared: {path}")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return digest


def _parquet_bytes(frame: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    frame.to_parquet(buffer)
    return buffer.getvalue()


def _request_window_stop(receipt, zone, first, stop, retrieved, operational):
    """Verify a real, completed-hour request bounded inside the recipe window."""
    if not operational:
        return stop
    window = receipt.get("request_window_utc", {}).get(zone)
    require(isinstance(window, list) and len(window) == 2,
            f"{zone}: actual operational request window required")
    begin, end = (_utc(value, "Operational request window") for value in window)
    require(begin == first and first < end <= stop and end == end.floor("h")
            and end <= retrieved.floor("h"), f"{zone}: operational request extends beyond observed physical hours")
    return end


def _verify_capture(directory: Path, day: str) -> tuple[pd.DataFrame, dict]:
    current, first, stop, cutoff = _source_window(day)
    receipt_path = directory / "capture.json"
    require(receipt_path.is_file(), f"No PIT exchange capture for {day}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    operational = is_operational(day)
    require(receipt.get("operational_snapshot") == snapshot_metadata(day).get("operational_snapshot"),
            f"{day}: exchange operational context differs")
    require(receipt.get("protocol") == PROTOCOL and receipt.get("delivery_day") == day
            and receipt.get("cutoff_utc") == cutoff.isoformat()
            and receipt.get("state") == "COMPLETE"
            and receipt.get("source_window_utc") == [first.isoformat(), stop.isoformat()],
            f"{day}: exchange capture identity differs")
    retrieved = _utc(receipt.get("retrieved_at_utc"), "Exchange retrieval")
    require(operational or retrieved <= cutoff, f"{day}: exchange captured after cutoff")
    if operational:
        require(receipt.get("asof_cutoff_verified") is False
                and receipt.get("delivery_snapshot_pre_cutoff_verified") is bool(retrieved <= cutoff),
                f"{day}: exchange retrieval provenance differs")
    hashes = receipt.get("artifact_sha256")
    require(isinstance(hashes, dict) and set(hashes) == set(CAPTURE_NAMES[:3]),
            f"{day}: exchange capture artifact list differs")
    for name, expected in hashes.items():
        path = directory / name
        require(path.is_file() and sha256(path) == expected,
                f"{day}: exchange capture bytes changed: {name}")
    de_stop = _request_window_stop(receipt, "DE", first, stop, retrieved, operational)
    fr_stop = _request_window_stop(receipt, "FR", first, stop, retrieved, operational)
    de, de_meta = _response((directory / "DE.json").read_bytes(), "DE",
                            first, de_stop, cutoff, retrieved, operational=operational)
    fr, fr_meta = _response((directory / "FR.json").read_bytes(), "FR",
                            first, fr_stop, cutoff, retrieved, operational=operational)
    require(receipt.get("provider_response") == {"DE": de_meta, "FR": fr_meta},
            f"{day}: provider response provenance differs")
    frame = pd.read_parquet(directory / "features.parquet")
    expected = _feature_frame(de, fr, current)
    pd.testing.assert_frame_equal(frame, expected, check_exact=True, check_freq=False)
    require(receipt.get("missing_hours_by_feature") == {
        level: int(frame[level].isna().sum()) for level in exchange.LEVELS},
        f"{day}: exchange missingness provenance differs")
    return frame, receipt


def capture(day: str, archive: Path = DEFAULT_ARCHIVE, *, session=None,
            now_utc: pd.Timestamp | None = None, utc_now=None,
            sleep=time.sleep) -> Path:
    """Freeze the two API responses before the target day's legal cutoff."""
    current, first, stop, cutoff = _source_window(day)
    require(now_utc is None or utc_now is None, "Choose one exchange capture clock")
    def clock():
        return _utc(utc_now() if utc_now is not None else pd.Timestamp.now(tz="UTC"),
                    "Capture clock")
    now = clock() if now_utc is None else _utc(now_utc, "Capture clock")
    operational = is_operational(day)
    require(operational or now < cutoff, "Energy-Charts capture starts at or after D-1 08:00")
    directory = archive.resolve() / day
    if (directory / "capture.json").exists():
        _verify_capture(directory, day)
        return directory / "capture.json"
    own_session = session is None
    if own_session:
        session = requests.Session()
    bodies, metadata, request_windows = {}, {}, {}
    try:
        for zone in ("DE", "FR"):
            request_now = clock() if now_utc is None else now
            request_stop = min(stop, request_now.floor("h")) if operational else stop
            request_windows[zone] = [first.isoformat(), request_stop.isoformat()]
            request_deadline = request_now + pd.Timedelta(hours=1) if operational else cutoff
            response, retrieved = _get_response(session, zone, first, request_stop, request_deadline,
                now_utc=now if now_utc is not None else None,
                utc_now=utc_now, sleep=sleep,
                deadline_label="operational request deadline" if operational else "D-1 08:00 cutoff")
            if now_utc is not None:
                now = retrieved
            obj, meta = _response(response.content, zone, first, request_stop,
                                  cutoff, retrieved, operational=operational)
            bodies[zone], metadata[zone] = (response.content, obj), meta
        frame = _feature_frame(bodies["DE"][1], bodies["FR"][1], current)
        retrieved = clock() if now_utc is None else now
        require(operational or retrieved < cutoff, "Energy-Charts capture finished at or after cutoff")
        values = {"DE.json": bodies["DE"][0], "FR.json": bodies["FR"][0],
                  "features.parquet": _parquet_bytes(frame)}
        receipt = {
            "protocol": PROTOCOL, "delivery_day": day, "state": "COMPLETE",
            "cutoff_utc": cutoff.isoformat(),
            "retrieved_at_utc": retrieved.isoformat(),
            "source_window_utc": [first.isoformat(), stop.isoformat()],
            "target_hours": len(current),
            "provider_response": metadata,
            "missing_hours_by_feature": {
                level: int(frame[level].isna().sum()) for level in exchange.LEVELS},
            "artifact_sha256": {name: hashlib.sha256(raw).hexdigest()
                                for name, raw in values.items()},
                "source_end_plus_nominal_hour_before_cutoff": bool(stop + pd.Timedelta(hours=1) <= cutoff),
            "provider_first_publication_timestamp_verified": False,
            "provider_revision_vintage_verified": False,
            "capture_code_sha256": sha256(Path(__file__)),
            "feature_code_sha256": sha256(Path(exchange.__file__)),
            **snapshot_metadata(day),
        }
        if operational:
            receipt.update(asof_cutoff_verified=False, request_window_utc=request_windows,
                           delivery_snapshot_pre_cutoff_verified=bool(retrieved <= cutoff))
        directory.mkdir(parents=True, exist_ok=True)
        for name, raw in values.items():
            _atomic_immutable(directory / name, raw)
        destination = directory / "capture.json"
        _atomic_immutable(destination, encoded(receipt))
        _verify_capture(directory, day)
        return destination
    finally:
        if own_session:
            session.close()


def _capture_days(day: str) -> list[str]:
    end = _day(day)
    return [(end - timedelta(days=n)).isoformat() for n in range(365, -1, -1)]


def _zip_captures(archive: Path, days: list[str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6) as output:
        for day in days:
            for name in CAPTURE_NAMES:
                source = archive / day / name
                info = zipfile.ZipInfo(f"{day}/{name}", date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                output.writestr(info, source.read_bytes())
    return buffer.getvalue()


def assemble(day: str, archive: Path = DEFAULT_ARCHIVE,
             bundle: Path | None = None) -> Path:
    """Publish 366 verified daily vintages or leave the bundle incomplete."""
    full, _, cutoff = delivery_grid(day)
    days = _capture_days(day)
    frames, receipts = [], {}
    current_capture_time = None
    for capture_day in days:
        frame, receipt = _verify_capture(archive.resolve() / capture_day, capture_day)
        frames.append(frame)
        receipts[capture_day] = sha256(archive.resolve() / capture_day / "capture.json")
        if capture_day == day:
            current_capture_time = receipt["retrieved_at_utc"]
    combined = pd.concat(frames)
    require(combined.index.equals(full.rename("timestamp_utc"))
            and tuple(combined.columns) == exchange.COLUMNS,
            "Exchange capture series does not cover exact annual training plus delivery grid")
    bundle = (ROOT / "runs/live/nyx_annual_cpu" / day if bundle is None else bundle).resolve()
    artifacts = {
        "source_artifacts/lagged_exchange/features.parquet": _parquet_bytes(combined),
        "source_artifacts/lagged_exchange/pit_captures.zip":
            _zip_captures(archive.resolve(), days),
    }
    hashes = {name: hashlib.sha256(raw).hexdigest() for name, raw in artifacts.items()}
    receipt = {
        "protocol": SOURCE_PROTOCOL, "source_group": "lagged_exchange",
        "delivery_day": day, "state": "COMPLETE",
        "asof_cutoff_verified": True, "training_window_complete": True,
        "asof_state_utc": current_capture_time,
        "availability_basis": "366 immutable daily API captures observed before each D-1 08:00 cutoff",
        "origin_snapshot_capture_verified": True,
        "provider_first_publication_timestamp_verified": False,
        "provider_revision_vintage_verified": False,
        "publication_vintage_limit": "Capture proves response state at each origin, not first publication time of individual observations",
        "captures": len(days), "capture_receipt_sha256": receipts,
        "first_target_day": days[0], "last_target_day": day,
        "target_hours": len(full), "feature_columns": list(exchange.COLUMNS),
        "artifact_sha256": hashes, "collector_code_sha256": sha256(Path(__file__)),
        "feature_code_sha256": sha256(Path(exchange.__file__)),
        "model_inputs_complete": False,
    }
    # Validate all source bytes and time assertions before writing a COMPLETE receipt.
    for relative, raw in artifacts.items():
        _atomic_immutable(bundle / relative, raw)
    validate_source_receipt(receipt, group="lagged_exchange", day=day,
                            bundle=bundle, cutoff=cutoff)
    destination = bundle / "source_receipts/lagged_exchange.json"
    _atomic_immutable(destination, encoded(receipt))
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("capture", "assemble", "bootstrap"), required=True)
    parser.add_argument("--delivery-day", required=True)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--history-archive", type=Path)
    args = parser.parse_args(argv)
    day = _day(args.delivery_day).isoformat()
    with exclusive_process_lock(args.archive.resolve() / "exchange_source.lock"):
        if args.action == "capture":
            output = capture(day, args.archive)
        elif args.action == "bootstrap":
            from chronos2_hourly.nyx_annual_public_history import publish_history
            output = publish_history("lagged_exchange", day,
                args.bundle or ROOT / "runs/live/nyx_annual_cpu" / day,
                archive=args.archive, history_archive=args.history_archive)
        else:
            output = assemble(day, args.archive, args.bundle)
    print(json.dumps({"state": "COMPLETE", "action": args.action,
                      "delivery_day": day, "output": str(output),
                      "model_inputs_complete": False}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
