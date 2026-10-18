"""Check the public exchange API without creating a forecast-source capture."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from uuid import uuid4

import pandas as pd
import requests

import run_nyx_annual_exchange_source as exchange_source
from chronos2_hourly.nyx_annual_live_preflight import ROOT
from chronos2_hourly.nyx_local_io import publish_bytes
from experiment_console.security import redact, redact_text


PROTOCOL = "nyx_annual_exchange_connection_probe_v1"


def probe(delivery_day: str, *, session=None, utc_now=None, sleep=time.sleep) -> dict:
    """Test exactly the requested physical source window; never claim an old vintage."""
    current, first, stop, forecast_cutoff = exchange_source._source_window(delivery_day)
    clock = utc_now or (lambda: pd.Timestamp.now(tz="UTC"))
    started = exchange_source._utc(clock(), "Probe clock")
    # This bounds diagnostic retries only. It is NOT a forecast cutoff, and no
    # response or receipt is published as an input to prepare/forecast.
    deadline = started + pd.Timedelta(minutes=5)
    sources, objects = {}, {}
    own_session = session is None
    if own_session:
        session = requests.Session()
    try:
        for zone in ("DE", "FR"):
            try:
                response, retrieved = exchange_source._get_response(
                    session, zone, first, stop, deadline, utc_now=clock, sleep=sleep,
                    deadline_label="diagnostic time limit")
                payload, metadata = exchange_source._response(
                    response.content, zone, first, stop, retrieved, retrieved,
                    require_generated_before_retrieval=False)
                objects[zone] = payload
                sources[zone] = {"state": "COMPLETE", "http_status": response.status_code,
                    "retrieved_at_utc": retrieved.isoformat(), "provider_response": metadata,
                    "url": exchange_source.URLS[zone], "parameters": exchange_source._params(zone, first, stop)}
            except (ValueError, KeyError, TypeError, requests.RequestException) as error:
                sources[zone] = {"state": "ERROR", "error": redact_text(str(error)),
                                 "url": exchange_source.URLS[zone]}
        feature_error = None
        if len(objects) == 2:
            try:
                exchange_source._feature_frame(objects["DE"], objects["FR"], current)
            except (ValueError, KeyError, TypeError, AttributeError, AssertionError) as error:
                feature_error = redact_text(str(error))
        return {"protocol": PROTOCOL, "action": "probe", "delivery_day": delivery_day,
            "state": "COMPLETE" if len(objects) == 2 and feature_error is None else "ERROR",
            "started_at_utc": started.isoformat(), "finished_at_utc": clock().isoformat(),
            "forecast_cutoff_utc": forecast_cutoff.isoformat(),
            "source_window_utc": [first.isoformat(), stop.isoformat()], "sources": sources,
            "feature_error": feature_error, "capture_written": False,
            "forecast_enabled": False, "qualified_for_production": False,
            "scope": "Connectivity and current-response format only; this is not a pre-cutoff capture"}
    finally:
        if own_session:
            session.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delivery-day", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%SZ")
    output = args.output or ROOT / "runs/diagnostics" / f"exchange_probe_{stamp}_{uuid4().hex[:8]}.json"
    output = output.resolve()
    # A diagnostic can never occupy an input receipt or a source archive path.
    require_roots = (ROOT / "data", ROOT / "runs/live", ROOT / "runs/nyx_annual_cpu_live",
                     ROOT / "runs/nyx_annual_cpu_reports")
    if any(output.is_relative_to(path.resolve()) for path in require_roots):
        print("Diagnostic output must be outside source and forecast directories", flush=True)
        return 2
    if output.exists():
        print("Diagnostic output already exists; choose a new file", flush=True)
        return 2
    try:
        result = probe(exchange_source._day(args.delivery_day).isoformat())
    except (ValueError, KeyError, TypeError, requests.RequestException) as error:
        result = {"protocol": PROTOCOL, "action": "probe", "state": "ERROR",
                  "delivery_day": args.delivery_day, "error": redact_text(str(error)),
                  "capture_written": False, "forecast_enabled": False}
    payload = json.dumps(redact(result), ensure_ascii=False, indent=2) + "\n"
    publish_bytes(output, payload.encode("utf-8"))
    print(payload, flush=True)
    print(f"Diagnostic : {output}", flush=True)
    return 0 if result["state"] == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
