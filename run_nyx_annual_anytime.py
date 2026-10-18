#!/usr/bin/env python
"""Run the original annual CPU country recipes for tomorrow at any Paris hour."""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import timedelta
import json
from pathlib import Path
import sys
from uuid import uuid4

import pandas as pd

from chronos2_hourly.nyx_annual_live_preflight import ROOT
from chronos2_hourly.nyx_annual_operational_context import context_scope, make_context
from chronos2_hourly.nyx_annual_pipeline import run as prepare, _write
from run_nyx_annual_cpu_reports import run as reports, _Tee
from experiment_console.security import redact_text

RUN_ROOT = ROOT / "runs/nyx_annual_anytime"
REPORT_ROOT = ROOT / "runs/nyx_annual_anytime_reports"
PROTOCOL = "nyx_annual_anytime_run_v1"


def run(*, delivery_day=None, resume_run=None, now_utc=None,
        run_root=RUN_ROOT, report_root=REPORT_ROOT, prepare_runner=None,
        report_runner=None):
    """Persist the snapshot before collection; fresh calls never overwrite a run."""
    run_root, report_root = Path(run_root).resolve(), Path(report_root).resolve()
    if resume_run:
        directory = Path(resume_run).resolve()
        if not directory.is_relative_to(run_root) or directory == run_root:
            raise ValueError("Resume directory must be inside the annual anytime run root")
        record = json.loads((directory / "run.json").read_text(encoding="utf-8"))
        if not isinstance(record, dict) or record.get("protocol") != PROTOCOL:
            raise ValueError("Resume run protocol differs")
        if record.get("state") == "REPORTS_COMPLETE":
            raise ValueError("Completed run is immutable; start a new calculation")
        if record.get("state") not in ("CREATED", "PREPARING", "FITTING", "BLOCKED"):
            raise ValueError("Resume run state differs")
        context = record["operational_snapshot"]
        day = context["delivery_day"]
        if delivery_day and delivery_day != day:
            raise ValueError("Resume delivery day differs")
        output = report_root / directory.name
        if record.get("output") != str(output):
            raise ValueError("Resume output path differs")
        if record.get("run_directory", str(directory)) != str(directory):
            raise ValueError("Resume run directory differs")
        with context_scope(context):
            pass  # Reject a changed or future context before creating a log.
    else:
        now = pd.Timestamp.now(tz="UTC") if now_utc is None else pd.Timestamp(now_utc)
        if pd.isna(now) or now.tzinfo is None:
            raise ValueError("Timezone-aware launch time required")
        now = now.tz_convert("UTC")
        day = delivery_day or (now.tz_convert("Europe/Paris").date() + timedelta(days=1)).isoformat()
        context = make_context(day, now)
        with context_scope(context):
            pass
        directory = run_root / f"{day}_{now.strftime('%Y%m%dT%H%M%SZ')}_{uuid4().hex[:12]}"
        directory.mkdir(parents=True, exist_ok=False)
        output = report_root / directory.name
        record = {"protocol": PROTOCOL, "operational_snapshot": context,
                  "output": str(output), "run_directory": str(directory), "state": "CREATED"}
        _write(directory / "run.json", record)
    bundle, cache = directory / "bundle", directory / "cache"
    record["run_directory"] = str(directory)
    log = directory / f"launch_{uuid4().hex}.log"
    prepare_runner, report_runner = prepare_runner or prepare, report_runner or reports
    with log.open("x", encoding="utf-8", buffering=1) as stream:
        with redirect_stdout(_Tee(sys.stdout, stream)), redirect_stderr(_Tee(sys.stderr, stream)):
            with context_scope(context):
                print(json.dumps({"delivery_day": day, "run_directory": str(directory),
                                  "log": str(log), "operational_snapshot": context}), flush=True)
                record.pop("error", None)
                record.pop("pipeline", None)
                record.update(state="PREPARING", log=str(log), bundle=str(bundle))
                _write(directory / "run.json", record)
                try:
                    state = prepare_runner(day, action="prepare", bundle=bundle,
                                           output=output, cache_root=cache)
                    if state.get("state") != "PREPARED":
                        failures = [str(stage["error"]) for stage in state.get("stages", [])
                                    if isinstance(stage, dict) and stage.get("error")]
                        if state.get("remediation"):
                            failures.append(str(state["remediation"]))
                        record.update(state="BLOCKED", pipeline=state,
                            error=redact_text("; ".join(failures) or "Preparation did not complete"))
                        _write(directory / "run.json", record)
                        return record
                    record["state"] = "FITTING"
                    _write(directory / "run.json", record)
                    receipt = report_runner(bundle=bundle, delivery_day=day, output=output)
                    if receipt.get("status") != "REPORTS_COMPLETE":
                        raise ValueError("Reports did not complete")
                    record.update(state="REPORTS_COMPLETE", index=receipt["index"],
                                  qualified_for_production=False)
                except Exception as error:
                    record.update(state="BLOCKED", error=redact_text(f"{type(error).__name__}: {error}"))
                _write(directory / "run.json", record)
                return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delivery-day")
    parser.add_argument("--resume-run", type=Path)
    parser.add_argument("--result-file", type=Path)
    args = parser.parse_args(argv)
    try:
        result = run(delivery_day=args.delivery_day, resume_run=args.resume_run)
    except (OSError, ValueError, KeyError, ImportError) as error:
        result = {"protocol": PROTOCOL, "state": "BLOCKED",
                  "error": redact_text(f"{type(error).__name__}: {error}")}
    if args.result_file:
        _write(args.result_file, result)
    print(json.dumps(result, ensure_ascii=False, default=str), flush=True)
    return 0 if result["state"] == "REPORTS_COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
