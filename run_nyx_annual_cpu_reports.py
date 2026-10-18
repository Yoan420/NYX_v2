#!/usr/bin/env python
"""Generate daily annual CPU reports with an explicit non-qualified status.

Consumes the same sealed prepare bundle and numerical recipes as the qualified
consumer. It never enables production, invents a qualification, or fetches data.
Late training-history downloads retain their real provenance in every report.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from html import escape
import json
import os
from pathlib import Path
import sys
import traceback
from uuid import uuid4

import pandas as pd

from chronos2_hourly.nyx_annual_cpu_live import (
    COUNTRIES, QUALIFICATION_CODE, ROOT, bundle_hashes, execute_models,
    load_bundle, require,
)
from chronos2_hourly.nyx_annual_cpu_reporting import write_country_report
from chronos2_hourly.nyx_annual_live_preflight import delivery_grid, sha256
from chronos2_hourly.nyx_annual_pipeline import validate_prepared_bundle
from chronos2_hourly.nyx_annual_operational_context import operational_context
from chronos2_hourly.nyx_local_io import promote_directory_retry, publish_bytes
from chronos2_hourly.process_lock import exclusive_process_lock
from experiment_console.security import redact, redact_text


PROTOCOL = "nyx_annual_cpu_daily_unqualified_reports_v1"
REPORT_ROOT = ROOT / "runs/nyx_annual_cpu_reports"


class _Tee:
    """Keep Python output durable even when PowerShell transcript omits it."""
    def __init__(self, console, log):
        self.console, self.log = console, log

    def write(self, value):
        self.log.write(value)
        self.log.flush()
        if self.console is not None:
            try:
                self.console.write(value)
            except UnicodeEncodeError:
                encoding = getattr(self.console, "encoding", None) or "ascii"
                self.console.write(value.encode(encoding, errors="replace").decode(encoding))
        return len(value)

    def flush(self):
        self.log.flush()
        if self.console is not None:
            self.console.flush()

    def isatty(self):
        return self.console is not None and self.console.isatty()

    @property
    def encoding(self):
        return getattr(self.console, "encoding", None) or "utf-8"


def _json(path: Path, value: dict) -> None:
    publish_bytes(path, (json.dumps(redact(value), ensure_ascii=False, indent=2,
                                   allow_nan=False) + "\n").encode("utf-8"))


def validate_paths(bundle: Path, output: Path) -> tuple[Path, Path]:
    bundle, output = bundle.resolve(), output.resolve()
    require(not (output.is_relative_to(bundle) or bundle.is_relative_to(output)),
            "Report output must not overlap the immutable input bundle")
    production = (ROOT / "runs/nyx_annual_cpu_live").resolve()
    require(not (output.is_relative_to(production) or production.is_relative_to(output)),
            "Non-qualified reports must not overlap qualified production outputs")
    attempts = (output.parent / f".{output.name}.attempts").resolve()
    require(not (attempts.is_relative_to(bundle) or bundle.is_relative_to(attempts)),
            "Report attempts must not overlap the immutable input bundle")
    require(not (attempts.is_relative_to(production) or production.is_relative_to(attempts)),
            "Report attempts must not overlap qualified production outputs")
    require(not os.path.lexists(output),
            "Report output already exists; choose a new immutable run directory")
    return bundle, output


def run(*, bundle: Path, delivery_day: str, output: Path) -> dict:
    """Publish four reports only after validating sources, baseline and reference."""
    _, _, cutoff = delivery_grid(delivery_day)
    require(pd.Timestamp.now(tz="UTC") >= cutoff,
            "Wait until D-1 08:00 Paris; capture public sources before then")
    bundle, output = validate_paths(bundle, output)
    # Serialize with prepare/forecast so a source producer cannot alter this
    # day's inputs during fitting. The consumer itself makes no source requests.
    with exclusive_process_lock(ROOT / "runs/live/nyx_annual_cpu/pipeline.lock"):
        with exclusive_process_lock(output.parent / f".{output.name}.lock"):
            bundle, output = validate_paths(bundle, output)
            sources = validate_prepared_bundle(bundle, delivery_day, require_asof=False)
            require(sources.get("passed") is True,
                    "Prepared source packet did not pass validation")
            require(isinstance(sources.get("source_snapshot_asof_verified"), bool),
                    "Prepared source packet has no factual cutoff status")
            return _run_attempt(bundle, delivery_day, output, sources, cutoff)


def _run_attempt(bundle: Path, delivery_day: str, output: Path,
                 sources: dict, cutoff: pd.Timestamp) -> dict:
    attempts = output.parent / f".{output.name}.attempts"
    attempt = attempts / uuid4().hex
    attempt.mkdir(parents=True, exist_ok=False)
    started = pd.Timestamp.now(tz="UTC").isoformat()

    def status(phase: str, state: str = "RUNNING", **details) -> None:
        payload = {"protocol": PROTOCOL, "status": state, "phase": phase,
                   "delivery_day": delivery_day, "started_at_utc": started,
                   "updated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                   "qualified_for_production": False, "forecast_published": False,
                   "attempt": str(attempt), **details}
        _json(attempt / "status.json", payload)
        print(json.dumps(redact(payload), ensure_ascii=False), flush=True)

    try:
        status("load_bundle")
        initial_hashes = bundle_hashes(bundle)
        data = load_bundle(bundle, delivery_day)
        require(bundle_hashes(bundle) == initial_hashes,
                "Prepared bundle changed while inputs were loaded")
        status("fit_models", total_models=7)
        frames, audit = execute_models(data, delivery_day, attempt / "models",
            progress=lambda phase, **details: status(phase, **details))
        require(bundle_hashes(bundle) == initial_hashes,
                "Prepared bundle changed during CPU retraining")
        require(validate_prepared_bundle(bundle, delivery_day, require_asof=False) == sources,
                "Prepared source evidence changed during CPU retraining")
        require(bundle_hashes(bundle) == initial_hashes,
                "Prepared bundle changed during final validation")
        status("write_reports")
        asof_verified = sources["source_snapshot_asof_verified"]
        context = operational_context(delivery_day)
        snapshot_note = ("Lancement : " + pd.Timestamp(context["requested_at_utc"]).tz_convert("Europe/Paris").isoformat()
                         + "; état Saturn interrogé : " + cutoff.tz_convert("Europe/Paris").isoformat()
                         + ". Collectes publiques horodatées dans les reçus; RMSE du protocole original à 08 h."
                         if context else "")
        countries = {}
        for zone in COUNTRIES:
            directory = attempt / "zones" / zone
            directory.mkdir(parents=True, exist_ok=False)
            name = f"forecast_{zone.lower()}_{delivery_day}_nyx_annual_cpu"
            frame = frames[zone]
            parquet_path = directory / f"{name}.parquet"
            csv_path = directory / f"{name}.csv"
            html_path = directory / f"{name}.html"
            frame.to_parquet(parquet_path, index=True)
            frame.to_csv(csv_path, index=True, float_format="%.17g")
            write_country_report(frame, html_path, country=zone, delivery_day=delivery_day,
                report_status={"qualification_status": "unqualified",
                               "public_history_before_forecast_cutoff": asof_verified,
                               "operational_snapshot": bool(context), "note": snapshot_note},
                de_performance_exception=zone == "DE")
            countries[zone] = {"hours": len(frame), "composition": audit["compositions"][zone],
                **{suffix: str(output / path.relative_to(attempt))
                   for suffix, path in (("html", html_path), ("csv", csv_path), ("parquet", parquet_path))},
                **{f"{suffix}_sha256": sha256(path)
                   for suffix, path in (("html", html_path), ("csv", csv_path), ("parquet", parquet_path))}}
        for family, item in audit["price_experts"].items():
            item["model"]["path"] = str(output / "models" / f"{family}.cbm")
        for zone, item in audit["negative_countries"].items():
            item["model"]["path"] = str(output / "models" / f"negative_{zone}.cbm")
        cutoff_note = ("Historiques publics disponibles avant la coupure de 08 h Paris."
                       if asof_verified else
                       "Historiques publics récupérés après la coupure de 08 h Paris : "
                       "ces rapports ne constituent pas une prévision validée à cette coupure.")
        if context:
            cutoff_note = escape(snapshot_note)
        links = "\n".join(f'<li><a href="zones/{zone}/{escape(Path(item["html"]).name)}">'
                          f'{zone} — prix et risque de prix négatif</a></li>'
                          for zone, item in countries.items())
        index = (f'<!doctype html><html lang="fr"><meta charset="utf-8">'
                 f'<title>NYX — {escape(delivery_day)} — rapports quotidiens</title>'
                 '<style>body{font:17px Arial;margin:40px;max-width:1000px;line-height:1.6}'
                 'aside{background:#fff2d9;padding:20px;border-left:5px solid #b36600}</style>'
                 f'<h1>NYX · Livraison du {escape(delivery_day)}</h1>'
                 '<aside><strong>Modèle non encore qualifié</strong><br>'
                 f'{cutoff_note}</aside><p>Rapports calculés le {escape(started)} (UTC).</p>'
                 f'<ul>{links}</ul></html>')
        (attempt / "index.html").write_text(index, encoding="utf-8")
        receipt = {"protocol": PROTOCOL, "status": "REPORTS_COMPLETE",
            "delivery_day": delivery_day, "started_at_utc": started,
            "finished_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "forecast_cutoff_utc": cutoff.isoformat(), "qualified": False,
            **({"operational_snapshot": context} if context else {}),
            "qualified_for_production": False, "forecast_published": False,
            "reports_published": True, "source_snapshot_asof_verified": asof_verified,
            "public_history_before_forecast_cutoff": asof_verified,
            "countries": countries, "bundle": str(bundle), "input_sha256": initial_hashes,
            "source_validation": sources, "model_audit": audit,
            "code_sha256": {name: sha256(ROOT / name) for name in (
                *QUALIFICATION_CODE, "run_nyx_annual_cpu_reports.py",
                "chronos2_hourly/nyx_annual_cpu_reporting.py")},
            "future_labels_used": False, "Storm_used_as_model_input": False,
            "saturn_fetched": False, "index": str(output / "index.html")}
        _json(attempt / "receipt.json", receipt)
        status("complete", "REPORTS_COMPLETE", receipt=str(output / "receipt.json"))
        promote_directory_retry(attempt, output)
        return receipt
    except BaseException as error:
        status("failed", "FAILED", error=redact_text(f"{type(error).__name__}: {error}"),
               traceback=redact_text(traceback.format_exc()))
        raise


def _launch(args) -> int:
    try:
        receipt = run(bundle=args.bundle, delivery_day=args.delivery_day,
                      output=args.output or REPORT_ROOT / args.delivery_day)
        print(json.dumps({"protocol": PROTOCOL, "status": "REPORTS_COMPLETE",
                          "qualified_for_production": False, "index": receipt["index"]},
                         ensure_ascii=False), flush=True)
        return 0
    except (OSError, ValueError, KeyError, TypeError, ImportError) as error:
        print(json.dumps({"protocol": PROTOCOL, "status": "BLOCKED",
                          "reason": redact_text(f"{type(error).__name__}: {error}")},
                         ensure_ascii=False), flush=True)
        return 2


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--delivery-day", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--log-file", type=Path)
    args = parser.parse_args(argv)
    if args.log_file is None:
        return _launch(args)
    try:
        log_path = args.log_file.resolve()
        bundle = args.bundle.resolve()
        production = (ROOT / "runs/nyx_annual_cpu_live").resolve()
        output = (args.output or REPORT_ROOT / args.delivery_day).resolve()
        require(not any(log_path.is_relative_to(path) for path in (bundle, production, output)),
                "Report log must be outside input and publication directories")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("x", encoding="utf-8", buffering=1) as stream:
            with redirect_stdout(_Tee(sys.stdout, stream)), redirect_stderr(_Tee(sys.stderr, stream)):
                return _launch(args)
    except (OSError, ValueError) as error:
        print(json.dumps({"protocol": PROTOCOL, "status": "BLOCKED",
                          "reason": redact_text(f"{type(error).__name__}: {error}")},
                         ensure_ascii=False), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
