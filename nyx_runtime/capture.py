"""Captures publiques horodatées avant la coupure commune des prévisions."""
from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
import importlib
import json
import os
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo

from . import runtime
from .engine import activate_runtime, redact_error
from .settings import RunSettings


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def capture_day(now: datetime, requested_day: date | None = None) -> date:
    """La collecte originale accepte demain, entre 01 h 15 et 08 h Paris."""
    if now.tzinfo is None:
        raise ValueError("L'heure de capture doit porter un fuseau explicite.")
    local = now.astimezone(ZoneInfo("Europe/Paris"))
    day = local.date() + timedelta(days=1)
    if requested_day is not None and requested_day != day:
        raise ValueError(f"La capture vise uniquement la livraison de demain : {day}.")
    if not time(1, 15) <= local.time().replace(tzinfo=None) < time(8):
        raise ValueError(
            "Capture autorisée uniquement entre 01:15 et 08:00 Europe/Paris "
            "(08:00 exclu). Planifier à 07:00 ; une capture tardive ne remplace "
            "pas les données disponibles à la coupure."
        )
    return day


def execute_capture(
    settings: RunSettings,
    requested_day: date | None = None,
    verify_only: bool = False,
    ca_bundle: Path | None = None,
) -> int:
    """Préparer les états, collecter sans modèle et conserver les reçus.

    Le même verrou et la même identité de snapshot que forecast préservent les
    captures lors du calcul suivant. Une source en échec n'annule pas les
    captures réussies des autres sources ; le code de sortie reste non nul.
    """
    from .pipeline import _prepare_runtime, _validate_locations

    started = _utc_now()
    action = "verify" if verify_only else "capture"
    result = {"action": action, "state": "ERROR", "started_at_utc": started.isoformat(),
              "model_inputs_complete": False, "forecast_enabled": False}
    log_path = None
    try:
        _validate_locations(settings)
        run_id = started.strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:12]
        log_path = settings.output_root.resolve() / "captures" / (run_id + ".json")
        if "NYX_ANNUAL_OPERATIONAL_CONTEXT" in os.environ:
            raise ValueError(
                "Retirer NYX_ANNUAL_OPERATIONAL_CONTEXT : la capture quotidienne "
                "exige la fenêtre réelle de publication avant 08:00 Paris."
            )
        if verify_only:
            if requested_day is None:
                raise ValueError("--verify-only exige --delivery-day YYYY-MM-DD.")
            day = requested_day
        else:
            day = capture_day(started, requested_day)
        result["delivery_day"] = day.isoformat()
        with runtime.runtime_lock():
            _prepare_runtime(settings)
            # Le contrôle est refait après la préparation, qui peut prendre du temps.
            if not verify_only:
                capture_day(_utc_now(), day)
            activate_runtime(runtime.RUNTIME)
            collector = importlib.import_module("run_nyx_annual_daily_capture")
            paths = {
                "jao_cache": runtime.RUNTIME / "data/pit/nyx_annual_jao_initial_live",
                "exchange_archive": runtime.RUNTIME / "data/pit/nyx_annual_exchange_captures",
                "hydro_archive": runtime.RUNTIME / "data/pit/nyx_annual_hydro_captures",
            }
            if verify_only:
                result = collector.verify_daily(day, **paths)
            else:
                result = collector.capture_daily(requested_day=day, ca_bundle=ca_bundle, **paths)
        # Les erreurs fournisseur peuvent contenir des paramètres de connexion.
        for source in result.get("sources", {}).values():
            if "error" in source:
                source["error"] = redact_error(source["error"])
    except Exception as error:
        result.update(state="ERROR", error_type=type(error).__name__, error=redact_error(error))
    if log_path is not None:
        result["capture_log"] = str(log_path)
        runtime.write_json(log_path, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0 if result["state"] == "COMPLETE" else 2
