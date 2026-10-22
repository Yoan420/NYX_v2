"""Étapes d'un lancement : préparer, vérifier, calculer, rendre, publier."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4

from . import runtime, snapshot
from .settings import RunSettings


def _validate_locations(settings: RunSettings) -> None:
    inputs, outputs = settings.input_root.resolve(), settings.output_root.resolve()
    if inputs == outputs or inputs.is_relative_to(outputs) or outputs.is_relative_to(inputs):
        raise ValueError("Les racines d'entrée et de sortie doivent être distinctes et non imbriquées.")
    for path in (inputs, outputs):
        if path == runtime.PRODUCTION or path.is_relative_to(runtime.PRODUCTION):
            raise ValueError("Les données doivent rester hors du dossier du code de production.")


def _prepare_runtime(settings: RunSettings) -> dict:
    """Vérifier le snapshot puis restaurer les états avant d'importer le calcul."""
    inputs = settings.input_root.resolve()
    document = snapshot.validate_snapshot(inputs)
    runtime.verify_runtime()
    os.environ["HF_HOME"] = str(inputs / "huggingface")
    os.environ["HF_HUB_CACHE"] = str(inputs / "huggingface/hub")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    identity = hashlib.sha256((inputs / "snapshot.json").read_bytes()).hexdigest()
    identity_path = runtime.RUNTIME / "runs/.nyx_snapshot.json"
    previous = json.loads(identity_path.read_text(encoding="utf-8")) if identity_path.is_file() else {}
    # Garder les états et captures récents tant que le seed ne change pas.
    if previous.get("snapshot_sha256") != identity:
        runtime.copy_state(inputs / "state", runtime.RUNTIME)
        runtime.write_json(identity_path, {"snapshot_sha256": identity})
    return document


def _record_failure(directory: Path, status: dict, error: BaseException) -> None:
    from .engine import redact_error
    status.update(
        state="INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "FAILED",
        error_type=type(error).__name__, error=redact_error(error),
    )
    runtime.write_json(directory / "status.json", status)


def execute(command: str, settings: RunSettings, plan: dict) -> int:
    """Exécuter le précontrôle ou le calcul complet sous les mêmes verrous."""
    if command not in ("preflight", "forecast"):
        raise ValueError("Le pipeline accepte preflight ou forecast.")
    _validate_locations(settings)
    with runtime.runtime_lock():
        document = _prepare_runtime(settings)
        # Imports différés : exporter des fichiers n'exige aucune dépendance
        # scientifique ni base de fuseaux horaires sur le poste d'origine.
        from . import detail_reports, forecast, preflight, reporting
        inputs = settings.input_root.resolve()
        config = {
            "repo_root": str(runtime.RUNTIME),
            "seed_bundle": str(inputs / "seed_bundle"),
            "seed_output": str(inputs / "seed_output"),
            "collect_reporting_sources": settings.collect_reporting_sources,
        }
        inspection = preflight.inspect_live(config, plan["first_day"], plan["forecast_origin_utc"])
        if command == "preflight":
            print(json.dumps({**inspection, "snapshot_protocol": document["protocol"]},
                             ensure_ascii=False, indent=2, default=str))
            return 0 if inspection["ready"] else 2

        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:12]
        directory = settings.output_root.resolve() / run_id
        directory.mkdir(parents=True, exist_ok=False)
        status = {**plan, "run_id": run_id, "state": "RUNNING", "new_predictions_computed": False}
        runtime.write_json(directory / "status.json", status)
        try:
            if not inspection["ready"]:
                status.update(state="BLOCKED", preflight=inspection)
                runtime.write_json(directory / "status.json", status)
                print(json.dumps(status, ensure_ascii=False, indent=2, default=str))
                return 2
            print(json.dumps({"state": "FITTING", "run_id": run_id, **plan}, ensure_ascii=False), flush=True)
            result = forecast.run_live(config, plan["first_day"], plan["forecast_origin_utc"], directory)
            result["reports"] = reporting.render_reports(Path(result["predictions_csv"]), directory / "reports")
            detail_reports.finalize_report_set(result["country_reports"], result["reports"])
            status.update(state="COMPLETE", new_predictions_computed=True, result=result)
            runtime.write_json(directory / "status.json", status)
            # Le pointeur avance seulement après le calcul et tous les HTML.
            runtime.write_json(settings.output_root.resolve() / "latest.json", {
                "run_id": run_id, "status_file": f"{run_id}/status.json",
                "predictions_csv": f"{run_id}/predictions.csv",
            })
            print(json.dumps(status, ensure_ascii=False, indent=2, default=str))
            return 0
        except BaseException as error:
            _record_failure(directory, status, error)
            raise


def render_saved_reports(csv_path: Path, output_root: Path) -> int:
    """Rendre les rapports d'un CSV existant avec le même suivi de lancement."""
    from . import reporting
    csv_path, output_root = csv_path.resolve(), output_root.resolve()
    if not csv_path.is_file():
        raise ValueError("reports exige un fichier CSV de prévisions existant.")
    if output_root == runtime.PRODUCTION or output_root.is_relative_to(runtime.PRODUCTION):
        raise ValueError("Les rapports doivent rester hors du dossier du code de production.")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_reports_" + uuid4().hex[:12]
    directory = output_root / run_id
    directory.mkdir(parents=True, exist_ok=False)
    status = {"protocol": "nyx_production_reports_v1", "run_id": run_id,
              "action": "reports", "state": "RUNNING", "new_predictions_computed": False}
    runtime.write_json(directory / "status.json", status)
    try:
        result = reporting.render_reports(csv_path, directory / "reports")
        status.update(state="COMPLETE", result=result)
        runtime.write_json(directory / "status.json", status)
        try:
            source_reference = os.path.relpath(csv_path, output_root)
        except ValueError:
            # Windows cannot express a relative path across drives/UNC shares.
            source_reference = str(csv_path)
        pointer = {
            "run_id": run_id, "action": "reports", "status_file": f"{run_id}/status.json",
            "predictions_csv": source_reference,
        }
        runtime.write_json(output_root / "latest_reports.json", pointer)
        print(json.dumps(status, ensure_ascii=False, indent=2, default=str))
        return 0
    except BaseException as error:
        _record_failure(directory, status, error)
        raise
