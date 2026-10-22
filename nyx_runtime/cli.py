"""Run the retained NYX CPU J+1/J+2 recipe locally or in AWS Batch."""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
from datetime import date, datetime, time, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tempfile
from uuid import uuid4
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
PRODUCTION = ROOT / "production"
RUNTIME = PRODUCTION / "runtime"
ZONES = ("FR", "DE", "BE", "NL")
STATE_PATHS = ("runs/nyx_daily_cpu_states", "runs/nyx_daily_refit_states",
               "data/pit/nyx_annual_hydro_captures", "data/pit/nyx_annual_exchange_captures",
               "data/pit/nyx_annual_jao_initial_live")


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                        allow_nan=False, default=str) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def make_plan(first_day: str | None = None, as_of: str | None = None) -> dict:
    """Freeze both civil dates from one real, explicit Paris auction cutoff."""
    paris = ZoneInfo("Europe/Paris")
    now = datetime.now(timezone.utc)
    origin = datetime.fromisoformat(as_of.replace("Z", "+00:00")) if as_of else None
    if origin is not None and (origin.tzinfo is None or origin.utcoffset() is None):
        raise ValueError("--as-of exige un fuseau ou un décalage UTC explicite.")
    if first_day:
        day = date.fromisoformat(first_day)
        if day.isoformat() != first_day:
            raise ValueError("--first-day exige YYYY-MM-DD.")
    else:
        day = (origin or now).astimezone(paris).date() + timedelta(days=1)
    expected = datetime.combine(day - timedelta(days=1), time(8), paris).astimezone(timezone.utc)
    if origin is not None and origin.astimezone(timezone.utc) != expected:
        raise ValueError("La recette exige exactement J 08:00 Europe/Paris pour les deux horizons.")
    if expected > now:
        raise ValueError("La coupure demandée n'est pas encore atteinte (08 h Europe/Paris).")
    deliveries = []
    for horizon in (1, 2):
        delivery = day + timedelta(days=horizon - 1)
        start = datetime.combine(delivery, time(), paris).astimezone(timezone.utc)
        end = datetime.combine(delivery + timedelta(days=1), time(), paris).astimezone(timezone.utc)
        deliveries.append({"horizon_day": horizon, "delivery_day": delivery.isoformat(),
                           "hours_per_country": int((end - start).total_seconds() / 3600)})
    return {"protocol": "nyx_aws_daily_refit_v1", "first_day": day.isoformat(),
            "forecast_origin_utc": expected.isoformat(), "countries": list(ZONES),
            "deliveries": deliveries, "qualified_for_production": False,
            "model": "amazon/chronos-2", "device": "cpu"}


def verify_runtime() -> None:
    document = json.loads((PRODUCTION / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    for item in document["source_files"]:
        relative = PurePosixPath(item["destination"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Chemin du manifeste source invalide.")
        path = PRODUCTION / relative
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(PRODUCTION.resolve()):
            raise ValueError(f"Source de production absente ou redirigée : {relative}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"Source de production modifiée : {relative}")


@contextmanager
def runtime_lock():
    """One writer for the real directories used by the retained runtime."""
    path = RUNTIME / "runs/.nyx_cloud.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        handle = path.open("x", encoding="utf-8")
    except FileExistsError as error:
        raise ValueError(f"Runtime déjà verrouillé : {path}. Vérifier le processus avant de retirer le verrou.") from error
    try:
        with handle:
            json.dump({"pid": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat()}, handle)
        yield
    finally:
        path.unlink(missing_ok=True)


def copy_state(source: Path, destination: Path) -> None:
    """Copy allowed data directories into real paths, never redirect model code."""
    for relative in STATE_PATHS:
        origin, target = source / relative, destination / relative
        if origin.is_symlink() or (origin.exists() and any(p.is_symlink() for p in origin.rglob("*"))):
            raise ValueError(f"Lien symbolique interdit dans un état : {origin}")
        if origin.exists() and not origin.is_dir():
            raise ValueError(f"Dossier d'état requis : {origin}")
        if target.exists():
            if target.is_symlink():
                raise ValueError(f"Destination d'état redirigée : {target}")
            shutil.rmtree(target)
        if not origin.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(origin, target)


def adapter():
    if str(PRODUCTION) not in sys.path:
        sys.path.insert(0, str(PRODUCTION))
    from nyx_production import live, report_io
    return live, report_io


def execute(args: argparse.Namespace, plan: dict) -> int:
    from .snapshot import validate_snapshot
    from .storage import S3Store, download_prefix, parse_s3_uri

    inputs, outputs = args.input_root.resolve(), args.output_root.resolve()
    if inputs == outputs or inputs.is_relative_to(outputs) or outputs.is_relative_to(inputs):
        raise ValueError("Les racines d'entrée et de sortie doivent être distinctes et non imbriquées.")
    for path in (inputs, outputs):
        if path == PRODUCTION or path.is_relative_to(PRODUCTION):
            raise ValueError("Les données doivent rester hors du dossier du code de production.")
    if args.input_s3_uri and args.output_s3_uri:
        input_bucket, input_prefix = parse_s3_uri(args.input_s3_uri)
        output_bucket, output_prefix = parse_s3_uri(args.output_s3_uri)
        if input_bucket == output_bucket and (
                not input_prefix or not output_prefix or input_prefix == output_prefix
                or input_prefix.startswith(output_prefix + "/")
                or output_prefix.startswith(input_prefix + "/")):
            raise ValueError("Les préfixes S3 d'entrée et de sortie doivent être distincts et non imbriqués.")
    store = S3Store(args.output_s3_uri) if args.output_s3_uri else None
    # The local lock also protects preflight's staging of state directories.
    with runtime_lock(), (store.lock() if store and args.command == "forecast" else nullcontext()):
        if args.input_s3_uri:
            if inputs.exists() and any(inputs.iterdir()):
                raise ValueError("Le téléchargement S3 exige un --input-root vide (snapshot immuable).")
            download_prefix(args.input_s3_uri, inputs)
        snapshot = validate_snapshot(inputs)
        verify_runtime()
        os.environ["HF_HOME"] = str(inputs / "huggingface")
        os.environ["HF_HUB_CACHE"] = str(inputs / "huggingface/hub")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        state_source = inputs / "state"
        snapshot_identity = hashlib.sha256((inputs / "snapshot.json").read_bytes()).hexdigest()
        local_identity_path = RUNTIME / "runs/.nyx_snapshot.json"
        local_identity = (json.loads(local_identity_path.read_text(encoding="utf-8"))
                          if local_identity_path.is_file() else {})
        # A published remote state supersedes the bootstrap snapshot as a whole.
        with tempfile.TemporaryDirectory(prefix="nyx-state-") as temporary:
            latest = store.restore_state(Path(temporary)) if store else None
            if latest:
                state_source = Path(temporary)
            if store or local_identity.get("snapshot_sha256") != snapshot_identity:
                copy_state(state_source, RUNTIME)
                write_json(local_identity_path, {"snapshot_sha256": snapshot_identity})
        live, report_io = adapter()
        config = {"repo_root": str(RUNTIME), "seed_bundle": str(inputs / "seed_bundle"),
                  "seed_output": str(inputs / "seed_output"),
                  "collect_reporting_sources": not args.skip_reporting_sources}
        inspection = live.inspect_live(config, plan["first_day"], plan["forecast_origin_utc"])
        if args.command == "preflight":
            print(json.dumps({**inspection, "snapshot_protocol": snapshot["protocol"]},
                             ensure_ascii=False, indent=2, default=str))
            return 0 if inspection["ready"] else 2
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:12]
        directory = outputs / run_id
        directory.mkdir(parents=True, exist_ok=False)
        status = {**plan, "run_id": run_id, "state": "RUNNING", "new_predictions_computed": False}
        write_json(directory / "status.json", status)
        try:
            if not inspection["ready"]:
                status.update(state="BLOCKED", preflight=inspection)
                write_json(directory / "status.json", status)
                if store:
                    store.publish_run(directory, run_id, success=False)
                print(json.dumps(status, ensure_ascii=False, indent=2, default=str))
                return 2
            print(json.dumps({"state": "FITTING", "run_id": run_id, **plan}, ensure_ascii=False), flush=True)
            result = live.run_live(config, plan["first_day"], plan["forecast_origin_utc"], directory)
            result["reports"] = report_io.render_reports(Path(result["predictions_csv"]), directory / "reports")
            status.update(state="COMPLETE", new_predictions_computed=True, result=result)
            write_json(directory / "status.json", status)
            if store:
                with tempfile.TemporaryDirectory(prefix="nyx-state-publish-") as temporary:
                    state = Path(temporary)
                    copy_state(RUNTIME, state)
                    store.publish_state(state, run_id)
                store.publish_run(directory, run_id, success=True)
            # A consumer only follows this pointer once both reports and remote publication succeed.
            write_json(outputs / "latest.json", {"run_id": run_id, "status_file": f"{run_id}/status.json",
                                                 "predictions_csv": f"{run_id}/predictions.csv"})
            print(json.dumps(status, ensure_ascii=False, indent=2, default=str))
            return 0
        except BaseException as error:
            status.update(state="INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "FAILED",
                          error_type=type(error).__name__, error=live._redact(error))
            write_json(directory / "status.json", status)
            if store:
                try:
                    store.publish_run(directory, run_id, success=False)
                except Exception:
                    print("Publication du diagnostic S3 impossible ; statut local conservé.", file=sys.stderr)
            raise


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    for name in ("plan", "preflight", "forecast"):
        command = commands.add_parser(name)
        command.add_argument("--first-day")
        command.add_argument("--as-of", help="Origine commune : J 08:00 Europe/Paris, avec décalage UTC.")
        command.add_argument("--horizons", type=int, nargs="+", default=[1, 2])
        command.add_argument("--zones", nargs="+", default=list(ZONES))
        if name != "plan":
            command.add_argument("--input-root", type=Path, default=Path("/work/input"))
            command.add_argument("--output-root", type=Path, default=Path("/work/output"))
            command.add_argument("--input-s3-uri", default=os.getenv("NYX_INPUT_S3_URI"))
            command.add_argument("--output-s3-uri", default=os.getenv("NYX_OUTPUT_S3_URI"))
            command.add_argument("--skip-reporting-sources", action="store_true")
    export = commands.add_parser("export-input", help="Exporter les seeds, poids et états du poste d'origine.")
    for flag in ("seed-bundle", "seed-output", "runtime-root", "hf-hub-cache", "destination"):
        export.add_argument("--" + flag, type=Path, required=True)
    reports = commands.add_parser("reports", help="Reconstruire les HTML depuis un CSV sans calcul de modèle.")
    reports.add_argument("--csv", type=Path, required=True)
    reports.add_argument("--output-root", type=Path, required=True)
    return result


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "export-input":
            from .snapshot import create_snapshot
            report = create_snapshot(seed_bundle=args.seed_bundle, seed_output=args.seed_output,
                runtime_root=args.runtime_root, hf_hub_cache=args.hf_hub_cache, destination=args.destination)
            print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
            return 0
        if args.command == "reports":
            sys.path.insert(0, str(PRODUCTION))
            from nyx_production.cli import main as production_main
            return production_main(["--action", "reports", "--csv", str(args.csv),
                                    "--output-root", str(args.output_root)])
        if sorted(args.horizons) != [1, 2] or len(args.zones) != 4 or set(args.zones) != set(ZONES):
            raise ValueError("La recette retenue exige J+1 et J+2 et les quatre pays FR/DE/BE/NL.")
        plan = make_plan(args.first_day, args.as_of)
        if args.command == "plan":
            print(json.dumps(plan, ensure_ascii=False, indent=2))
            return 0
        return execute(args, plan)
    except KeyboardInterrupt:
        print(json.dumps({"state": "INTERRUPTED"}), file=sys.stderr)
        return 130
    except Exception as error:
        # Includes provider failures; avoid exposing credentials in batch logs.
        sys.path.insert(0, str(PRODUCTION))
        from nyx_production.live import _redact
        print(json.dumps({"state": "BLOCKED", "error_type": type(error).__name__,
                          "error": _redact(error)}, ensure_ascii=False), file=sys.stderr)
        return 2
