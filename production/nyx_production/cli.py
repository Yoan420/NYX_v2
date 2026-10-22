"""Point d'entrée de production. Les outils de restitution restent sans dépendance lourde."""
from __future__ import annotations

import argparse
from datetime import date, datetime, time, timedelta, timezone
import json
import os
from pathlib import Path
import sys
from uuid import uuid4
from zoneinfo import ZoneInfo

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def _write(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2,
                                        allow_nan=False, default=str) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_config(path: Path | None) -> dict:
    path = (path or PACKAGE_ROOT / "config.local.json").resolve()
    if not path.is_file():
        raise ValueError("Configuration absente : copier config.exemple.json vers config.local.json "
                         "et renseigner les chemins des états d'apprentissage. " + str(path))
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(config, dict):
        raise ValueError("La configuration doit être un objet JSON.")
    for key in ("repo_root", "seed_bundle", "seed_output", "output_root"):
        value = config.get(key)
        if value:
            candidate = Path(value).expanduser()
            config[key] = str((candidate if candidate.is_absolute() else path.parent / candidate).resolve())
    config.setdefault("repo_root", str(PACKAGE_ROOT / "runtime"))
    config.setdefault("output_root", str(PACKAGE_ROOT / "outputs"))
    return config


def _origin(first_day: str | None, cutoff: str | None) -> tuple[str, str]:
    paris = ZoneInfo("Europe/Paris")
    now = datetime.now(timezone.utc)
    day = date.fromisoformat(first_day) if first_day else now.astimezone(paris).date() + timedelta(days=1)
    if cutoff:
        stamp = datetime.fromisoformat(cutoff.replace("Z", "+00:00"))
        if stamp.tzinfo is None or stamp.utcoffset() is None:
            raise ValueError("Le cutoff doit comporter son fuseau ou son décalage UTC.")
        stamp = stamp.astimezone(timezone.utc)
    else:
        stamp = datetime.combine(day - timedelta(days=1), time(8), paris).astimezone(timezone.utc)
    if stamp >= datetime.combine(day, time(), paris).astimezone(timezone.utc):
        raise ValueError("L'origine doit précéder le premier jour de livraison.")
    if stamp > now:
        raise ValueError("Le cutoff demandé n'est pas encore atteint ; attendre 08 h Paris pour demain.")
    return day.isoformat(), stamp.isoformat()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("preflight", "forecast", "reports", "import"), required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--first-day")
    parser.add_argument("--cutoff-utc")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--html", type=Path, nargs="+")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    run_dir = None
    status = None
    try:
        config = load_config(args.config) if args.action in ("preflight", "forecast") else {}
        first_day = cutoff = None
        if args.action in ("preflight", "forecast"):
            first_day, cutoff = _origin(args.first_day, args.cutoff_utc)
            from .live import inspect_live, run_live
            inspection = inspect_live(config, first_day, cutoff)
            if args.action == "preflight":
                print(json.dumps(inspection, ensure_ascii=False, indent=2, default=str))
                return 0 if inspection.get("ready") is True else 2
            if inspection.get("ready") is not True:
                print(json.dumps(inspection, ensure_ascii=False, indent=2, default=str), file=sys.stderr)
                return 2
        if args.action == "reports" and (args.csv is None or not args.csv.is_file()):
            raise ValueError("L'action reports exige --csv et un fichier de prévisions existant.")
        if args.action == "import" and (not args.html or any(not p.is_file() for p in args.html)):
            raise ValueError("L'action import exige --html et les rapports HTML existants.")
        root = (args.output_root or Path(config.get("output_root", PACKAGE_ROOT / "outputs"))).resolve()
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + "_" + args.action + "_" + uuid4().hex[:8]
        run_dir = root / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        status = {"protocol": "nyx_production_reports_v1", "run_id": run_id, "action": args.action,
                  "state": "RUNNING", "first_delivery_day": first_day, "forecast_origin_utc": cutoff,
                  "started_at_utc": datetime.now(timezone.utc).isoformat(),
                  "new_predictions_computed": args.action == "forecast"}
        _write(run_dir / "status.json", status)
        from .report_io import import_reports, render_reports
        if args.action == "import":
            result = import_reports([p.resolve() for p in args.html], run_dir)
        elif args.action == "reports":
            result = render_reports(args.csv.resolve(), run_dir)
        else:
            result = run_live(config, first_day, cutoff, run_dir)
            csv_path = result.get("predictions_csv")
            if not csv_path or not Path(csv_path).is_file():
                raise ValueError("Le calcul n'a pas publié le CSV de prévisions attendu.")
            rendered = render_reports(Path(csv_path), run_dir / "reports")
            result["reports"] = rendered
        status.update(state="COMPLETE", finished_at_utc=datetime.now(timezone.utc).isoformat(), result=result)
        _write(run_dir / "status.json", status)
        pointer = {"run_id": run_id, "run_directory": run_id, "action": args.action,
                   "paths_relative_to": "this_file_parent", "status_file": run_id + "/status.json"}
        if result.get("predictions_csv"):
            pointer["predictions_csv"] = os.path.relpath(result["predictions_csv"], root)
        _write(root / ("latest_" + args.action + ".json"), pointer)
        _write(root / "latest.json", pointer)
        print(json.dumps(status, ensure_ascii=False, indent=2, default=str))
        return 0
    except KeyboardInterrupt:
        failure = {"state": "INTERRUPTED", "error_type": "KeyboardInterrupt",
                   "error": "Lancement interrompu par l'utilisateur.",
                   "finished_at_utc": datetime.now(timezone.utc).isoformat()}
        if run_dir is not None and status is not None:
            status.update(failure)
            _write(run_dir / "status.json", status)
        print(json.dumps(failure, ensure_ascii=False), file=sys.stderr)
        return 130
    except Exception as error:
        failure = {"state": "FAILED", "error_type": type(error).__name__, "error": str(error),
                   "finished_at_utc": datetime.now(timezone.utc).isoformat()}
        if run_dir is not None and status is not None:
            status.update(failure)
            _write(run_dir / "status.json", status)
        print(json.dumps(failure, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
