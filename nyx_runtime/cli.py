"""Commandes NYX : exporter, vérifier, prévoir et produire les rapports."""
from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import sys

from .planning import ZONES, make_plan
from .settings import load_settings


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    for name in ("plan", "preflight", "forecast"):
        command = commands.add_parser(name)
        command.add_argument("--first-day", help="Première livraison, au format YYYY-MM-DD.")
        command.add_argument("--as-of", help="Origine commune : J 08:00 Europe/Paris, avec décalage UTC.")
        command.add_argument("--horizons", type=int, nargs="+", default=[1, 2])
        command.add_argument("--zones", nargs="+", default=list(ZONES))
        if name != "plan":
            command.add_argument("--config", type=Path, help="Configuration JSON des chemins locaux.")
            command.add_argument("--input-root", type=Path)
            command.add_argument("--output-root", type=Path)
            sources = command.add_mutually_exclusive_group()
            sources.add_argument("--collect-reporting-sources", dest="collect_reporting_sources",
                                 action="store_true", default=None)
            sources.add_argument("--skip-reporting-sources", dest="collect_reporting_sources",
                                 action="store_false")

    export = commands.add_parser("export-input", help="Copier les seeds, poids et états du poste d'origine.")
    for flag in ("seed-bundle", "seed-output", "runtime-root", "hf-hub-cache", "destination"):
        export.add_argument("--" + flag, type=Path, required=True)
    export.add_argument("--restore-git-newlines", action="store_true",
                        help="Restaurer les fins de ligne JSON seulement si le SHA-256 du reçu seed correspond exactement.")

    capture = commands.add_parser("capture", help="Capturer les sources publiques avant 08 h à Paris.")
    capture.add_argument("--config", type=Path)
    capture.add_argument("--input-root", type=Path)
    capture.add_argument("--output-root", type=Path)
    capture.add_argument("--delivery-day", type=date.fromisoformat,
                         help="Vérification de date : demain à Paris, ou jour archivé avec --verify-only.")
    capture.add_argument("--verify-only", action="store_true")
    capture.add_argument("--ca-bundle", type=Path, help="Certificats PEM approuvés pour le proxy JAO.")

    reports = commands.add_parser("reports", help="Produire les HTML depuis un CSV de prévisions.")
    reports.add_argument("--csv", type=Path, required=True)
    reports.add_argument("--config", type=Path)
    reports.add_argument("--output-root", type=Path)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "export-input":
            from .snapshot import create_snapshot
            report = create_snapshot(
                seed_bundle=args.seed_bundle, seed_output=args.seed_output,
                runtime_root=args.runtime_root, hf_hub_cache=args.hf_hub_cache,
                destination=args.destination,
                restore_git_newlines=args.restore_git_newlines,
            )
            print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
            return 0
        if args.command == "reports":
            from .pipeline import render_saved_reports
            settings = load_settings(args.config, output_root=args.output_root)
            return render_saved_reports(args.csv, settings.output_root)

        if args.command == "capture":
            from .capture import execute_capture
            settings = load_settings(args.config, input_root=args.input_root,
                                     output_root=args.output_root)
            return execute_capture(settings, requested_day=args.delivery_day,
                                   verify_only=args.verify_only, ca_bundle=args.ca_bundle)

        if sorted(args.horizons) != [1, 2] or len(args.zones) != 4 or set(args.zones) != set(ZONES):
            raise ValueError("La recette retenue exige J+1 et J+2 et les quatre pays FR/DE/BE/NL.")
        plan = make_plan(args.first_day, args.as_of)
        if args.command == "plan":
            print(json.dumps(plan, ensure_ascii=False, indent=2))
            return 0

        from .pipeline import execute
        settings = load_settings(
            args.config, input_root=args.input_root, output_root=args.output_root,
            collect_reporting_sources=args.collect_reporting_sources,
        )
        return execute(args.command, settings, plan)
    except KeyboardInterrupt:
        print(json.dumps({"state": "INTERRUPTED"}), file=sys.stderr)
        return 130
    except Exception as error:
        from .engine import redact_error
        print(json.dumps({
            "state": "FAILED" if args.command == "reports" else "BLOCKED",
            "error_type": type(error).__name__, "error": redact_error(error),
        }, ensure_ascii=False), file=sys.stderr)
        return 2
