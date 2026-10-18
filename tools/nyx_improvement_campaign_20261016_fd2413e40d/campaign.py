"""Run isolated NYX research stages against a pinned existing test session.

No data collectors, pretrained foundation inference or production writes occur.
The optional Replay stage refits canonical price/reference recipes; Challengers
fits explicitly different research models. Historical vintages remain conditional.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import html
from importlib.metadata import version, PackageNotFoundError
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from uuid import uuid4

from replay import atomic_json, coverage, read_json, require, safe_bound, seed_record, sha

ROOT = Path(__file__).resolve().parent
PROTOCOL = "nyx_improvement_campaign_v1"


def code_inventory():
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(ROOT.rglob("*"))
            if p.is_file() and p.suffix in (".py", ".ps1") and "__pycache__" not in p.parts}


def runtime():
    result = {"python": platform.python_version()}
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "catboost", "pyarrow", "filelock"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = "MISSING"
    return result


def child(name, *arguments):
    command = [sys.executable, "-u", str(ROOT / name), *map(str, arguments)]
    print("[NYX campagne] " + name + " " + " ".join(map(str, arguments)), flush=True)
    subprocess.run(command, check=True)


def create_or_load(annual_repo, app_repo, base_session=None, session_dir=None, new_session=False):
    annual_repo = Path(annual_repo).resolve()
    app_repo = Path(app_repo).resolve()
    require((annual_repo / ".venv-annual/Scripts/python.exe").is_file() or os.name != "nt",
            "Python annuel absent : corriger AnnualRepo, sans creer un nouvel environnement.")
    parent = annual_repo / "runs/nyx_improvement_campaign"
    parent.mkdir(parents=True, exist_ok=True)
    pointer = parent / "last_campaign.json"
    if session_dir is None and not new_session and pointer.exists():
        session_dir = Path(read_json(pointer)["session_dir"])
    if session_dir is not None:
        directory = Path(session_dir).resolve()
        session = read_json(directory / "campaign.json")
        require(session.get("protocol") == PROTOCOL, "Session de campagne incompatible")
        require(Path(session["annual_repo"]).resolve() == annual_repo and Path(session["app_repo"]).resolve() == app_repo,
                "Les depots de cette session sont figes : utiliser ses chemins d origine.")
        require(session["code_sha256"] == code_inventory(),
                "Les scripts ont change : creer une nouvelle campagne avec -NewSession.")
        require(session["runtime_versions"] == runtime(),
                "Les versions ont change : restaurer l environnement ou creer une nouvelle campagne.")
        require(sha(session["base_session_file"]) == session["base_session_sha256"],
                "La session de tests donneuse a change ; ne pas melanger les campagnes.")
        return directory, session
    if base_session is None:
        last = read_json(annual_repo / "runs/nyx_performance_tests/last_session.json")
        base_session = Path(last["session_dir"])
    base_file = Path(base_session).resolve() / "session.json"
    base = read_json(base_file)
    require(Path(base["annual_repo"]).resolve() == annual_repo and Path(base["app_repo"]).resolve() == app_repo,
            "La session donneuse concerne d autres depots : fournir -BaseSession explicitement.")
    require(isinstance(base.get("seed_run"), str) and base["seed_run"],
            "La session donneuse doit avoir termine son inventaire Pilot.")
    seed = seed_record(annual_repo, base["seed_run"])
    suffix = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:6]
    directory = parent / suffix
    directory.mkdir()
    session = {"protocol": PROTOCOL, "session_dir": str(directory), "annual_repo": str(annual_repo),
               "app_repo": str(app_repo), "base_session": str(base_file.parent),
               "base_session_file": str(base_file), "base_session_sha256": sha(base_file),
               "seed_run": str(seed["run_path"].parent), "threads": int(base.get("threads", 8)),
               "code_sha256": code_inventory(), "runtime_versions": runtime(),
               "production_modified": False, "historical_provider_vintages_certified": False}
    atomic_json(directory / "campaign.json", session)
    atomic_json(pointer, {"session_dir": str(directory)})
    return directory, session


def preflight():
    import numpy  # noqa: F401
    import pandas  # noqa: F401
    import sklearn  # noqa: F401
    import pyarrow  # noqa: F401
    import catboost  # noqa: F401
    import filelock  # noqa: F401


def stage(directory, name, inputs, function):
    """Reuse only complete stages with identical bound inputs and intact outputs."""
    receipt = directory / "stage_receipts" / (name + ".json")
    expected = {str(Path(p).resolve()): sha(p) for p in inputs}
    target = directory / name
    if receipt.exists():
        saved = read_json(receipt)
        require(saved["input_sha256"] == expected, name + ": les entrees ont change ; nouvelle session requise")
        for relative, digest in saved["output_sha256"].items():
            safe_bound(target, relative, saved["output_sha256"])
        print("[NYX campagne] Reutilise : " + name, flush=True)
        return target
    if target.exists():
        quarantine = directory / "interrupted" / (name + "_" + uuid4().hex)
        quarantine.parent.mkdir(exist_ok=True)
        shutil.move(str(target), str(quarantine))
    function(target)
    require(target.is_dir(), name + ": aucun resultat genere")
    require(expected == {str(Path(p).resolve()): sha(p) for p in inputs}, name + ": entree modifiee pendant le calcul")
    files = {p.relative_to(target).as_posix(): sha(p) for p in sorted(target.rglob("*")) if p.is_file()}
    require(files, name + ": resultats vides")
    atomic_json(receipt, {"input_sha256": expected, "output_sha256": files})
    return target


def import_replay(directory, session):
    """Copy verified completed old checkpoints; never write the old research run."""
    destination = directory / "base_replay"
    if (directory / "replay_import.json").exists():
        return destination
    source = Path(session["base_session"]) / "final_replay"
    copied = []
    plan = source / "replay_plan.json"
    if plan.is_file():
        document = read_json(plan)
        require(Path(document["seed_run"]).resolve() == Path(session["seed_run"]).resolve(), "Donneur du rejeu different")
        require(document["threads"] == session["threads"], "Threads du rejeu donneur differents")
        plan_digest = sha(plan)
        destination.mkdir(exist_ok=True)
        shutil.copy2(plan, destination / plan.name)
        for origin in sorted((source / "origins").glob("*")):
            receipt = origin / "receipt.json"
            if not receipt.is_file():
                continue
            doc = read_json(receipt)
            require(doc.get("plan_sha256") == plan_digest and doc.get("predictions_sealed_before_scoring") is True,
                    "Checkpoint donneur sans scellement valide")
            hashes = doc.get("artifact_sha256", {})
            require(hashes and doc.get("day") == origin.name, "Checkpoint donneur incomplet")
            for relative in hashes:
                safe_bound(origin, relative, hashes)
            target = destination / "origins" / origin.name
            target.mkdir(parents=True, exist_ok=True)
            for relative in hashes:
                path = target / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(origin / relative, path)
                require(sha(path) == hashes[relative], "Copie de checkpoint incomplete")
            shutil.copy2(receipt, target / "receipt.json")
            require(sha(plan) == plan_digest and sha(receipt) == sha(target / "receipt.json"), "Le rejeu donneur a change pendant la copie")
            copied.append(origin.name)
    atomic_json(directory / "replay_import.json", {"source": str(source), "copied_sealed_origins": copied,
                                                   "original_research_run_modified": False})
    return destination


def reporting_stage(directory, name, csv, session):
    return stage(directory, name, [csv, csv.with_suffix(".manifest.json")],
                 lambda output: child("final_study.py", "--input", csv, "--output", output,
                                      "--annual-repo", session["annual_repo"], "--app-repo", session["app_repo"]))


def study_stage(directory, name, csv, extra=(), holdout=30):
    arguments = ["--input", csv, "--holdout-days", str(holdout)]
    if extra:
        arguments += ["--extra-components", ",".join(extra)]
    result = stage(directory, name, [csv, csv.with_suffix(".manifest.json")],
                   lambda output: child("study.py", *arguments, "--output", output))
    sources = [result / "study.json", result / "predictions.csv", result / "complete.json"]
    stage(directory, name + "_intervals", sources,
          lambda output: child("intervals.py", "--study-dir", result, "--output", output))
    stage(directory, name + "_stability", sources,
          lambda output: child("stability.py", "--study-dir", result, "--output", output))
    return result


def run_phase(phase, directory, session):
    seed = seed_record(session["annual_repo"], session["seed_run"])
    pinned = [seed["run_path"], seed["receipt_path"]]
    if phase == "Screen":
        def audit(output):
            child("source_audit.py", "--annual-repo", session["annual_repo"], "--seed-run", session["seed_run"], "--output", output)
        stage(directory, "source_audit", pinned, audit)
        reference = stage(directory, "reference_inputs", pinned,
                          lambda output: child("reference_pilot.py", "--annual-repo", session["annual_repo"],
                                               "--app-repo", session["app_repo"], "--seed-run", session["seed_run"], "--output", output))
        study_stage(directory, "reference_study", reference / "OOF.csv")
    elif phase == "Replay":
        available = coverage(seed)
        atomic_json(directory / "coverage.json", available)
        if not available["conditional_j1_available_days"]:
            atomic_json(directory / "replay_not_ready.json", {"status": "NOT_READY", "reason": available["explanation"]})
            print("[NYX campagne] Aucun rejeu final complet disponible. Consulter coverage.json", flush=True)
            return
        target = import_replay(directory, session)
        child("replay.py", "replay", "--annual-repo", session["annual_repo"], "--seed-run", session["seed_run"],
              "--output", target, "--days", "0", "--threads", session["threads"], "--implementation-root", ROOT / "annual_extension")
        reporting = reporting_stage(directory, "base_reporting", target / "OOF.csv", session)
        study_stage(directory, "base_study", reporting / "OOF.csv")
    elif phase == "Challengers":
        target = import_replay(directory, session)
        if not list((target / "origins").glob("*/receipt.json")):
            atomic_json(directory / "challengers_not_ready.json", {"status": "NOT_READY", "reason": "Lancer Replay pour produire au moins une origine scellee."})
            print("[NYX campagne] Aucun checkpoint final scelle. Lancer -Phase Replay.", flush=True)
            return
        child("challengers.py", "--annual-repo", session["annual_repo"], "--seed-run", session["seed_run"],
              "--base-replay", target, "--output", directory / "challengers", "--threads", session["threads"],
              "--implementation-root", ROOT / "annual_extension")
        reporting = reporting_stage(directory, "challenger_reporting", directory / "challengers/OOF.csv", session)
        extra = read_json(directory / "challengers/OOF.manifest.json").get("extra_components", [])
        require(extra, "Le manifeste des nouveaux experts ne declare pas ses colonnes")
        study_stage(directory, "challenger_study", reporting / "OOF.csv", extra)
    elif phase == "Archive":
        suffix = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:6]
        generation = directory / "archives" / suffix
        generation.mkdir(parents=True)
        child("reference_pilot.py", "--annual-repo", session["annual_repo"], "--app-repo", session["app_repo"],
              "--seed-run", session["seed_run"], "--output", generation / "labels")
        child("replay.py", "collect", "--annual-repo", session["annual_repo"], "--app-repo", session["app_repo"],
              "--labels", generation / "labels/report_labels.csv", "--output", generation / "raw")
        manifest_path = generation / "raw/OOF.manifest.json"
        manifest = read_json(manifest_path)
        manifest["seed_run"] = session["seed_run"]
        atomic_json(manifest_path, manifest)
        if manifest["scorable_rows"]:
            reporting = reporting_stage(generation, "reporting", generation / "raw/OOF.csv", session)
            study_stage(generation, "study", reporting / "OOF.csv")
        else:
            atomic_json(generation / "status.json", {"status": "NOT_READY", "reason": "Aucune journee EPEX complete", "pending_rows": manifest["pending_rows"]})
        atomic_json(directory / "last_archive.json", {"generation": str(generation)})
    elif phase == "Annual":
        choices = []
        for name in ("challenger_reporting", "base_reporting"):
            csv = directory / name / "OOF.csv"
            if csv.is_file():
                choices.append((name, csv))
        if (directory / "last_archive.json").exists():
            csv = Path(read_json(directory / "last_archive.json")["generation"]) / "reporting/OOF.csv"
            if csv.is_file():
                choices.append(("archive_" + csv.parent.parent.name, csv))
        for name, csv in choices:
            manifest = read_json(csv.with_suffix(".manifest.json"))
            study_stage(directory, "annual_" + name, csv, manifest.get("extra_components", []), 365)
        atomic_json(directory / "annual_readiness.json", {"datasets_checked": [name for name, _ in choices],
                    "production_qualification": False,
                    "note": "365 jours de test exigent aussi un historique distinct pour initialisation et selection. Les sources conditionnelles ne certifient jamais la production."})


def summary(directory):
    statuses = []
    for path in sorted(directory.rglob("status.json")):
        if any(p in ("interrupted", "base_replay", "origins") for p in path.relative_to(directory).parts):
            continue
        statuses.append({"path": path.relative_to(directory).as_posix(), "result": read_json(path)})
    reports = [p.relative_to(directory).as_posix() for p in sorted(directory.rglob("report.html"))
               if "interrupted" not in p.relative_to(directory).parts]
    body = '<h1>NYX — campagne d’amélioration</h1><p>Chronos reste fixe. Aucun modèle de production n’est modifié. Les études de référence, les replays conditionnels et les archives finales sont évalués séparément. Un statut NOT_READY est un résultat de couverture, pas une preuve de performance.</p>'
    body += '<ul>' + ''.join('<li><a href="' + html.escape(p, quote=True) + '">' + html.escape(p) + '</a></li>' for p in reports) + '</ul>'
    body += '<p>Ordre : Screen, Replay, Challengers, Archive, Annual. Le pilote de référence ne qualifie pas les experts finaux. J+2 n’est jamais reconstruit depuis J+1. Les journées sans EPEX restent en attente. Les comparaisons Storm sont conditionnelles à la révision disponible.</p>'
    flags = {name: read_json(directory / name) for name in ("coverage.json", "replay_not_ready.json", "challengers_not_ready.json", "annual_readiness.json") if (directory / name).is_file()}
    if flags:
        body += '<h2>Couverture et étapes disponibles</h2><pre>' + html.escape(json.dumps(flags, ensure_ascii=False, indent=2)) + '</pre>'
    body += '<pre>' + html.escape(json.dumps(statuses, ensure_ascii=False, indent=2)) + '</pre>'
    (directory / "index.html").write_text('<!doctype html><html lang="fr"><meta charset="utf-8"><title>NYX — campagne</title><style>body{font:16px/1.5 system-ui;max-width:1100px;margin:32px auto;padding:0 20px}pre{white-space:pre-wrap;overflow-wrap:anywhere}li{margin:8px 0}</style>' + body + '</html>', encoding="utf-8")
    atomic_json(directory / "campaign_status.json", {"report": str(directory / "index.html"), "studies": statuses,
                                                    "production_modified": False, "production_qualification": False})
    print("[NYX campagne] Rapport : " + str(directory / "index.html"), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("Screen", "Replay", "Challengers", "Archive", "Annual", "Summary"), required=True)
    parser.add_argument("--annual-repo", type=Path, required=True)
    parser.add_argument("--app-repo", type=Path, required=True)
    parser.add_argument("--base-session", type=Path)
    parser.add_argument("--session-dir", type=Path)
    parser.add_argument("--new-session", action="store_true")
    args = parser.parse_args(argv)
    preflight()
    from filelock import FileLock
    root_lock = Path(args.annual_repo) / "runs/nyx_improvement_campaign/.campaign.lock"
    root_lock.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(root_lock, timeout=0):
        directory, session = create_or_load(args.annual_repo, args.app_repo, args.base_session, args.session_dir, args.new_session)
        require(not directory.is_relative_to(Path(session["seed_run"]).resolve()), "Sortie de recherche dans le donneur interdit")
        print("[NYX campagne] Dossier : " + str(directory), flush=True)
        print("[NYX campagne] Phase : " + args.phase, flush=True)
        try:
            run_phase(args.phase, directory, session)
        except Exception as exc:
            atomic_json(directory / "last_error.json", {"phase": args.phase, "type": type(exc).__name__, "message": str(exc),
                        "at_utc": datetime.now(timezone.utc).isoformat()})
            raise
        finally:
            summary(directory)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError, ImportError) as exc:
        print("[NYX campagne] ARRET : " + str(exc), file=sys.stderr, flush=True)
        raise SystemExit(2)
