"""Emplacement du noyau scientifique, écritures atomiques et états locaux."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
PRODUCTION = ROOT / "production"
RUNTIME = PRODUCTION / "runtime"
STATE_PATHS = (
    "runs/nyx_daily_cpu_states",
    "runs/nyx_daily_refit_states",
    "data/pit/nyx_annual_hydro_captures",
    "data/pit/nyx_annual_exchange_captures",
    "data/pit/nyx_annual_jao_initial_live",
)


def write_json(path: Path, value: dict) -> None:
    """Publier un document entier par remplacement atomique du fichier."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                        allow_nan=False, default=str) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def verify_runtime() -> None:
    """Contrôler les sources scientifiques contre leur manifeste de référence."""
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
    """Réserver le runtime à un seul précontrôle ou calcul à la fois."""
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
    """Remplacer les cinq dossiers d'état par ceux de la source sélectionnée.

    Un dossier absent de la source est retiré de la destination pour éviter
    de mélanger des états provenant de deux snapshots différents.
    """
    for relative in STATE_PATHS:
        origin, target = source / relative, destination / relative
        if origin.is_symlink() or (origin.exists() and any(p.is_symlink() for p in origin.rglob("*"))):
            raise ValueError(f"Lien symbolique interdit dans un état : {origin}")
        if origin.exists() and not origin.is_dir():
            raise ValueError(f"Dossier d'état requis : {origin}")
        if target.is_symlink():
            raise ValueError(f"Destination d'état redirigée : {target}")
    for relative in STATE_PATHS:
        origin, target = source / relative, destination / relative
        if target.exists():
            shutil.rmtree(target)
        if origin.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(origin, target)
