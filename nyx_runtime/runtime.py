"""Emplacement du noyau scientifique, écritures atomiques et états locaux."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
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
    # Un rendu absent doit bloquer avant de lancer les entraînements coûteux.
    reporting = json.loads((PRODUCTION / "REPORTING_MANIFEST.json").read_text(encoding="utf-8"))
    for name, expected in reporting["files"].items():
        relative = PurePosixPath(name)
        path = PRODUCTION / "reporting" / relative
        if (relative.is_absolute() or ".." in relative.parts or path.is_symlink()
                or not path.is_file() or not path.resolve().is_relative_to((PRODUCTION / "reporting").resolve())
                or hashlib.sha256(path.read_bytes()).hexdigest() != expected):
            raise ValueError(f"Source de reporting absente ou modifiée : {relative}")
    if not (ROOT / "nyx_runtime/templates/report.html").is_file():
        raise ValueError("Modèle HTML CWE absent : nyx_runtime/templates/report.html")


@contextmanager
def runtime_lock():
    """Réserver le runtime à un seul précontrôle ou calcul à la fois."""
    root = _checked_root(RUNTIME)
    path = root / "runs/.nyx_cloud.lock"
    _check_location(root, path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _check_location(root, path)
    try:
        handle = path.open("x", encoding="utf-8")
    except FileExistsError as error:
        raise ValueError(f"Runtime déjà verrouillé : {path}. Vérifier le processus avant de retirer le verrou.") from error
    try:
        with handle:
            json.dump({"pid": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat()}, handle)
        yield
    finally:
        _check_location(root, path)
        path.unlink(missing_ok=True)


def _redirected(path: Path) -> bool:
    """Détecter aussi les jonctions Windows sur Python 3.11."""
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    return (stat.S_ISLNK(info.st_mode)
            or getattr(info, "st_reparse_tag", 0) in (0xA0000003, 0xA000000C))


def _checked_root(path: Path) -> Path:
    path = Path(path).absolute()
    # Contrôler les composants avant resolve(), qui effacerait les alias.
    for candidate in (path, *path.parents):
        if _redirected(candidate):
            raise ValueError(f"Racine ou parent redirigé par lien/jonction : {candidate}")
    return path.resolve()


def _check_location(root: Path, path: Path) -> None:
    if not path.is_relative_to(root) or not path.resolve().is_relative_to(root):
        raise ValueError(f"Chemin d'état hors de la racine autorisée : {path}")
    for candidate in (path, *path.parents):
        if _redirected(candidate):
            raise ValueError(f"Chemin d'état redirigé par lien/jonction : {candidate}")
        if candidate == root:
            break


def _check_tree(root: Path, directory: Path) -> None:
    """Examiner les entrées sans traverser de lien ou de jonction."""
    _check_location(root, directory)
    if not directory.exists():
        return
    if not directory.is_dir():
        raise ValueError(f"Dossier d'état requis : {directory}")
    pending = [directory]
    while pending:
        current = pending.pop()
        with os.scandir(current) as entries:
            for entry in entries:
                path = Path(entry.path)
                _check_location(root, path)
                if entry.is_dir(follow_symlinks=False):
                    pending.append(path)
                elif not entry.is_file(follow_symlinks=False):
                    raise ValueError(f"Fichier d'état non régulier : {path}")


def copy_state(source: Path, destination: Path) -> None:
    """Remplacer les cinq dossiers d'état par ceux de la source sélectionnée.

    Un dossier absent de la source est retiré de la destination pour éviter
    de mélanger des états provenant de deux snapshots différents.
    """
    source, destination = _checked_root(source), _checked_root(destination)
    if (source == destination or source.is_relative_to(destination)
            or destination.is_relative_to(source)):
        raise ValueError("Les racines des états doivent être distinctes et non imbriquées.")
    # Tout valider avant la première suppression, y compris les parents et
    # les jonctions internes qu'un parcours récursif pourrait sinon suivre.
    for relative in STATE_PATHS:
        _check_tree(source, source / relative)
        _check_tree(destination, destination / relative)
    for relative in STATE_PATHS:
        origin, target = source / relative, destination / relative
        _check_location(source, origin)
        _check_location(destination, target)
        if target.exists():
            shutil.rmtree(target)
        if origin.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(origin, target)
