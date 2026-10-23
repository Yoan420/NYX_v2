"""Contrats communs du moteur NYX : fichiers, reçus et chargement du noyau.

Le noyau scientifique reste sous ses noms de modules historiques pour préserver
les modèles sérialisés et les empreintes des états déjà produits.
"""
from __future__ import annotations

from datetime import date, timedelta
import hashlib
import importlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import sys
from typing import Any, Mapping


PROTOCOL = "nyx_production_daily_refit_csv_v1"

ZONES = ("FR", "DE", "BE", "NL")

CHRONOS_REVISION = "29ec3766d36d6f73f0696f85560a422f50e8498c"

FAMILIES = ("fr_residual_1000", "cwe_residual_2000", "cwe_absolute_2000")

EXTENSIONS = (
    "nyx_daily_refit", "nyx_daily_refit_experts", "nyx_daily_refit_history",
    "nyx_daily_refit_reference", "nyx_daily_state_cache", "nyx_daily_two_day_baseline",
    "nyx_saved_horizon_inference", "nyx_saved_horizon_sources",
    "nyx_saved_horizon_resume", "nyx_saved_horizon_reporting_sources",
)

def sha256_file(path: Path) -> str:
    """Calcule le SHA-256 d’un fichier sans le charger en mémoire."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def redact_error(error: object) -> str:
    """Masque les secrets usuels dans un diagnostic avant publication."""
    message = str(error)
    message = re.sub(r"(https?://)[^/@\s]+:[^/@\s]+@", r"\1[masqué]@", message)
    message = re.sub(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+", "Bearer [masqué]", message)
    message = re.sub(r"(?i)((?:password|token|authorization|secret|api[_-]?key)\s*[=:]\s*)[^\s,;]+",
                     r"\1[masqué]", message)
    return message[:2000]


def read_json(path: Path) -> dict:
    """Lit un reçu JSON local et refuse les fichiers absents ou redirigés."""
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Fichier JSON requis absent ou redirigé : {path}")
    with path.open(encoding="utf-8-sig") as handle:
        document = json.load(handle)
    if not isinstance(document, dict):
        raise ValueError(f"Objet JSON requis : {path}")
    return document


def resolve_directory(config: Mapping[str, Any], key: str) -> Path:
    """Résout un dossier explicitement configuré par le lanceur."""
    raw = config.get(key)
    if not isinstance(raw, (str, os.PathLike)) or not str(raw).strip():
        raise ValueError(f"Configurer explicitement {key}.")
    value = Path(raw).expanduser()
    if not value.is_absolute():
        raise ValueError(f"{key} doit être un chemin absolu résolu par le lanceur.")
    value = value.resolve()
    if not value.is_dir():
        raise ValueError(f"Dossier requis absent : {key} = {value}")
    return value


def verify_artifacts(root: Path, hashes: Any, label: str) -> int:
    """Vérifie les chemins et empreintes de tous les artefacts d’un reçu."""
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError(f"{label} : inventaire d'empreintes absent.")
    for name, expected in hashes.items():
        if (not isinstance(name, str) or not name or "\\" in name
                or PurePosixPath(name).is_absolute() or PureWindowsPath(name).drive
                or ".." in PurePosixPath(name).parts
                or not re.fullmatch(r"[0-9a-f]{64}", str(expected))):
            raise ValueError(f"{label} : identité d'artefact invalide.")
        target = root / name
        if (not target.resolve().is_relative_to(root) or not target.is_file()
                or target.is_symlink() or sha256_file(target) != expected):
            raise ValueError(f"{label} : artefact absent ou empreinte différente : {name}")
    return len(hashes)


def validate_request(first_day: str, cutoff: Any) -> tuple[str, Any]:
    """Valide les deux livraisons issues de la coupure commune J 08 h Paris."""
    import pandas as pd
    day = date.fromisoformat(str(first_day))
    if day.isoformat() != str(first_day):
        raise ValueError("first_day doit utiliser YYYY-MM-DD.")
    stamp = pd.Timestamp(cutoff)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError("La coupure doit porter un fuseau explicite.")
    stamp = stamp.tz_convert("UTC")
    expected = pd.Timestamp(f"{day - timedelta(days=1)} 08:00", tz="Europe/Paris").tz_convert("UTC")
    if stamp != expected:
        raise ValueError("La recette exige exactement D-1 08:00 Europe/Paris pour les deux jours.")
    if stamp > pd.Timestamp.now(tz="UTC"):
        raise ValueError("La coupure demandée n'est pas encore atteinte.")
    return day.isoformat(), stamp


def activate_runtime(repo: Path):
    """Expose retained extension files under their original package namespace."""
    for name in ("chronos2_hourly", "chronos2_modular", "experiment_console", "marginal_cost_expert"):
        loaded = sys.modules.get(name)
        if loaded is not None:
            origin = getattr(loaded, "__file__", None)
            if origin and not Path(origin).resolve().is_relative_to(repo):
                raise RuntimeError(f"Un autre runtime {name} est déjà importé ; lancer un nouveau processus.")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    package = importlib.import_module("chronos2_hourly")
    extension = str(repo / "annual_extension")
    if extension not in package.__path__:
        package.__path__.append(extension)
    return package
