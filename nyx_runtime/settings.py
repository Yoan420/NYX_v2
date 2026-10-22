"""Chemins locaux et options de reporting, indépendants des modèles."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from .runtime import ROOT

CONFIG_KEYS = {
    "input_root", "output_root", "collect_reporting_sources",
}


@dataclass(frozen=True)
class RunSettings:
    input_root: Path
    output_root: Path
    collect_reporting_sources: bool = True


def _path(value: str | Path, base: Path, name: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(f"{name} doit être un chemin non vide.")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def load_settings(
    config_path: Path | None = None,
    *,
    input_root: Path | None = None,
    output_root: Path | None = None,
    collect_reporting_sources: bool | None = None,
) -> RunSettings:
    """Appliquer les options de commande, JSON puis défauts.

    Les chemins du JSON sont relatifs à ce fichier ; ceux de la commande
    sont relatifs au dossier courant. Aucun paramètre scientifique n'est lu ici.
    """
    config = {}
    base = ROOT
    if config_path is not None:
        config_path = config_path.expanduser().resolve()
        config = json.loads(config_path.read_text(encoding="utf-8-sig"))
        if not isinstance(config, dict):
            raise ValueError("La configuration doit être un objet JSON.")
        unknown = config.keys() - CONFIG_KEYS
        if unknown:
            raise ValueError("Clés de configuration inconnues : " + ", ".join(sorted(unknown)))
        base = config_path.parent
    paths = {
        "input_root": _path(config["input_root"], base, "input_root")
            if "input_root" in config else ROOT / "snapshot",
        "output_root": _path(config["output_root"], base, "output_root")
            if "output_root" in config else ROOT / "output",
    }
    for name, value in (("input_root", input_root), ("output_root", output_root)):
        if value is not None:
            paths[name] = _path(value, Path.cwd(), name)
    collect = config.get("collect_reporting_sources", True)
    if collect_reporting_sources is not None:
        collect = collect_reporting_sources
    if not isinstance(collect, bool):
        raise ValueError("collect_reporting_sources doit être true ou false.")
    return RunSettings(
        **paths,
        collect_reporting_sources=collect,
    )
