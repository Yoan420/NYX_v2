"""Configuration des chemins et du stockage, indépendante des modèles."""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Mapping

from .runtime import ROOT

CONFIG_KEYS = {
    "input_root", "output_root", "input_s3_uri", "output_s3_uri", "collect_reporting_sources",
}


@dataclass(frozen=True)
class RunSettings:
    input_root: Path
    output_root: Path
    input_s3_uri: str | None = None
    output_s3_uri: str | None = None
    collect_reporting_sources: bool = True


def _path(value: str | Path, base: Path, name: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(f"{name} doit être un chemin non vide.")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _uri(value: str | None, name: str) -> str | None:
    if value is not None and not isinstance(value, str):
        raise ValueError(f"{name} doit être une URI S3 ou null.")
    return (value.strip() or None) if value is not None else None


def load_settings(
    config_path: Path | None = None,
    *,
    input_root: Path | None = None,
    output_root: Path | None = None,
    input_s3_uri: str | None = None,
    output_s3_uri: str | None = None,
    collect_reporting_sources: bool | None = None,
    environ: Mapping[str, str] | None = None,
) -> RunSettings:
    """Appliquer les options de commande, variables S3, JSON puis défauts.

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
    environment = os.environ if environ is None else environ
    input_uri = environment.get("NYX_INPUT_S3_URI", config.get("input_s3_uri"))
    output_uri = environment.get("NYX_OUTPUT_S3_URI", config.get("output_s3_uri"))
    if input_s3_uri is not None:
        input_uri = input_s3_uri
    if output_s3_uri is not None:
        output_uri = output_s3_uri
    collect = config.get("collect_reporting_sources", True)
    if collect_reporting_sources is not None:
        collect = collect_reporting_sources
    if not isinstance(collect, bool):
        raise ValueError("collect_reporting_sources doit être true ou false.")
    return RunSettings(
        **paths,
        input_s3_uri=_uri(input_uri, "input_s3_uri"),
        output_s3_uri=_uri(output_uri, "output_s3_uri"),
        collect_reporting_sources=collect,
    )
