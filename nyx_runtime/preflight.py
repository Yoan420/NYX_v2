"""Précontrôle des entrées et des états avant un calcul NYX.

Aucun fournisseur distant n’est contacté et aucun estimateur n’est chargé.
Les bibliothèques scientifiques sont importées seulement lors des contrôles
qui en ont besoin.
"""
from __future__ import annotations

from datetime import date
import importlib.metadata
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping

from .engine import (
    CHRONOS_REVISION,
    EXTENSIONS,
    FAMILIES,
    PROTOCOL,
    ZONES,
    activate_runtime,
    read_json,
    redact_error,
    resolve_directory,
    sha256_file,
    validate_request,
    verify_artifacts,
)


class LivePreflightError(ValueError):
    """A live request lacks required local inputs; no calculation was started."""

    def __init__(self, report: dict):
        self.report = report
        super().__init__("Précontrôle bloqué : " + "; ".join(report["blockers"]))


REQUIREMENTS = Path(__file__).resolve().parents[1] / "requirements.txt"


def _release_version(value: str) -> tuple[int, ...] | None:
    """Normaliser les versions finales épinglées, selon leur identité PEP 440.

    Les pins de cette recette sont des versions finales numériques. Un suffixe
    local valide (+cpu, par exemple) ne change pas la version publique. Les
    versions pre/dev/post et les epochs non nuls ne sont pas ces releases.
    """
    match = re.fullmatch(
        r"\s*[vV]?(?:0!)?(\d+(?:\.\d+)*)(?:\+[a-zA-Z0-9]+(?:[._-][a-zA-Z0-9]+)*)?\s*",
        str(value),
    )
    if match is None:
        return None
    release = tuple(int(part) for part in match.group(1).split("."))
    while len(release) > 1 and release[-1] == 0:
        release = release[:-1]
    return release


def pinned_requirements(path: Path | None = None) -> dict[str, str]:
    """Lire le contrat installé depuis requirements.txt, sans importer de paquet."""
    path = REQUIREMENTS if path is None else path
    expected = {}
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        match = re.fullmatch(r"([A-Za-z0-9][A-Za-z0-9._-]*)==([0-9]+(?:\.[0-9]+)*)", line)
        if match is None:
            raise ValueError(f"requirements.txt:{number} exige une version finale exacte package==version.")
        name, version = match.groups()
        normalized = re.sub(r"[-_.]+", "-", name).lower()
        if normalized in expected:
            raise ValueError(f"requirements.txt contient un pin dupliqué : {normalized}.")
        expected[normalized] = version
    if not expected:
        raise ValueError("requirements.txt ne contient aucun pin de dépendance.")
    if "tshistory-lite" in expected and expected["tshistory-lite"] != "0.5":
        raise ValueError("Le client Saturn privé doit conserver tshistory_lite==0.5.")
    expected["tshistory-lite"] = "0.5"
    return expected


def inspect_versions(expected: Mapping[str, str]) -> dict:
    """Comparer uniquement les métadonnées des distributions à la recette."""
    versions, mismatches, missing = {}, [], []
    for distribution, wanted in expected.items():
        try:
            observed = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            observed = None
            missing.append(distribution)
        versions[distribution] = observed
        if observed is None or _release_version(observed) != _release_version(wanted):
            mismatches.append({"distribution": distribution, "expected": wanted, "observed": observed})
    return {"passed": not mismatches, "expected_versions": dict(expected),
            "installed_versions": versions, "mismatches": mismatches,
            "missing_distributions": missing, "metadata_only": True}


def check_python_version() -> dict:
    observed = ".".join(str(part) for part in sys.version_info[:3])
    if sys.version_info[:2] != (3, 11):
        raise ValueError(f"Python observé {observed} ; attendu 3.11.x. Recréer l'environnement avec Python 3.11.")
    return {"observed": observed, "expected": "3.11.x"}


def validate_seed(bundle: Path, output: Path, first_day: str) -> dict:
    """Vérifie les reçus, modèles et poids du calcul initial utilisé comme seed."""
    receipt = read_json(output / "receipt.json")
    if receipt.get("status") != "REPORTS_COMPLETE" or receipt.get("future_labels_used") is not False:
        raise ValueError("seed_output doit être un rapport annuel complet, sans labels futurs.")
    if date.fromisoformat(receipt["delivery_day"]) > date.fromisoformat(first_day):
        raise ValueError("Le seed est postérieur à la première livraison demandée.")
    counts = {"input_artifacts": verify_artifacts(bundle, receipt.get("input_sha256"), "Entrées seed")}
    audits = receipt.get("model_audit", {})
    models = {}
    for family in FAMILIES:
        models[f"models/{family}.cbm"] = audits["price_experts"][family]["model"]["sha256"]
    for zone in ZONES:
        models[f"models/negative_{zone}.cbm"] = audits["negative_countries"][zone]["model"]["sha256"]
    counts["saved_final_models"] = verify_artifacts(output, models, "Modèles intermédiaires seed")
    for group in ("saturn", "auction_prices", "fuel", "thermal_capacity", "jao_initial",
                  "public_hydro", "lagged_exchange", "nyx_quantiles", "scarcity_confirmed_pair"):
        document = read_json(bundle / "source_receipts" / f"{group}.json")
        if document.get("source_group") != group or document.get("state") != "COMPLETE":
            raise ValueError(f"Reçu source seed incomplet : {group}")
        counts[group] = verify_artifacts(bundle, document.get("artifact_sha256"), f"Seed {group}")
    identities = []
    for zone in ZONES:
        audit = read_json(bundle / "baseline_audits" / f"{zone}.json")
        identities.append(audit.get("chronos_model_identity"))
    if not identities[0] or any(identity != identities[0] for identity in identities):
        raise ValueError("Les quatre baselines seed doivent utiliser les mêmes poids Chronos.")
    identity = identities[0]
    if (identity.get("model_id") != "amazon/chronos-2" or identity.get("revision") != CHRONOS_REVISION
            or identity.get("device") != "cpu" or identity.get("dtype") != "torch.float32"):
        raise ValueError("Le snapshot Chronos du seed diffère de la recette CPU retenue.")
    files = identity.get("files")
    if (not isinstance(files, dict) or "config.json" not in files
            or not any(name.endswith((".safetensors", ".bin")) for name in files)):
        raise ValueError("Inventaire des poids Chronos absent du seed.")
    return {"receipt_sha256": sha256_file(output / "receipt.json"), "counts": counts,
            "chronos_identity": identity}


def inspect_live(config: Mapping[str, Any], first_day: str, cutoff: Any) -> dict:
    """Return a JSON-compatible, read-only local preflight; no network or fit."""
    report = {"protocol": PROTOCOL, "status": "BLOCKED", "ready": False,
              "first_day": str(first_day), "blockers": [], "checks": {},
              "live_sources_checked": False, "models_loaded": False,
              "qualified_for_production": False,
              "notes": ["Les vintages Saturn sont vérifiés pendant la collecte amont, avant les fits.",
                        "Les absences publiques autorisées gardent les indicateurs de la recette originale."]}

    def check(name, callback):
        try:
            value = callback()
            report["checks"][name] = {"passed": True, "detail": value}
            return value
        except Exception as error:
            message = redact_error(error)
            report["checks"][name] = {"passed": False, "reason": message}
            report["blockers"].append(f"{name} : {message}")
            return None

    check("python_version", check_python_version)
    request = check("request", lambda: validate_request(first_day, cutoff))
    def saturn_author():
        if not os.environ.get("SATURN_AUTHOR", "").strip():
            raise ValueError("Définir SATURN_AUTHOR avec votre identifiant professionnel avant la prévision.")
        return {"configured": True, "network_access_checked": False}
    check("saturn_author", saturn_author)
    def fixed_cutoff_context():
        if "NYX_ANNUAL_OPERATIONAL_CONTEXT" in os.environ:
            raise ValueError("Retirer NYX_ANNUAL_OPERATIONAL_CONTEXT : ce lanceur utilise la coupure fixe de 08 h Paris.")
        return {"shared_cutoff_override": False}
    check("fixed_cutoff_context", fixed_cutoff_context)
    if request:
        first_day, stamp = request
        report["cutoff_utc"] = stamp.isoformat()
        report["checks"]["request"]["detail"] = {"first_day": first_day, "cutoff_utc": stamp.isoformat()}
    paths = {key: check(key, lambda key=key: resolve_directory(config, key))
             for key in ("repo_root", "seed_bundle", "seed_output")}
    for key, path in paths.items():
        if path is not None:
            report["checks"][key]["detail"] = str(path)
    repo, bundle, output = (paths[key] for key in ("repo_root", "seed_bundle", "seed_output"))

    def runtime_files():
        files = [f"annual_extension/{name}.py" for name in EXTENSIONS]
        files += ["chronos2_hourly/__init__.py", "chronos2_hourly/nyx_annual_live_preflight.py",
                  "chronos2_hourly/nyx_annual_cpu_baseline.py", "config/nyx_annual_cpu_ordered_features.json",
                  "config/nyx_historical_hgb_feature_schemas.json",
                  "config/kalman_operational.yaml", "run_nyx_annual_auction_prices_source.py",
                  "run_nyx_annual_hydro_source.py", "run_nyx_annual_exchange_source.py"]
        missing = [name for name in files if not (repo / name).is_file()]
        if missing:
            raise ValueError("Fichiers du runtime absents : " + ", ".join(missing))
        return {"files_present": len(files)}
    if repo:
        check("runtime_files", runtime_files)

    expected = check("dependency_manifest", pinned_requirements)
    dependencies = inspect_versions(expected) if expected else {"passed": False, "installed_versions": {}}
    report["installed_versions"] = dependencies["installed_versions"]
    report["checks"]["dependencies"] = dependencies
    for mismatch in dependencies.get("mismatches", []):
        name, wanted, observed = mismatch["distribution"], mismatch["expected"], mismatch["observed"]
        remedy = ("Installer le wheel Saturn approuvé tshistory_lite==0.5."
                  if name == "tshistory-lite" else "Réinstaller les versions de requirements.txt.")
        report["blockers"].append(
            f"dependencies : {name} observé {observed or 'absent'} ; attendu {wanted}. {remedy}"
        )
    seed = check("seed_artifacts", lambda: validate_seed(bundle, output, first_day)) if bundle and output and request else None
    if seed:
        report["checks"]["seed_artifacts"]["detail"] = {
            "receipt_sha256": seed["receipt_sha256"], "counts": seed["counts"]}

    if seed and dependencies["passed"]:
        def model_cache():
            from huggingface_hub import snapshot_download
            local = Path(snapshot_download("amazon/chronos-2", revision=CHRONOS_REVISION,
                                           local_files_only=True)).resolve()
            count = verify_artifacts(local, seed["chronos_identity"]["files"], "Cache Chronos local")
            return {"snapshot": str(local), "revision": CHRONOS_REVISION, "verified_files": count,
                    "download_requested": False}
        check("chronos_local_cache", model_cache)

    if not report["blockers"]:
        def continuation():
            activate_runtime(repo)
            from chronos2_hourly.nyx_annual_live_preflight import ROOT
            if Path(ROOT).resolve() != repo:
                raise ValueError("Le runtime importé ne correspond pas à repo_root.")
            from chronos2_hourly.nyx_daily_state_cache import build_contract, load_state
            from chronos2_hourly.nyx_daily_two_day_baseline import MAX_BRIDGE_DAYS
            contract = build_contract(bundle, output, threads=2)
            cache = repo / "runs/nyx_daily_cpu_states" / contract["seed_report_receipt_sha256"]
            state = load_state(cache, contract, first_delivery_day=first_day, cutoff=stamp)
            anchor = state["last_history_day"] if state else read_json(
                bundle / "source_receipts/nyx_quantiles.json")["delivery_day"]
            gap = (date.fromisoformat(first_day) - date.fromisoformat(anchor)).days - 1
            if not 0 <= gap <= MAX_BRIDGE_DAYS:
                raise ValueError("État quotidien compatible manquant : "
                                 f"{gap} jours à reconstruire, maximum {MAX_BRIDGE_DAYS}.")
            return {"anchor_day": anchor, "bridge_days": gap, "rolling_state": bool(state),
                    "cache_root": str(cache)}
        check("daily_continuation", continuation)
    report["ready"] = not report["blockers"]
    report["status"] = "LOCAL_INPUTS_READY" if report["ready"] else "BLOCKED"
    return report
