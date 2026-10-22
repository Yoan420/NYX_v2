"""Run the retained, unqualified NYX daily refit recipe for two civil days.

``inspect_live`` reads local files and metadata only.  It never contacts a
provider, downloads weights, loads a saved estimator, or starts a fit.
``run_live`` is the explicit calculation entry point.  It uses the source
recipe unchanged and publishes CSV only after all four forecasts validate.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import hashlib
import importlib
import importlib.metadata
import importlib.util
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
DEPENDENCIES = {
    "numpy": "numpy", "pandas": "pandas", "pyarrow": "pyarrow",
    "catboost": "catboost", "sklearn": "scikit-learn", "scipy": "scipy",
    "joblib": "joblib", "holidays": "holidays", "pykalman": "pykalman",
    "torch": "torch", "chronos": "chronos-forecasting",
    "huggingface_hub": "huggingface-hub", "transformers": "transformers",
    "safetensors": "safetensors", "accelerate": "accelerate", "einops": "einops",
    "requests": "requests", "httpx": "httpx", "yaml": "PyYAML",
    "threadpoolctl": "threadpoolctl", "psutil": "psutil", "filelock": "filelock",
    "tshistory_lite": "tshistory_lite",
}
CANONICAL_COLUMNS = (
    "delivery_day", "zone", "delivery_start_utc", "delivery_start_local",
    "forecast_origin_utc", "horizon_day", "prediction_eur_mwh",
    "lower_bound_eur_mwh", "upper_bound_eur_mwh", "storm_eur_mwh",
    "observed_eur_mwh", "interval_method", "nominal_interval_level",
    "interval_calibrated", "source_html_sha256", "source_generated_at_utc",
    "source_html_name",
)


class LivePreflightError(ValueError):
    """A live request lacks required local inputs; no calculation was started."""

    def __init__(self, report: dict):
        self.report = report
        super().__init__("Précontrôle bloqué : " + "; ".join(report["blockers"]))


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _redact(error: object) -> str:
    message = str(error)
    message = re.sub(r"(https?://)[^/@\s]+:[^/@\s]+@", r"\1[masqué]@", message)
    message = re.sub(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+", "Bearer [masqué]", message)
    message = re.sub(r"(?i)((?:password|token|authorization|secret|api[_-]?key)\s*[=:]\s*)[^\s,;]+",
                     r"\1[masqué]", message)
    return message[:2000]


def _json(path: Path) -> dict:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Fichier JSON requis absent ou redirigé : {path}")
    with path.open(encoding="utf-8-sig") as handle:
        document = json.load(handle)
    if not isinstance(document, dict):
        raise ValueError(f"Objet JSON requis : {path}")
    return document


def _path(config: Mapping[str, Any], key: str) -> Path:
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


def _verify_manifest(root: Path, hashes: Any, label: str) -> int:
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
                or target.is_symlink() or _sha(target) != expected):
            raise ValueError(f"{label} : artefact absent ou empreinte différente : {name}")
    return len(hashes)


def _request(first_day: str, cutoff: Any) -> tuple[str, Any]:
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


def _activate_runtime(repo: Path):
    """Expose retained extension files under their original package namespace."""
    for name in ("chronos2_hourly", "chronos2_modular", "experiment_console"):
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


def _seed_checks(bundle: Path, output: Path, first_day: str) -> dict:
    receipt = _json(output / "receipt.json")
    if receipt.get("status") != "REPORTS_COMPLETE" or receipt.get("future_labels_used") is not False:
        raise ValueError("seed_output doit être un rapport annuel complet, sans labels futurs.")
    if date.fromisoformat(receipt["delivery_day"]) > date.fromisoformat(first_day):
        raise ValueError("Le seed est postérieur à la première livraison demandée.")
    counts = {"input_artifacts": _verify_manifest(bundle, receipt.get("input_sha256"), "Entrées seed")}
    audits = receipt.get("model_audit", {})
    models = {}
    for family in FAMILIES:
        models[f"models/{family}.cbm"] = audits["price_experts"][family]["model"]["sha256"]
    for zone in ZONES:
        models[f"models/negative_{zone}.cbm"] = audits["negative_countries"][zone]["model"]["sha256"]
    counts["saved_final_models"] = _verify_manifest(output, models, "Modèles intermédiaires seed")
    for group in ("saturn", "auction_prices", "fuel", "thermal_capacity", "jao_initial",
                  "public_hydro", "lagged_exchange", "nyx_quantiles", "scarcity_confirmed_pair"):
        document = _json(bundle / "source_receipts" / f"{group}.json")
        if document.get("source_group") != group or document.get("state") != "COMPLETE":
            raise ValueError(f"Reçu source seed incomplet : {group}")
        counts[group] = _verify_manifest(bundle, document.get("artifact_sha256"), f"Seed {group}")
    identities = []
    for zone in ZONES:
        audit = _json(bundle / "baseline_audits" / f"{zone}.json")
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
    return {"receipt_sha256": _sha(output / "receipt.json"), "counts": counts,
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
            message = _redact(error)
            report["checks"][name] = {"passed": False, "reason": message}
            report["blockers"].append(f"{name} : {message}")
            return None

    request = check("request", lambda: _request(first_day, cutoff))
    if request:
        first_day, stamp = request
        report["cutoff_utc"] = stamp.isoformat()
        report["checks"]["request"]["detail"] = {"first_day": first_day, "cutoff_utc": stamp.isoformat()}
    paths = {key: check(key, lambda key=key: _path(config, key))
             for key in ("repo_root", "seed_bundle", "seed_output")}
    for key, path in paths.items():
        if path is not None:
            report["checks"][key]["detail"] = str(path)
    repo, bundle, output = (paths[key] for key in ("repo_root", "seed_bundle", "seed_output"))

    def runtime_files():
        files = [f"annual_extension/{name}.py" for name in EXTENSIONS]
        files += ["chronos2_hourly/__init__.py", "chronos2_hourly/nyx_annual_live_preflight.py",
                  "chronos2_hourly/nyx_annual_cpu_baseline.py", "config/nyx_annual_cpu_ordered_features.json",
                  "config/kalman_operational.yaml", "run_nyx_annual_auction_prices_source.py",
                  "run_nyx_annual_hydro_source.py", "run_nyx_annual_exchange_source.py"]
        missing = [name for name in files if not (repo / name).is_file()]
        if missing:
            raise ValueError("Fichiers du runtime absents : " + ", ".join(missing))
        return {"files_present": len(files)}
    if repo:
        check("runtime_files", runtime_files)

    versions, missing = {}, []
    for module, distribution in DEPENDENCIES.items():
        try:
            present = importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            present = False
        if not present:
            missing.append(module)
        else:
            try:
                versions[distribution] = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError:
                versions[distribution] = "module présent ; distribution non identifiée"
    report["installed_versions"] = versions
    report["checks"]["dependencies"] = {"passed": not missing, "missing_modules": missing}
    if missing:
        report["blockers"].append("dependencies : modules absents : " + ", ".join(missing))
    seed = check("seed_artifacts", lambda: _seed_checks(bundle, output, first_day)) if bundle and output and request else None
    if seed:
        report["checks"]["seed_artifacts"]["detail"] = {
            "receipt_sha256": seed["receipt_sha256"], "counts": seed["counts"]}

    if seed and "huggingface_hub" not in missing:
        def model_cache():
            from huggingface_hub import snapshot_download
            local = Path(snapshot_download("amazon/chronos-2", revision=CHRONOS_REVISION,
                                           local_files_only=True)).resolve()
            count = _verify_manifest(local, seed["chronos_identity"]["files"], "Cache Chronos local")
            return {"snapshot": str(local), "revision": CHRONOS_REVISION, "verified_files": count,
                    "download_requested": False}
        check("chronos_local_cache", model_cache)

    if not report["blockers"]:
        def continuation():
            _activate_runtime(repo)
            from chronos2_hourly.nyx_annual_live_preflight import ROOT
            if Path(ROOT).resolve() != repo:
                raise ValueError("Le runtime importé ne correspond pas à repo_root.")
            from chronos2_hourly.nyx_daily_state_cache import build_contract, load_state
            from chronos2_hourly.nyx_daily_two_day_baseline import MAX_BRIDGE_DAYS
            contract = build_contract(bundle, output, threads=2)
            cache = repo / "runs/nyx_daily_cpu_states" / contract["seed_report_receipt_sha256"]
            state = load_state(cache, contract, first_delivery_day=first_day, cutoff=stamp)
            anchor = state["last_history_day"] if state else _json(
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


def _write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False, default=str)
        handle.write("\n")


def _canonical_forecasts(forecasts: Mapping[str, Any], first_day: str, cutoff: Any,
                         comparisons: Mapping[str, Any] | None = None):
    """Preserve native quantiles; display only the source adapter's envelope."""
    import numpy as np
    import pandas as pd
    first = date.fromisoformat(first_day)
    expected = pd.date_range(first_day, str(first + timedelta(days=2)),
                             tz="Europe/Paris", freq="h", inclusive="left").tz_convert("UTC")
    if set(forecasts) != set(ZONES):
        raise ValueError("Les quatre pays FR/DE/BE/NL sont obligatoires.")
    frames = []
    for zone in ZONES:
        frame = forecasts[zone]
        required = {"price_eur_mwh", "p_negative", "p_negative_raw", "reference_eur_mwh",
                    "nyx__q10", "nyx__q50", "nyx__q90", "forecast_origin_utc", "horizon_day"}
        if (not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique
                or not required <= set(frame) or not isinstance(frame.index, pd.DatetimeIndex)
                or str(frame.index.tz) != "UTC" or not frame.index.equals(expected)):
            raise ValueError(f"{zone} : grille physique J+1/J+2 ou colonnes natives incomplètes.")
        if any(str(name).lower() in {"actual", "target", "observed"} for name in frame):
            raise ValueError(f"{zone} : labels observés interdits dans les prévisions futures.")
        numeric = frame[list(required - {"forecast_origin_utc"})].to_numpy(dtype=float)
        if not np.isfinite(numeric).all():
            raise ValueError(f"{zone} : prévisions natives absentes ou non finies.")
        quantiles = frame[["nyx__q10", "nyx__q50", "nyx__q90"]].to_numpy(dtype=float)
        if (np.diff(quantiles, axis=1) < 0).any():
            raise ValueError(f"{zone} : quantiles natifs croisés.")
        if not frame[["p_negative", "p_negative_raw"]].ge(0).all().all() or not frame[["p_negative", "p_negative_raw"]].le(1).all().all():
            raise ValueError(f"{zone} : probabilités natives hors de [0,1].")
        origins = pd.to_datetime(frame["forecast_origin_utc"], utc=True, errors="raise")
        if origins.isna().any() or not (origins == pd.Timestamp(cutoff)).all():
            raise ValueError(f"{zone} : origine différente de la coupure commune.")
        local = expected.tz_convert("Europe/Paris")
        horizon = [(stamp.date() - first).days + 1 for stamp in local]
        if list(frame["horizon_day"].astype(float)) != horizon:
            raise ValueError(f"{zone} : horizon_day incohérent avec les deux dates physiques.")
        point = frame["price_eur_mwh"].to_numpy(dtype=float)
        values = {"delivery_day": [stamp.date().isoformat() for stamp in local], "zone": zone,
                  "delivery_start_utc": [stamp.isoformat() for stamp in expected],
                  "delivery_start_local": [stamp.isoformat() for stamp in local],
                  "forecast_origin_utc": [pd.Timestamp(cutoff).isoformat()] * len(expected),
                  "horizon_day": horizon, "prediction_eur_mwh": point,
                  "lower_bound_eur_mwh": np.minimum(quantiles[:, 0], point),
                  "upper_bound_eur_mwh": np.maximum(quantiles[:, 2], point),
                  "storm_eur_mwh": np.nan, "observed_eur_mwh": np.nan,
                  "interval_method": "baseline_envelope_uncalibrated", "nominal_interval_level": .8,
                  "interval_calibrated": False, "source_html_sha256": "",
                  "source_generated_at_utc": "", "source_html_name": ""}
        if comparisons:
            for group, column in (("observed", "observed_eur_mwh"), ("storm", "storm_eur_mwh")):
                series = comparisons.get(group, {}).get(zone)
                if series is not None:
                    selected = series.reindex(expected).to_numpy(dtype=float)
                    if np.isinf(selected).any():
                        raise ValueError(f"{zone} : comparaison de reporting infinie.")
                    values[column] = selected
        frames.append(pd.DataFrame(values, columns=CANONICAL_COLUMNS))
    return pd.concat(frames, ignore_index=True)


def run_live(config: Mapping[str, Any], first_day: str, cutoff: Any, output_dir: Any) -> dict:
    """Explicitly refit, export untouched native forecasts and canonical CSV.

    The caller owns the unique output directory and its status file.  Absolute
    upstream audit paths remain stable: the directory must not be renamed.
    The returned document contains paths and metadata, never DataFrames.
    """
    preflight = inspect_live(config, first_day, cutoff)
    if not preflight["ready"]:
        raise LivePreflightError(preflight)
    first_day, stamp = _request(first_day, cutoff)
    repo, bundle, seed_output = (_path(config, key) for key in ("repo_root", "seed_bundle", "seed_output"))
    output = Path(output_dir).expanduser().resolve()
    for protected in (repo, bundle, seed_output):
        if output == protected or protected.is_relative_to(output):
            raise ValueError("Le dossier de sortie ne peut contenir le runtime ni les seeds.")
    for protected in (bundle, seed_output):
        if output.is_relative_to(protected):
            raise ValueError("Le dossier de sortie doit être distinct des seeds immuables.")
    reserved = ("forecast_extension", "native_forecasts", "predictions.csv", "live_audit.json", "live_result.json")
    if any((output / name).exists() for name in reserved):
        raise ValueError("Sortie live déjà commencée : choisir un nouvel identifiant de lancement.")
    context = config.get("context")
    if context is not None and not isinstance(context, dict):
        raise ValueError("context doit être un objet explicite fourni par l'appelant, ou absent.")
    collect_reporting = config.get("collect_reporting_sources", True)
    if not isinstance(collect_reporting, bool):
        raise ValueError("collect_reporting_sources doit être un booléen.")
    output.mkdir(parents=True, exist_ok=True)
    _activate_runtime(repo)
    producer = importlib.import_module("chronos2_hourly.nyx_daily_refit")
    try:
        result = producer.produce_refit_horizon(repo, bundle, seed_output, first_day, stamp,
                                                output / "forecast_extension", context=context)
    except Exception as error:
        raise RuntimeError("Refit NYX interrompu : " + _redact(error)) from error
    # First validation happens before the optional, strictly report-only queries.
    _canonical_forecasts(result["forecasts"], first_day, stamp)
    audit = result.get("model_audit", {})
    if (audit.get("daily_refit_completed") is not True or audit.get("future_labels_used") is not False
            or audit.get("refit_all_trainable_nyx_models") is not True
            or audit.get("seed_inference_used_for_final_forecast") is not False
            or audit.get("final_model_refit_attempts") != 7
            or audit.get("reference", {}).get("current_fit_counts") != {"hgb": 12, "test2_pairs": 2}):
        raise ValueError("Le producteur n'atteste pas le réentraînement quotidien complet attendu.")
    refit_receipt = output / "forecast_extension/daily_refit/receipt.json"
    sealed = _json(refit_receipt)
    if (sealed.get("protocol") != "nyx_daily_model_refit_v1"
            or sealed.get("status") != "MODELS_COMPLETE" or sealed.get("future_labels_used") is not False
            or sealed.get("first_delivery_day") != first_day or sealed.get("forecast_days") != 2
            or sealed.get("shared_cutoff_utc") != stamp.isoformat()
            or sealed.get("new_forecast_refit_required") is not True):
        raise ValueError("Reçu du refit complet absent ou invalide.")
    _verify_manifest(refit_receipt.parent, sealed.get("artifact_sha256"), "Prévisions et modèles fraîchement scellés")
    comparisons = None
    reporting_audit = {"status": "not_requested", "used_for_prediction": False, "used_for_training": False}
    reporting_receipt = None
    if collect_reporting:
        try:
            import pandas as pd
            helper = importlib.import_module("chronos2_hourly.nyx_saved_horizon_reporting_sources")
            index = result["baseline"][ZONES[0]].index
            future = result["forecasts"][ZONES[0]].index
            expected = pd.date_range(min(index[0], future[0]),
                                     max(index[-1], future[-1]), freq="h", tz="UTC")
            comparisons = helper.collect_horizon_reporting_sources(repo, first_day, output, expected)
            reporting_audit = comparisons["audit"]
            reporting_receipt = comparisons.get("receipt_path")
        except Exception as error:
            reporting_audit = {"status": "unavailable", "reason": _redact(error),
                               "used_for_prediction": False, "used_for_training": False,
                               "fallback_to_model_inputs": False}
    canonical = _canonical_forecasts(result["forecasts"], first_day, stamp, comparisons)
    native = output / "native_forecasts"
    native.mkdir(exist_ok=False)
    native_paths = {}
    for zone in ZONES:
        path = native / f"{zone}.csv"
        result["forecasts"][zone].to_csv(path, index=True, index_label="timestamp_utc", encoding="utf-8")
        native_paths[zone] = str(path)
    canonical_path = output / "predictions.csv"
    canonical.to_csv(canonical_path, index=False, encoding="utf-8")
    audit_path = output / "live_audit.json"
    _write_json(audit_path, {"protocol": PROTOCOL, "model_audit": audit,
                            "sources_audit": result.get("sources_audit", {}),
                            "reporting_audit": reporting_audit,
                            "interval_method": "baseline_envelope_uncalibrated",
                            "final_price_quantiles_available": False,
                            "native_baseline_quantiles_preserved": True,
                            "qualified_for_production": False})
    document = {"protocol": PROTOCOL, "status": "CSV_COMPLETE", "first_day": first_day,
                "delivery_days": [first_day, (date.fromisoformat(first_day) + timedelta(days=1)).isoformat()],
                "cutoff_utc": stamp.isoformat(), "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "rows": len(canonical), "countries": list(ZONES), "qualified_for_production": False,
                "predictions_csv": str(canonical_path), "forecasts_csv": native_paths,
                "csv_files": [str(canonical_path), *native_paths.values()],
                "forecast_extension_dir": str(output / "forecast_extension"),
                "model_audit_path": str(refit_receipt.parent / "model_audit.json"),
                "source_audit_path": str(output / "forecast_extension/horizon_sources/receipt.json"),
                "reporting_audit_path": reporting_receipt, "audit_path": str(audit_path),
                "refit_receipt_path": str(refit_receipt), "receipt_path": str(refit_receipt),
                "baseline_receipt_path": str(output / "forecast_extension/baseline_extension/receipt.json"),
                "native_columns": {zone: list(result["forecasts"][zone].columns) for zone in ZONES},
                "output_sha256": {path.name if path.parent == output else path.relative_to(output).as_posix(): _sha(path)
                                  for path in [canonical_path, audit_path, *(Path(p) for p in native_paths.values())]}}
    _write_json(output / "live_result.json", document)
    return document
