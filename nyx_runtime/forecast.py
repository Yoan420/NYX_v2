"""Calcul quotidien J+1/J+2 et publication des prévisions natives en CSV.

Le producteur scientifique est chargé après le précontrôle. Les comparaisons
récupérées pour les rapports restent distinctes des entrées des modèles.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import importlib
import json
from pathlib import Path
from typing import Any, Mapping

from .engine import (
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
from .preflight import LivePreflightError, inspect_live
from .reporting import CSV_COLUMNS
from .detail_reports import render_country_reports


def write_json(path: Path, value: dict) -> None:
    """Crée un reçu de sortie une seule fois, sans remplacer un reçu existant."""
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False, default=str)
        handle.write("\n")


def canonical_forecasts(forecasts: Mapping[str, Any], first_day: str, cutoff: Any,
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
        frames.append(pd.DataFrame(values, columns=CSV_COLUMNS))
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
    first_day, stamp = validate_request(first_day, cutoff)
    repo, bundle, seed_output = (resolve_directory(config, key) for key in ("repo_root", "seed_bundle", "seed_output"))
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
    activate_runtime(repo)
    producer = importlib.import_module("chronos2_hourly.nyx_daily_refit")
    try:
        result = producer.produce_refit_horizon(repo, bundle, seed_output, first_day, stamp,
                                                output / "forecast_extension", context=context)
    except Exception as error:
        raise RuntimeError("Refit NYX interrompu : " + redact_error(error)) from error
    # First validation happens before the optional, strictly report-only queries.
    canonical_forecasts(result["forecasts"], first_day, stamp)
    audit = result.get("model_audit", {})
    if (audit.get("daily_refit_completed") is not True or audit.get("future_labels_used") is not False
            or audit.get("refit_all_trainable_nyx_models") is not True
            or audit.get("seed_inference_used_for_final_forecast") is not False
            or audit.get("final_model_refit_attempts") != 7
            or audit.get("reference", {}).get("current_fit_counts") != {"hgb": 12, "test2_pairs": 2}):
        raise ValueError("Le producteur n'atteste pas le réentraînement quotidien complet attendu.")
    refit_receipt = output / "forecast_extension/daily_refit/receipt.json"
    sealed = read_json(refit_receipt)
    if (sealed.get("protocol") != "nyx_daily_model_refit_v1"
            or sealed.get("status") != "MODELS_COMPLETE" or sealed.get("future_labels_used") is not False
            or sealed.get("first_delivery_day") != first_day or sealed.get("forecast_days") != 2
            or sealed.get("shared_cutoff_utc") != stamp.isoformat()
            or sealed.get("new_forecast_refit_required") is not True):
        raise ValueError("Reçu du refit complet absent ou invalide.")
    verify_artifacts(refit_receipt.parent, sealed.get("artifact_sha256"), "Prévisions et modèles fraîchement scellés")
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
            reporting_audit = {"status": "unavailable", "reason": redact_error(error),
                               "used_for_prediction": False, "used_for_training": False,
                               "fallback_to_model_inputs": False}
    canonical = canonical_forecasts(result["forecasts"], first_day, stamp, comparisons)
    native = output / "native_forecasts"
    native.mkdir(exist_ok=False)
    native_paths = {}
    for zone in ZONES:
        path = native / f"{zone}.csv"
        result["forecasts"][zone].to_csv(path, index=True, index_label="timestamp_utc", encoding="utf-8")
        native_paths[zone] = str(path)
    canonical_path = output / "predictions.csv"
    canonical.to_csv(canonical_path, index=False, encoding="utf-8")
    # The original country reports contain negative-price probabilities and
    # historical baseline statistics that the compact CWE overview cannot replace.
    # Any rendering failure must propagate before this run can be published.
    country_reports = render_country_reports(result, first_day, stamp,
                                              output / "reports", comparisons, reporting_audit)
    audit_path = output / "live_audit.json"
    write_json(audit_path, {"protocol": PROTOCOL, "model_audit": audit,
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
                "country_reports": country_reports, "index": country_reports["index"],
                "csv_files": [str(canonical_path), *native_paths.values()],
                "forecast_extension_dir": str(output / "forecast_extension"),
                "model_audit_path": str(refit_receipt.parent / "model_audit.json"),
                "source_audit_path": str(output / "forecast_extension/horizon_sources/receipt.json"),
                "reporting_audit_path": reporting_receipt, "audit_path": str(audit_path),
                "refit_receipt_path": str(refit_receipt), "receipt_path": str(refit_receipt),
                "baseline_receipt_path": str(output / "forecast_extension/baseline_extension/receipt.json"),
                "native_columns": {zone: list(result["forecasts"][zone].columns) for zone in ZONES},
                "output_sha256": {path.name if path.parent == output else path.relative_to(output).as_posix(): sha256_file(path)
                                  for path in [canonical_path, audit_path, *(Path(p) for p in native_paths.values())]}}
    write_json(output / "live_result.json", document)
    return document
