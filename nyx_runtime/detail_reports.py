"""Keep the original four country reports and interval CSV/Parquet outputs.

The readable display modules come from the verified 15 October report installer.
They preserve the distinction between current final forecasts and historical
CPU baseline statistics. They never fit estimators or collect new observations.
"""
from __future__ import annotations

from datetime import date, timedelta
from html import escape
import importlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

from .engine import ZONES, read_json, sha256_file, verify_artifacts


REPORTING_ROOT = Path(__file__).resolve().parents[1] / "production" / "reporting"


def _renderer():
    manifest = read_json(REPORTING_ROOT.parent / "REPORTING_MANIFEST.json")
    verify_artifacts(REPORTING_ROOT, manifest["files"], "Sources des rapports détaillés")
    for name in ("report_detail_adapter", "nyx_prod_templates"):
        loaded = sys.modules.get(name)
        if loaded is not None and not Path(loaded.__file__).resolve().is_relative_to(REPORTING_ROOT):
            raise RuntimeError("Un autre moteur de rapports est importé : lancer un nouveau processus.")
    if str(REPORTING_ROOT) not in sys.path:
        sys.path.insert(0, str(REPORTING_ROOT))
    return importlib.import_module("report_detail_adapter").render_details


def interval_forecasts(forecasts: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict:
    """Use exactly the original display envelope; leave native forecasts intact."""
    import numpy as np
    result = {}
    for zone in ZONES:
        final = forecasts[zone].copy(deep=True)
        source = baseline[zone].reindex(final.index)
        point = final.price_eur_mwh.to_numpy(float)
        low = source.nyx__q10.to_numpy(float)
        high = source.nyx__q90.to_numpy(float)
        if not np.isfinite(np.column_stack([low, point, high])).all():
            raise ValueError(f"{zone} : intervalle source incomplet.")
        for column in ("nyx__q10", "nyx__q50", "nyx__q90"):
            if not np.array_equal(final[column].to_numpy(float), source[column].to_numpy(float)):
                raise ValueError(f"{zone} : quantiles natifs et socle de reporting différents.")
        final["interval_lower_eur_mwh"] = np.minimum(low, point)
        final["interval_upper_eur_mwh"] = np.maximum(high, point)
        final["interval_nominal_level"] = .8
        final["interval_method"] = "baseline_envelope_uncalibrated"
        final["interval_final_expert_calibrated"] = False
        result[zone] = final
    return result


def render_country_reports(result: Mapping[str, Any], first_day: str, cutoff: Any,
                           output_dir: Path, comparisons: Mapping[str, Any] | None,
                           reporting_audit: Mapping[str, Any]) -> dict:
    """Render all historical reports before the caller can publish success.

Absent report-only observations use only the original frozen historical EPEX
values in the country renderer. Future observations and missing Storm values
remain absent. Model predictions and training inputs are never modified.
"""
    import numpy as np
    import pandas as pd
    render_details = _renderer()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    final = interval_forecasts(result["forecasts"], result["baseline"])
    countries, csv_files, parquet_files, hashes = {}, {}, {}, {}
    names = [f"forecast_{zone.lower()}_{first_day}_nyx_annual_cpu.html" for zone in ZONES]
    names += [f"forecast_{zone.lower()}_{first_day}_nyx_annual_interval.{suffix}"
              for zone in ZONES for suffix in ("csv", "parquet")]
    names += ["index.html", "report_audit.json"]
    if any((output / name).exists() for name in names):
        raise FileExistsError("Un rapport détaillé existe déjà dans cette destination.")
    model_audit = result["model_audit"]
    model_audit_path = output.parent / "forecast_extension/daily_refit/model_audit.json"
    model_reference = {"path": str(model_audit_path)}
    bound_inputs = {}
    receipt_reference = model_audit.get("daily_refit_receipt")
    if receipt_reference is not None:
        receipt_path = Path(receipt_reference["path"]).resolve()
        if sha256_file(receipt_path) != receipt_reference["sha256"]:
            raise ValueError("Le reçu des modèles a changé avant le rendu des rapports.")
        receipt = read_json(receipt_path)
        expected = receipt.get("artifact_sha256", {}).get("model_audit.json")
        if (receipt.get("protocol") != "nyx_daily_model_refit_v1" or receipt.get("status") != "MODELS_COMPLETE"
                or receipt_path.parent != model_audit_path.parent or not isinstance(expected, str)
                or sha256_file(model_audit_path) != expected):
            raise ValueError("L'audit des modèles diffère de son reçu scellé.")
        model_reference["sha256"] = expected
        bound_inputs = {str(receipt_path): receipt_reference["sha256"],
                        str(model_audit_path): expected}
    common = {
        "protocol": "nyx_annual_report_only_standard_templates_v1",
        "delivery_day": first_day, "horizon_days": 2,
        "forecast_cutoff_utc": pd.Timestamp(cutoff).isoformat(),
        "qualified_for_production": False,
        "historical_scores_model": "NYX CPU baseline only",
        "final_expert_oos_history_available": False,
        "final_expert_price_quantiles_available": False,
        "prediction_interval_method": "baseline_envelope_uncalibrated",
        "prediction_interval_nominal_level": .8,
        "final_expert_interval_coverage_calibrated": False,
        "final_expert_recipes_refit_attempted": model_audit.get("daily_refit_completed") is True,
        "final_models_fitted": model_audit.get("final_models_fitted", 0),
        "reference_models_refitted": model_audit.get("reference", {}).get("current_fit_counts", {}),
        "daily_refit_receipt": model_audit.get("daily_refit_receipt"),
        "model_audit": model_reference,
        "price_reference": "EPEX", "reporting_sources": dict(reporting_audit),
    }
    for zone in ZONES:
        baseline = result["baseline"][zone]
        observed = (comparisons or {}).get("observed", {}).get(zone)
        storm = (comparisons or {}).get("storm", {}).get(zone)
        if observed is None:
            observed = pd.Series(np.nan, index=baseline.index, name="actual")
        if storm is None:
            storm = pd.Series(np.nan, index=baseline.index, name="storm")
        sources = (comparisons or {}).get("audits", {}).get(zone, {
            "observed": {"status": "unavailable", "used_for_prediction": False},
            "storm": {"status": "unavailable", "used_for_prediction": False},
        })
        path = output / f"forecast_{zone.lower()}_{first_day}_nyx_annual_cpu.html"
        render_details(zone, first_day, final[zone], baseline, result["covariates"],
                       observed, storm, path, {**common, "zone_sources": sources})
        countries[zone] = str(path)
        csv_path = output / f"forecast_{zone.lower()}_{first_day}_nyx_annual_interval.csv"
        parquet_path = csv_path.with_suffix(".parquet")
        final[zone].to_csv(csv_path, index_label="timestamp_utc", float_format="%.17g")
        final[zone].to_parquet(parquet_path)
        csv_files[zone], parquet_files[zone] = str(csv_path), str(parquet_path)
        for artifact in (path, csv_path, parquet_path):
            hashes[artifact.name] = sha256_file(artifact)
    next_day = (date.fromisoformat(first_day) + timedelta(days=1)).isoformat()
    overviews = [f"CWE_Model_Storm_J1_J2_{first_day}.html",
                 f"CWE_Model_Storm_{first_day}.html", f"CWE_Model_Storm_{next_day}.html"]
    links = [(name, name.removesuffix(".html")) for name in overviews]
    links += [(Path(path).name, zone + " — prévision et historique de référence")
              for zone, path in countries.items()]
    index = output / "index.html"
    index.write_text('<!doctype html><html lang="fr"><meta charset="utf-8">'
        '<title>NYX — ' + escape(first_day) + '</title>'
        '<style>body{font:17px Arial;max-width:950px;margin:40px;line-height:1.7}</style>'
        '<h1>NYX — J+1 / J+2</h1><p>Les prévisions finales sont distinctes des '
        'statistiques historiques du socle CPU. L’enveloppe indicative n’est pas calibrée '
        'pour les experts finaux.</p><ul>' + ''.join(
            '<li><a href="' + escape(name) + '">' + escape(label) + '</a></li>'
            for name, label in links) + '</ul></html>', encoding="utf-8")
    hashes[index.name] = sha256_file(index)
    for path, expected in bound_inputs.items():
        if sha256_file(Path(path)) != expected:
            raise ValueError("Un artefact modèle a changé pendant le rendu des rapports.")
    audit_path = output / "report_audit.json"
    audit_path.write_text(json.dumps({**common, "status": "COUNTRY_REPORTS_COMPLETE",
        "original_artifacts_sha256": bound_inputs, "reports_sha256": hashes},
        ensure_ascii=False, indent=2, allow_nan=False, default=str)
        + "\n", encoding="utf-8")
    return {"html_files": list(countries.values()), "countries": countries,
            "csv_files": csv_files, "parquet_files": parquet_files,
            "index": str(index), "audit_path": str(audit_path), "output_sha256": hashes}


def finalize_report_set(country_result: Mapping[str, Any], cwe_result: Mapping[str, Any]) -> None:
    """Mark the original full report set complete only after all seven HTMLs exist."""
    from .runtime import write_json
    audit_path = Path(country_result["audit_path"]).resolve()
    output = audit_path.parent
    audit = read_json(audit_path)
    if audit.get("status") != "COUNTRY_REPORTS_COMPLETE":
        raise ValueError("Les rapports détaillés ne sont pas prêts à être finalisés.")
    verify_artifacts(output, audit["reports_sha256"], "Rapports détaillés produits")
    day = audit["delivery_day"]
    next_day = (date.fromisoformat(day) + timedelta(days=1)).isoformat()
    expected = {f"CWE_Model_Storm_{day}.html", f"CWE_Model_Storm_{next_day}.html",
                f"CWE_Model_Storm_J1_J2_{day}.html"}
    files = [Path(path).resolve() for path in cwe_result["html_files"]]
    if len(files) != 3 or {path.name for path in files} != expected:
        raise ValueError("Les trois rapports CWE J+1/J+2 sont requis.")
    for path in files:
        if path.parent != output or not path.is_file() or path.is_symlink():
            raise ValueError("Un rapport CWE est absent ou hors du dossier de publication.")
        audit["reports_sha256"][path.name] = sha256_file(path)
    audit["status"] = "REPORTS_COMPLETE"
    audit["combined_cwe_report"] = f"CWE_Model_Storm_J1_J2_{day}.html"
    write_json(audit_path, audit)
