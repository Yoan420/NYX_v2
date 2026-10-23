"""Original report values, DST hours and truthful historical model identity."""
from datetime import date, timedelta
import base64
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import pytest

from nyx_runtime import detail_reports, reporting
from nyx_runtime.engine import ZONES
from nyx_runtime.forecast import canonical_forecasts


def assert_final_chart_values(text, expected):
    """Read the original renderer's real Plotly payload, not rounded HTML text."""
    def values(trace):
        payload = trace["y"]
        if isinstance(payload, dict):
            return np.frombuffer(base64.b64decode(payload["bdata"]), dtype=payload["dtype"])
        return np.asarray(payload, dtype=float)
    chart = json.loads(re.search(r'<script id="annual-final-price-[^"]+-data" type="application/json">(.*?)</script>',
                                 text, re.S).group(1))
    traces = {item["name"]: item for item in chart["data"]}
    np.testing.assert_array_equal(values(traces["NYX"]), expected.price_eur_mwh)
    np.testing.assert_array_equal(values(traces["Borne supérieure indicative"]), expected.interval_upper_eur_mwh)
    np.testing.assert_array_equal(values(traces["Intervalle indicatif · 80 % nominal"]), expected.interval_lower_eur_mwh)
    chart = json.loads(re.search(r'<script id="annual-final-negative-[^"]+-data" type="application/json">(.*?)</script>',
                                 text, re.S).group(1))
    probability = next(item for item in chart["data"] if item["name"] == "P(prix < 0)")
    np.testing.assert_array_equal(values(probability), expected.p_negative.to_numpy() * 100)


def report_fixture(first_day="2026-10-24"):
    first = date.fromisoformat(first_day)
    cutoff = pd.Timestamp(f"{first - timedelta(days=1)} 08:00", tz="Europe/Paris").tz_convert("UTC")
    full = pd.date_range(str(first - timedelta(days=365)), str(first + timedelta(days=2)),
                         tz="Europe/Paris", freq="h", inclusive="left").tz_convert("UTC")
    future = full[full >= pd.Timestamp(first_day, tz="Europe/Paris").tz_convert("UTC")]
    baseline, forecasts = {}, {}
    for number, zone in enumerate(ZONES):
        middle = 30 + number + np.sin(np.arange(len(full)) / 24)
        base = pd.DataFrame({"nyx__q10": middle - 15, "nyx__q50": middle,
                             "nyx__q90": middle + 12, "actual": middle + 3}, index=full)
        base.loc[future, "actual"] = np.nan
        point = base.loc[future, "nyx__q50"].to_numpy() + np.linspace(-20, 20, len(future))
        forecast = base.loc[future, ["nyx__q10", "nyx__q50", "nyx__q90"]].copy()
        forecast["price_eur_mwh"] = point
        forecast["p_negative"] = np.linspace(.01, .8, len(future))
        forecast["p_negative_raw"] = .2
        forecast["reference_eur_mwh"] = point - 1
        forecast["forecast_origin_utc"] = cutoff
        forecast["horizon_day"] = [(stamp.date() - first).days + 1
                                    for stamp in future.tz_convert("Europe/Paris")]
        baseline[zone], forecasts[zone] = base, forecast
    return first_day, cutoff, {
        "forecasts": forecasts, "baseline": baseline,
        "covariates": pd.DataFrame({"load_fr": np.full(len(full), 50.0)}, index=full),
        "model_audit": {"daily_refit_completed": True, "final_models_fitted": 7,
                        "reference": {"current_fit_counts": {"hgb": 12, "test2_pairs": 2}}},
    }


@pytest.mark.parametrize("first_day,hours", [("2026-03-28", 47), ("2026-10-24", 49)])
def test_country_exports_match_canonical_csv_and_original_envelope(first_day, hours):
    day, cutoff, result = report_fixture(first_day)
    before = {zone: frame.copy(deep=True) for zone, frame in result["forecasts"].items()}
    final = detail_reports.interval_forecasts(result["forecasts"], result["baseline"])
    canonical = canonical_forecasts(result["forecasts"], day, cutoff)
    for zone in ZONES:
        assert len(final[zone]) == hours
        rows = canonical.loc[canonical.zone.eq(zone)]
        np.testing.assert_array_equal(final[zone].price_eur_mwh, rows.prediction_eur_mwh)
        np.testing.assert_array_equal(final[zone].interval_lower_eur_mwh, rows.lower_bound_eur_mwh)
        np.testing.assert_array_equal(final[zone].interval_upper_eur_mwh, rows.upper_bound_eur_mwh)
        pd.testing.assert_frame_equal(result["forecasts"][zone], before[zone])


def test_mismatched_baseline_cannot_silently_change_report_bounds():
    _, _, result = report_fixture()
    result["baseline"]["FR"].loc[result["forecasts"]["FR"].index[0], "nyx__q10"] -= 1
    with pytest.raises(ValueError, match="quantiles natifs"):
        detail_reports.interval_forecasts(result["forecasts"], result["baseline"])


def test_original_country_report_retains_both_days_probabilities_and_history(tmp_path):
    day, cutoff, result = report_fixture()
    model_dir = tmp_path / "forecast_extension/daily_refit"
    model_dir.mkdir(parents=True)
    model_audit = model_dir / "model_audit.json"
    model_audit.write_text(json.dumps(result["model_audit"]), encoding="utf-8")
    receipt = model_dir / "receipt.json"
    receipt.write_text(json.dumps({"protocol": "nyx_daily_model_refit_v1", "status": "MODELS_COMPLETE",
        "artifact_sha256": {"model_audit.json": hashlib.sha256(model_audit.read_bytes()).hexdigest()}}),
        encoding="utf-8")
    result["model_audit"]["daily_refit_receipt"] = {
        "path": str(receipt), "sha256": hashlib.sha256(receipt.read_bytes()).hexdigest()}
    output = detail_reports.render_country_reports(result, day, cutoff, tmp_path / "reports", None,
                                                   {"status": "not_requested"})
    assert set(output["countries"]) == set(ZONES)
    assert Path(output["index"]).is_file()
    for zone, html_path in output["countries"].items():
        text = Path(html_path).read_text(encoding="utf-8")
        audit = json.loads(re.search(r'<script id="nyx-report-audit" type="application/json">(.*?)</script>',
                                    text, re.S).group(1))
        assert audit["forecast_physical_hours"] == 49
        assert audit["horizon_days"] == 2
        assert pd.Timestamp(audit["forecast_origin_utc"]) == cutoff
        assert audit["model_roles"] == {"current": "annual_final_point_forecast",
                                        "historical_statistics": "cpu_baseline_quantiles"}
        assert audit["final_annual_oof_history_available"] is False
        assert audit["final_interval_coverage_guaranteed"] is False
        assert 'data-final-oof-history="unavailable"' in text
        assert "P(prix &lt; 0)" in text
        assert "25/10/2026 02:00 CEST +0200" in text
        assert "25/10/2026 02:00 CET +0100" in text
        exported = pd.read_parquet(output["parquet_files"][zone])
        expected = detail_reports.interval_forecasts(result["forecasts"], result["baseline"])[zone]
        assert_final_chart_values(text, expected)
        # Parquet retains physical instants but normalizes datetime units and
        # does not retain pandas' inferred frequency metadata.
        exported["forecast_origin_utc"] = exported["forecast_origin_utc"].astype("datetime64[ns, UTC]")
        expected["forecast_origin_utc"] = expected["forecast_origin_utc"].astype("datetime64[ns, UTC]")
        pd.testing.assert_frame_equal(exported, expected, check_freq=False)
        raw = pd.read_csv(output["csv_files"][zone], float_precision="round_trip")
        np.testing.assert_array_equal(raw.price_eur_mwh, expected.price_eur_mwh)
        np.testing.assert_array_equal(raw.p_negative, expected.p_negative)
    assert all(frame.actual.loc[result["forecasts"][zone].index].isna().all()
               for zone, frame in result["baseline"].items())
    with pytest.raises(ValueError, match="trois rapports"):
        detail_reports.finalize_report_set(output, {"html_files": []})
    canonical_path = tmp_path / "predictions.csv"
    canonical_forecasts(result["forecasts"], day, cutoff).to_csv(canonical_path, index=False)
    cwe = reporting.render_reports(canonical_path, tmp_path / "reports")
    detail_reports.finalize_report_set(output, cwe)
    complete = json.loads(Path(output["audit_path"]).read_text(encoding="utf-8"))
    assert complete["status"] == "REPORTS_COMPLETE"
    assert len([name for name in complete["reports_sha256"] if name.endswith(".html")]) == 8
    with pytest.raises(FileExistsError):
        detail_reports.render_country_reports(result, day, cutoff, tmp_path / "reports", None, {})
