"""Publication checks for daily reports, independent of production activation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly.nyx_annual_cpu_live import COMPOSITIONS, COUNTRIES, PRICE_EXPERTS
from chronos2_hourly.nyx_annual_live_preflight import delivery_grid, sha256
import run_nyx_annual_cpu_reports as reports


@pytest.fixture
def wired(tmp_path, monkeypatch):
    day = "2026-10-03"
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    source = bundle / "bound_input.bin"
    source.write_bytes(b"sealed input")
    root = tmp_path / "repo"
    for name in (*reports.QUALIFICATION_CODE, "run_nyx_annual_cpu_reports.py",
                 "chronos2_hourly/nyx_annual_cpu_reporting.py"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"pinned code")
    manifest = root / "config/nyx_annual_cwe_historical.json"
    manifest.parent.mkdir()
    manifest.write_text('{"forecast_enabled":false}', encoding="utf-8")
    monkeypatch.setattr(reports, "ROOT", root)
    monkeypatch.setattr(reports, "bundle_hashes", lambda _: {"bound_input.bin": sha256(source)})
    monkeypatch.setattr(reports, "load_bundle", lambda *_: object())
    calls = []

    def validated(_bundle, _day, *, require_asof):
        assert not require_asof
        calls.append("validate")
        return {"passed": True, "source_snapshot_asof_verified": False,
                "checks": {"raw_sources": "verified"}}

    monkeypatch.setattr(reports, "validate_prepared_bundle", validated)

    def fitted(_data, delivery_day, model_dir, progress):
        calls.append("fit")
        _, current, _ = delivery_grid(delivery_day)
        frames = {zone: pd.DataFrame({"price_eur_mwh": np.arange(len(current))-6.,
                                      "p_negative": np.linspace(.1, .8, len(current))},
                                     index=current) for zone in COUNTRIES}
        model_dir.mkdir()
        price, negative = {}, {}
        for family in PRICE_EXPERTS:
            path = model_dir / f"{family}.cbm"
            path.write_bytes(family.encode())
            price[family] = {"model": {"path": str(path), "sha256": sha256(path)}}
        for zone in COUNTRIES:
            path = model_dir / f"negative_{zone}.cbm"
            path.write_bytes(zone.encode())
            negative[zone] = {"model": {"path": str(path), "sha256": sha256(path)}}
        return frames, {"compositions": COMPOSITIONS.copy(), "price_experts": price,
                        "negative_countries": negative}

    monkeypatch.setattr(reports, "execute_models", fitted)
    return day, bundle, tmp_path / "reports", source, manifest, calls


def test_unqualified_reports_publish_four_html_and_actual_provenance(wired):
    day, bundle, output, _, manifest, calls = wired
    original = manifest.read_bytes()
    receipt = reports.run(bundle=bundle, delivery_day=day, output=output)
    assert calls == ["validate", "fit", "validate"]
    assert manifest.read_bytes() == original
    assert receipt["status"] == "REPORTS_COMPLETE"
    assert receipt["qualified_for_production"] is False
    assert receipt["forecast_published"] is False
    assert receipt["reports_published"] is True
    assert receipt["source_snapshot_asof_verified"] is False
    assert receipt["future_labels_used"] is False
    assert set(receipt["countries"]) == set(COUNTRIES)
    assert (output / "index.html").is_file()
    assert "Modèle non encore qualifié" in (output / "index.html").read_text(encoding="utf-8")
    for zone, info in receipt["countries"].items():
        assert info["hours"] == 24
        html = Path(info["html"])
        document = html.read_text(encoding="utf-8")
        assert "Modèle non encore qualifié" in document
        assert "chaîne annuelle qualifiée" not in document
        assert sha256(html) == info["html_sha256"]
        assert Path(info["csv"]).is_file() and Path(info["parquet"]).is_file()
        assert all(Path(item["model"]["path"]).is_file() for item in
                   receipt["model_audit"]["price_experts"].values())
    assert json.loads((output / "receipt.json").read_text())["protocol"] == reports.PROTOCOL


@pytest.mark.parametrize("failure", ["Delivery capture missing", "Observed future labels", "Source checksum differs"])
def test_invalid_bundle_never_fits_or_publishes(wired, monkeypatch, failure):
    day, bundle, output, _, _, calls = wired

    def blocked(*args, **kwargs):
        raise ValueError(failure)

    monkeypatch.setattr(reports, "validate_prepared_bundle", blocked)
    with pytest.raises(ValueError, match=failure):
        reports.run(bundle=bundle, delivery_day=day, output=output)
    assert "fit" not in calls
    assert not output.exists()
    assert not (output.parent / f".{output.name}.attempts").exists()


def test_future_cutoff_never_fits(wired):
    _, bundle, output, _, _, calls = wired
    with pytest.raises(ValueError, match="Wait until D-1"):
        reports.run(bundle=bundle, delivery_day="2100-10-03", output=output)
    assert calls == []
    assert not output.exists()


def test_mutated_input_retains_failed_attempt_without_publication(wired, monkeypatch):
    day, bundle, output, source, _, _ = wired
    original_fit = reports.execute_models

    def changed(*args, **kwargs):
        result = original_fit(*args, **kwargs)
        source.write_bytes(b"changed during fit")
        return result

    monkeypatch.setattr(reports, "execute_models", changed)
    with pytest.raises(ValueError, match="changed during CPU retraining"):
        reports.run(bundle=bundle, delivery_day=day, output=output)
    assert not output.exists()
    statuses = list((output.parent / f".{output.name}.attempts").glob("*/status.json"))
    assert len(statuses) == 1
    assert json.loads(statuses[0].read_text())["status"] == "FAILED"


def test_existing_reports_never_overwritten(wired):
    day, bundle, output, _, _, calls = wired
    reports.run(bundle=bundle, delivery_day=day, output=output)
    first = (output / "receipt.json").read_bytes()
    calls.clear()
    with pytest.raises(ValueError, match="already exists"):
        reports.run(bundle=bundle, delivery_day=day, output=output)
    assert (output / "receipt.json").read_bytes() == first
    assert calls == []


@pytest.mark.parametrize("suffix", ["", "/2026-10-03", "/2026-10-03/zones/FR"])
def test_report_override_cannot_enter_qualified_outputs(wired, suffix):
    day, bundle, _, _, _, calls = wired
    output = Path(str(reports.ROOT / "runs/nyx_annual_cpu_live") + suffix)
    with pytest.raises(ValueError, match="qualified production outputs"):
        reports.run(bundle=bundle, delivery_day=day, output=output)
    assert calls == []
    assert not output.exists()


def test_reports_cannot_overlap_inputs(wired):
    day, bundle, _, _, _, calls = wired
    with pytest.raises(ValueError, match="immutable input bundle"):
        reports.run(bundle=bundle, delivery_day=day, output=bundle / "reports")
    assert calls == []


def test_preflight_failure_is_saved_in_cli_log_without_output(wired):
    _, bundle, output, _, _, calls = wired
    log = output.parent / "logs/reports.log"
    code = reports.main(["--bundle", str(bundle), "--delivery-day", "2100-10-03",
                         "--output", str(output), "--log-file", str(log)])
    assert code == 2
    entry = json.loads(log.read_text(encoding="utf-8"))
    assert entry["status"] == "BLOCKED"
    assert "Wait until D-1" in entry["reason"]
    assert calls == [] and not output.exists()
