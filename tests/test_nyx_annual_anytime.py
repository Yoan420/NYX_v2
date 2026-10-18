"""Any-hour orchestration uses the real recipes and isolated source snapshots."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess

import pandas as pd
import pytest

import run_nyx_annual_anytime as runner
from chronos2_hourly import nyx_annual_operational_context as context
from chronos2_hourly import nyx_annual_live_preflight as gate
from chronos2_hourly import nyx_annual_pipeline as pipeline
from test_nyx_annual_cpu_reports import wired


@pytest.fixture(autouse=True)
def clean_context(monkeypatch):
    monkeypatch.delenv(context.ENV, raising=False)


def _mock_consumers(calls):
    def prepare(day, **kwargs):
        calls.append(("prepare", day, kwargs, context.operational_context()))
        assert kwargs["action"] == "prepare"
        assert kwargs["cache_root"].parent == kwargs["bundle"].parent
        assert kwargs["output"] != kwargs["bundle"]
        return {"state": "PREPARED", "forecast_published": False}
    def reports(**kwargs):
        calls.append(("reports", kwargs, context.operational_context()))
        output = kwargs["output"]
        assert not output.exists()
        output.mkdir(parents=True)
        index = output / "index.html"
        index.write_text("annual country reports", encoding="utf-8")
        return {"status": "REPORTS_COMPLETE", "index": str(index)}
    return prepare, reports


@pytest.mark.parametrize("local,day,origin,hours", [
    ("2026-09-28 00:05+02:00", "2026-09-29", "2026-09-27T22:05:00+00:00", 24),
    ("2026-09-28 07:59+02:00", "2026-09-29", "2026-09-28T05:59:00+00:00", 24),
    ("2026-09-28 08:01+02:00", "2026-09-29", "2026-09-28T06:00:00+00:00", 24),
    ("2026-09-28 14:00+02:00", "2026-09-29", "2026-09-28T06:00:00+00:00", 24),
    ("2026-09-28 23:59+02:00", "2026-09-29", "2026-09-28T06:00:00+00:00", 24),
    ("2026-03-28 23:59+01:00", "2026-03-29", "2026-03-28T07:00:00+00:00", 23),
    ("2026-10-24 23:59+02:00", "2026-10-25", "2026-10-24T06:00:00+00:00", 25),
])
def test_any_hour_reaches_the_report_consumer_with_one_effective_origin(monkeypatch, tmp_path, local, day, origin, hours):
    now = pd.Timestamp(local).tz_convert("UTC")
    monkeypatch.setattr(context, "_now", lambda: now + pd.Timedelta(minutes=2))
    calls = []
    prepare, reports = _mock_consumers(calls)
    def checked_prepare(delivery_day, **kwargs):
        assert delivery_day == day
        _, current, cutoff = gate.delivery_grid(delivery_day)
        assert len(current) == hours and cutoff == pd.Timestamp(origin)
        return prepare(delivery_day, **kwargs)
    result = runner.run(now_utc=now, run_root=tmp_path / "runs", report_root=tmp_path / "reports",
                        prepare_runner=checked_prepare, report_runner=reports)
    assert result["state"] == "REPORTS_COMPLETE"
    assert result["qualified_for_production"] is False
    assert result["operational_snapshot"]["delivery_day"] == day
    assert result["operational_snapshot"]["requested_at_utc"] == now.isoformat()
    assert calls[0][-1] == calls[1][-1] == result["operational_snapshot"]
    assert context.operational_context() is None
    assert json.loads((Path(result["run_directory"]) / "run.json").read_text(encoding="utf-8")) == result


def test_repeated_launches_create_new_outputs_without_overwriting_previous_reports(tmp_path):
    calls = []
    prepare, reports = _mock_consumers(calls)
    kwargs = dict(now_utc="2026-09-28T12:00:00+00:00", run_root=tmp_path / "runs",
                  report_root=tmp_path / "reports", prepare_runner=prepare, report_runner=reports)
    first = runner.run(**kwargs)
    old_index = Path(first["index"])
    old_record = (Path(first["run_directory"]) / "run.json").read_bytes()
    second = runner.run(**kwargs)
    assert second["run_directory"] != first["run_directory"]
    assert second["output"] != first["output"]
    assert old_index.read_text(encoding="utf-8") == "annual country reports"
    assert (Path(first["run_directory"]) / "run.json").read_bytes() == old_record


def test_completed_resume_is_refused_before_mutating_success(tmp_path):
    calls = []
    prepare, reports = _mock_consumers(calls)
    result = runner.run(now_utc="2026-09-28T12:00:00+00:00", run_root=tmp_path / "runs",
                        report_root=tmp_path / "reports", prepare_runner=prepare, report_runner=reports)
    directory = Path(result["run_directory"])
    original = {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()}
    with pytest.raises(ValueError, match="Completed run is immutable"):
        runner.run(resume_run=directory, run_root=tmp_path / "runs", report_root=tmp_path / "reports",
                   prepare_runner=lambda *a, **k: pytest.fail("Completed resume reran models"))
    assert {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()} == original


def test_interrupted_resume_keeps_context_and_restores_surrounding_environment(tmp_path):
    launch = "2026-09-28T01:00:00+00:00"
    calls = []
    prepare, reports = _mock_consumers(calls)
    def failed_prepare(day, **kwargs):
        prepare(day, **kwargs)
        return {"state": "BLOCKED", "stages": [{"state": "ERROR", "error": "JAO response empty"}]}
    first = runner.run(now_utc=launch, run_root=tmp_path / "runs", report_root=tmp_path / "reports",
                       prepare_runner=failed_prepare, report_runner=reports)
    assert first["state"] == "BLOCKED" and "JAO response empty" in first["error"]
    assert len(calls) == 1
    outer = context.make_context("2026-09-30", "2026-09-29T12:00:00+00:00")
    with context.context_scope(outer):
        resumed = runner.run(resume_run=first["run_directory"], run_root=tmp_path / "runs",
                             report_root=tmp_path / "reports", prepare_runner=prepare, report_runner=reports)
        assert context.operational_context() == outer
    assert resumed["state"] == "REPORTS_COMPLETE"
    assert "error" not in resumed and "pipeline" not in resumed
    assert resumed["operational_snapshot"] == first["operational_snapshot"]
    assert calls[1][-1] == calls[2][-1] == first["operational_snapshot"]
    assert len(list(Path(resumed["run_directory"]).glob("launch_*.log"))) == 2
    assert context.operational_context() is None


@pytest.mark.parametrize("change,reason", [
    ({"asof_utc": "2026-09-28T06:00:00+00:00"}, "as-of time"),
    ({"delivery_day": "2026-09-30"}, "tomorrow"),
    ({"requested_at_utc": "2026-09-28T01:00:00"}, "timezone-aware"),
])
def test_resume_context_tampering_stops_before_logs_or_source_calls(tmp_path, change, reason):
    first = runner.run(now_utc="2026-09-28T01:00:00+00:00", run_root=tmp_path / "runs",
                       report_root=tmp_path / "reports", prepare_runner=lambda *a, **k: {"state": "BLOCKED"})
    directory = Path(first["run_directory"])
    record_path = directory / "run.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["operational_snapshot"].update(change)
    record_path.write_text(json.dumps(record), encoding="utf-8")
    before = record_path.read_bytes()
    with pytest.raises(ValueError, match=reason):
        runner.run(resume_run=directory, run_root=tmp_path / "runs", report_root=tmp_path / "reports",
                   prepare_runner=lambda *a, **k: pytest.fail("Changed context contacted a source"))
    assert record_path.read_bytes() == before
    assert len(list(directory.glob("launch_*.log"))) == 1
    assert context.operational_context() is None


def test_resume_path_and_delivery_are_bound_to_the_record(tmp_path):
    result = runner.run(now_utc="2026-09-28T01:00:00+00:00", run_root=tmp_path / "runs",
                        report_root=tmp_path / "reports", prepare_runner=lambda *a, **k: {"state": "BLOCKED"})
    with pytest.raises(ValueError, match="inside"):
        runner.run(resume_run=result["run_directory"], run_root=tmp_path / "different")
    with pytest.raises(ValueError, match="delivery day differs"):
        runner.run(resume_run=result["run_directory"], delivery_day="2026-09-30",
                   run_root=tmp_path / "runs", report_root=tmp_path / "reports")
    with pytest.raises(ValueError, match="output path differs"):
        runner.run(resume_run=result["run_directory"], run_root=tmp_path / "runs",
                   report_root=tmp_path / "different_reports")


def test_report_failure_retains_resume_directory_and_restores_context(tmp_path):
    def failure(**kwargs):
        raise ValueError("forecast output incomplete")
    result = runner.run(now_utc="2026-09-28T01:00:00+00:00", run_root=tmp_path / "runs",
                        report_root=tmp_path / "reports", prepare_runner=lambda *a, **k: {"state": "PREPARED"},
                        report_runner=failure)
    assert result["state"] == "BLOCKED" and "forecast output incomplete" in result["error"]
    assert Path(result["run_directory"]).is_dir()
    assert context.operational_context() is None


def test_invalid_naive_launch_and_past_delivery_leave_no_run(tmp_path):
    for kwargs, reason in (({"now_utc": "2026-09-28T12:00:00"}, "Timezone-aware"),
                           ({"now_utc": "2026-09-28T12:00:00+00:00", "delivery_day": "2026-09-28"}, "tomorrow")):
        with pytest.raises(ValueError, match=reason):
            runner.run(run_root=tmp_path / "runs", report_root=tmp_path / "reports", **kwargs)
    assert not (tmp_path / "runs").exists()


def test_pipeline_uses_snapshot_capture_archives_and_shared_fingerprinted_baseline_only(monkeypatch, tmp_path):
    from chronos2_hourly import nyx_annual_cpu_baseline as baseline
    from chronos2_hourly import nyx_annual_cpu_bundle_builder as builder
    from chronos2_hourly import nyx_annual_cpu_reference_builder as reference
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    builds = []
    monkeypatch.setattr(baseline, "build_from_bundle", lambda *a, **k: builds.append(("baseline", k)))
    monkeypatch.setattr(builder, "materialize_features", lambda *a, **k: builds.append(("features", k)))
    monkeypatch.setattr(reference, "build_from_bundle", lambda *a, **k: builds.append(("reference", k)))
    monkeypatch.setattr(builder, "seal_bundle", lambda *a, **k: builds.append(("seal", k)))
    day, cache, calls = "2026-09-29", tmp_path / "run/cache", []
    snapshot = context.make_context(day, "2026-09-28T01:00:00+00:00")
    with context.context_scope(snapshot):
        state = pipeline.run(day, action="prepare", bundle=tmp_path / "run/bundle", output=tmp_path / "out",
            cache_root=cache, runner=lambda script, args: calls.append((script, args)) or 0)
    assert state["state"] == "PREPARED" and state["forecast_published"] is False
    assert state["operational_snapshot"] == snapshot
    assert calls[0][0] == calls[1][0] == "run_nyx_annual_daily_capture.py"
    assert "--verify-only" not in calls[0][1] and "--verify-only" in calls[1][1]
    for flag, name in (("--jao-cache", "jao_capture"), ("--exchange-archive", "exchange_capture"),
                       ("--hydro-archive", "hydro_capture")):
        for _, args in calls[:2]:
            assert args[args.index(flag) + 1] == str(cache / name)
    commands = {script: args for script, args in calls[2:]}
    assert commands["run_nyx_annual_jao_source.py"][-2:] == ["--cache-root", str(cache / "jao_capture")]
    assert commands["run_nyx_annual_hydro_source.py"][-2:] == ["--archive", str(cache / "hydro_capture")]
    assert commands["run_nyx_annual_exchange_source.py"][-2:] == ["--archive", str(cache / "exchange_capture")]
    assert commands["run_nyx_annual_saturn_source.py"][-2:] == ["--cache", str(cache / "saturn")]
    assert commands["run_nyx_annual_thermal_source.py"][-2:] == ["--cache", str(cache / "thermal")]
    assert commands["run_nyx_annual_fuel_source.py"][-2:] == ["--cache-dir", str(cache / "fuel")]
    assert [name for name, _ in builds] == ["baseline", "features", "reference", "seal"]
    assert builds[0][1]["checkpoint_cache"] == tmp_path / "runs/live/nyx_annual_cpu/_cpu_baseline_cache"


def test_operational_prepare_cannot_use_default_shared_sources_or_production(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    snapshot = context.make_context("2026-09-29", "2026-09-28T01:00:00+00:00")
    with context.context_scope(snapshot):
        for kwargs in ({"action": "forecast"}, {"action": "prepare"},
                       {"action": "prepare", "bundle": tmp_path / "bundle", "output": tmp_path / "out"}):
            with pytest.raises(ValueError, match="isolated cache"):
                pipeline.run("2026-09-29", runner=lambda *a: pytest.fail("Unsafe prepare contacted a source"), **kwargs)


def test_sealed_operational_bundle_is_revalidated_without_new_capture(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    bundle = tmp_path / "run/bundle"
    manifest = bundle / gate.MATERIALIZATION_PATH
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}", encoding="utf-8")
    checked = []
    monkeypatch.setattr(pipeline, "validate_prepared_bundle", lambda *a, **k: checked.append((a, k)))
    snapshot = context.make_context("2026-09-29", "2026-09-28T01:00:00+00:00")
    with context.context_scope(snapshot):
        result = pipeline.run("2026-09-29", action="prepare", bundle=bundle, output=tmp_path / "out",
                              cache_root=tmp_path / "run/cache",
                              runner=lambda *a: pytest.fail("Sealed snapshot downloaded newer data"))
    assert result["state"] == "PREPARED"
    assert checked == [((bundle, "2026-09-29"), {"require_asof": False})]


def test_baseline_plan_and_quantile_origins_agree_for_current_and_historical_days():
    from chronos2_hourly import nyx_annual_cpu_baseline as baseline
    from chronos2_hourly import nyx_annual_nyx_quantiles_gate as quantiles
    snapshot = context.make_context("2026-09-29", "2026-09-28T01:00:00+00:00")
    with context.context_scope(snapshot):
        plan = baseline.build_delivery_plan(pd.Timestamp("2026-09-29").date(), timezone="Europe/Paris")
        past_plan = baseline.build_delivery_plan(pd.Timestamp("2026-09-28").date(), timezone="Europe/Paris")
        assert plan.forecast_origin_utc == pd.Timestamp(snapshot["asof_utc"])
        assert past_plan.forecast_origin_utc == pd.Timestamp("2026-09-27T06:00:00+00:00")
        index = pd.DatetimeIndex(["2026-09-28T00:00:00+00:00", "2026-09-29T00:00:00+00:00"])
        expected = quantiles._expected_origins(index)
        assert list(expected) == [past_plan.forecast_origin_utc, plan.forecast_origin_utc]


def test_strict_consumer_refuses_operational_context_before_manifest_or_sources():
    from chronos2_hourly import nyx_annual_cpu_live as live
    snapshot = context.make_context("2026-09-29", "2026-09-28T01:00:00+00:00")
    with context.context_scope(snapshot):
        with pytest.raises(ValueError, match="anytime report consumer"):
            live.verify_activation()


def test_early_snapshot_reports_record_actual_origin_and_keep_all_country_recipes(wired):
    import run_nyx_annual_cpu_reports as reports
    from chronos2_hourly.nyx_annual_cpu_live import COMPOSITIONS
    day, bundle, output, _, manifest, calls = wired
    original_manifest = manifest.read_bytes()
    snapshot = context.make_context(day, "2026-10-02T01:00:00+00:00")
    with context.context_scope(snapshot):
        receipt = reports.run(bundle=bundle, delivery_day=day, output=output)
    assert receipt["operational_snapshot"] == snapshot
    assert receipt["forecast_cutoff_utc"] == snapshot["asof_utc"]
    assert receipt["source_snapshot_asof_verified"] is False
    assert receipt["qualified_for_production"] is False
    assert receipt["future_labels_used"] is False
    assert receipt["model_audit"]["compositions"] == COMPOSITIONS
    assert manifest.read_bytes() == original_manifest
    assert calls == ["validate", "fit", "validate"]
    for details in receipt["countries"].values():
        html = Path(details["html"]).read_text(encoding="utf-8")
        assert "État Saturn interrogé : 02/10/2026 à 03:00" in html
        assert "Coupure de prévision (J−1 à 08:00)" not in html
        assert "RMSE du protocole original à 08 h" in html
        assert "Modèle non encore qualifié" in html
        assert details["hours"] == 24


@pytest.mark.parametrize("exit_code", [0, 2])
def test_powershell_launcher_runs_and_reports_the_correct_resume_directory(tmp_path, exit_code):
    powershell = shutil.which("powershell.exe")
    if powershell is None:
        pytest.skip("Windows PowerShell is unavailable")
    launcher = tmp_path / "NYXAnnualAnytime.ps1"
    launcher.write_bytes((runner.ROOT / "NYXAnnualAnytime.ps1").read_bytes())
    fake = tmp_path / "fake_python.ps1"
    fake.write_text("""$NyxResultPath = $args[3]
$NyxRunPath = Join-Path $PSScriptRoot 'isolated_run'
$NyxFakeResult = @{ state='BLOCKED'; run_directory=$NyxRunPath; bundle=(Join-Path $NyxRunPath 'bundle'); error='JAO response empty'; index=(Join-Path $PSScriptRoot 'index.html') }
New-Item -ItemType Directory -Path (Split-Path -Parent $NyxResultPath) -Force | Out-Null
$NyxFakeResult | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $NyxResultPath -Encoding UTF8
$global:LASTEXITCODE = [int]$env:NYX_FAKE_EXIT_CODE
""", encoding="utf-8")
    result = subprocess.run([powershell, "-NoProfile", "-NonInteractive", "-File", str(launcher),
                             "-PythonExecutable", str(fake), "-NoOpen"],
                            capture_output=True, text=True, errors="replace", timeout=30,
                            env={**os.environ, "NYX_FAKE_EXIT_CODE": str(exit_code)})
    console = result.stdout + result.stderr
    if exit_code == 0:
        assert result.returncode == 0 and "Rapports termines" in console
    else:
        assert result.returncode != 0
        assert "Dossier de reprise" in console and "JAO response empty" in console
        # PowerShell wraps its native error display at the console width.
        compact_console = "".join(console.split())
        assert "".join(str(tmp_path / "isolated_run").split()) in compact_console
        assert "".join(str(tmp_path / "isolated_run/bundle").split()) not in compact_console
