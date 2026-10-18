"""Exercise the Windows entry points without fetching sources or training models."""
from __future__ import annotations

from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from zoneinfo import ZoneInfo

import pytest


ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(os.name != "nt", reason="Windows PowerShell entry points")


def quote(value):
    return "'" + str(value).replace("'", "''") + "'"


def powershell(command, *, cwd):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    env["PYTHONIOENCODING"] = "utf-8"
    script = ("[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding; "
              "$ErrorActionPreference = 'Stop'; " + command)
    result = subprocess.run(
        # This process only executes the local synthetic test project. Do not
        # change the machine/user policy or use the less restrictive Bypass.
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "RemoteSigned", "-Command", script],
        cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=45,
    )
    return result.returncode, result.stdout.decode("utf-8", errors="replace")


@pytest.fixture
def project(tmp_path):
    project = tmp_path / "NYX rapports avec espaces"
    project.mkdir()
    for filename in ("NYXAnnualReports.ps1", "NYXAnnualCPU.ps1", "run_nyx_annual_scheduled.py"):
        shutil.copy2(ROOT / filename, project / filename)
    return project


def synthetic_pipeline(project, *, prepare_code=0, report_code=0):
    state = "PREPARED" if prepare_code == 0 else "BLOCKED"
    (project / "run_nyx_annual_pipeline.py").write_text(
        "import argparse, json, sys\nfrom pathlib import Path\n"
        "p = argparse.ArgumentParser(); p.add_argument('--action'); p.add_argument('--delivery-day'); a = p.parse_args()\n"
        "with Path('calls.jsonl').open('a', encoding='utf-8') as f: f.write(json.dumps(vars(a)) + '\\n')\n"
        "print('Preparation et avertissement sur stderr', file=sys.stderr, flush=True)\n"
        "bundle = Path('runs/live/nyx_annual_cpu') / a.delivery_day\n"
        f"if {prepare_code} == 0: bundle.mkdir(parents=True, exist_ok=True)\n"
        f"print(json.dumps(dict(state={state!r}, stages=[])), flush=True)\n"
        f"raise SystemExit({prepare_code})\n", encoding="utf-8",
    )
    (project / "run_nyx_annual_cpu_reports.py").write_text(
        "import argparse, json, sys\nfrom pathlib import Path\n"
        "p = argparse.ArgumentParser(); p.add_argument('--bundle', type=Path); p.add_argument('--delivery-day'); p.add_argument('--output', type=Path); p.add_argument('--log-file', type=Path, required=True); a = p.parse_args()\n"
        "assert a.bundle == Path.cwd() / 'runs/live/nyx_annual_cpu' / a.delivery_day\n"
        "assert a.output == Path.cwd() / 'runs/nyx_annual_cpu_reports' / a.delivery_day\n"
        "assert a.bundle.is_dir()\n"
        "with Path('calls.jsonl').open('a', encoding='utf-8') as f: f.write(json.dumps({'action': 'reports', 'delivery_day': a.delivery_day}) + '\\n')\n"
        "print('Rapports et avertissement sur stderr', file=sys.stderr, flush=True)\n"
        "assert a.log_file.parent == Path.cwd() / 'runs/logs/nyx_annual_cpu_reports' / a.delivery_day\n"
        "with a.log_file.open('x', encoding='utf-8') as f: f.write('Rapports et avertissement sur stderr\\n')\n"
        f"if {report_code} == 0:\n"
        "    a.output.mkdir(parents=True, exist_ok=False)\n"
        "    (a.output / 'index.html').write_text('Non qualifies', encoding='utf-8')\n"
        f"raise SystemExit({report_code})\n", encoding="utf-8",
    )


def run_reports(project, *, day="2026-10-03"):
    command = ("try { & " + quote(project / "NYXAnnualReports.ps1")
               + " -PythonExecutable " + quote(sys.executable))
    if day:
        command += " -DeliveryDay " + quote(day)
    command += "; Write-Output ('WRAPPER_EXIT=' + $LASTEXITCODE) } catch { Write-Output ('WRAPPER_ERROR=' + $_.Exception.Message); exit 1 }"
    return powershell(command, cwd=project.parent)


def calls(project):
    path = project / "calls.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def test_reports_prepare_then_publish_and_refuse_to_overwrite(project):
    synthetic_pipeline(project)
    code, output = run_reports(project)
    assert code == 0, output
    assert "WRAPPER_EXIT=0" in output
    assert "NativeCommandError" not in output
    assert [call["action"] for call in calls(project)] == ["prepare", "reports"]
    assert all(call["delivery_day"] == "2026-10-03" for call in calls(project))
    html = project / "runs/nyx_annual_cpu_reports/2026-10-03/index.html"
    assert html.read_text() == "Non qualifies"
    assert not (project / "runs/nyx_annual_cpu_live/2026-10-03").exists()
    log_root = project / "runs/logs/nyx_annual_cpu_reports/2026-10-03"
    logs = [path for path in log_root.glob("*.log") if not path.name.endswith(".report.log")]
    assert len(logs) == 1
    log = logs[0].read_text(encoding="utf-8-sig")
    assert "Etape 1/2" in log and "Etape 2/2" in log
    report_logs = list(log_root.glob("*.report.log"))
    assert len(report_logs) == 1
    assert "Rapports et avertissement sur stderr" in report_logs[0].read_text(encoding="utf-8")
    assert str(report_logs[0]) in log
    code, output = run_reports(project)
    assert code == 1 and "immuable" in output
    assert len(calls(project)) == 2
    assert html.read_text() == "Non qualifies"


@pytest.mark.parametrize("prepare_code,report_code,expected", [(2, 0, ["prepare"]), (0, 7, ["prepare", "reports"])])
def test_failed_stage_stops_and_retains_durable_log(project, prepare_code, report_code, expected):
    synthetic_pipeline(project, prepare_code=prepare_code, report_code=report_code)
    code, output = run_reports(project)
    assert code == 1
    assert [call["action"] for call in calls(project)] == expected
    assert "WRAPPER_ERROR=" in output and "Journal :" in output
    assert "NativeCommandError" not in output
    assert not (project / "runs/nyx_annual_cpu_reports/2026-10-03").exists()
    logs = list((project / "runs/logs/nyx_annual_cpu_reports/2026-10-03").glob("*.log"))
    assert len(logs) == (1 if prepare_code else 2)


def test_default_delivery_is_tomorrow_in_paris(project):
    synthetic_pipeline(project)
    before = (datetime.now(ZoneInfo("Europe/Paris")).date() + timedelta(days=1)).isoformat()
    code, output = run_reports(project, day="")
    after = (datetime.now(ZoneInfo("Europe/Paris")).date() + timedelta(days=1)).isoformat()
    assert code == 0, output
    assert calls(project)[0]["delivery_day"] in {before, after}
    assert calls(project)[1]["delivery_day"] == calls(project)[0]["delivery_day"]


SCHEDULE_MOCKS = """
$global:Registrations = @()
function Get-TimeZone { [pscustomobject]@{Id='Romance Standard Time'} }
function New-ScheduledTaskPrincipal { param($UserId,$LogonType,$RunLevel) 'principal' }
function New-ScheduledTaskSettingsSet { param([switch]$AllowStartIfOnBatteries,[switch]$DontStopIfGoingOnBatteries,$MultipleInstances,$ExecutionTimeLimit) 'settings' }
function New-ScheduledTaskAction { param($Execute,$Argument,$WorkingDirectory) [pscustomobject]@{Execute=$Execute;Argument=$Argument;WorkingDirectory=$WorkingDirectory} }
function New-ScheduledTaskTrigger { param([switch]$Daily,[datetime]$At) $At.ToString('HH:mm') }
function Register-ScheduledTask { param($TaskName,$Action,$Trigger,$Settings,$Principal,$Description,[switch]$Force) $global:Registrations += [pscustomobject]@{Name=$TaskName;Action=$Action;Trigger=$Trigger;Description=$Description} }
"""


@pytest.mark.parametrize("daily", [False, True])
def test_installer_preserves_capture_and_adds_only_requested_reports(project, daily):
    shutil.copy2(ROOT / "Install-NYXAnnualTasks.ps1", project)
    python_path = project / ".venv-annual/Scripts/python.exe"
    python_path.parent.mkdir(parents=True)
    python_path.touch()
    command = SCHEDULE_MOCKS + "& " + quote(project / "Install-NYXAnnualTasks.ps1")
    if daily:
        command += " -EnableDailyReports"
    command += "; Write-Output ('TASKS=' + (ConvertTo-Json -InputObject @($global:Registrations) -Depth 6 -Compress))"
    code, output = powershell(command, cwd=project.parent)
    assert code == 0, output
    tasks = json.loads(next(line.removeprefix("TASKS=") for line in output.splitlines() if line.startswith("TASKS=")))
    assert [task["Name"] for task in tasks] == (["NYX CWE PIT", "NYX CWE CPU rapports"] if daily else ["NYX CWE PIT"])
    assert tasks[0]["Trigger"] == ["07:00", "07:15", "07:30", "07:45", "07:55"]
    assert tasks[0]["Action"]["Argument"].endswith('run_nyx_annual_scheduled.py" --action capture')
    if daily:
        task = tasks[1]
        assert task["Trigger"] == "08:05"
        assert Path(task["Action"]["Execute"]).name in {"powershell.exe", "pwsh.exe"}
        assert task["Action"]["Argument"] == '-NoProfile -NonInteractive -WindowStyle Hidden -File "' + str(project / "NYXAnnualReports.ps1") + '"'
        assert task["Action"]["WorkingDirectory"] == str(project)
        assert "non qualifies" in task["Description"]


def test_installer_rejects_mutually_exclusive_modes_before_any_task_change(project):
    shutil.copy2(ROOT / "Install-NYXAnnualTasks.ps1", project)
    command = (SCHEDULE_MOCKS + "try { & " + quote(project / "Install-NYXAnnualTasks.ps1")
               + " -EnableDailyReports -EnableQualifiedForecast; exit 9 } catch { Write-Output $_.Exception.Message; Write-Output ('COUNT=' + $global:Registrations.Count) }")
    code, output = powershell(command, cwd=project.parent)
    assert code == 0, output
    assert "Choisir soit" in output and "COUNT=0" in output
