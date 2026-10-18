<# Programme la capture07h et, sur demande, les rapports ou la production08h05. #>
[CmdletBinding()]
param(
    [switch]$EnableQualifiedForecast,
    [switch]$EnableDailyReports
)
$ErrorActionPreference = 'Stop'
if ($EnableQualifiedForecast -and $EnableDailyReports) {
    throw 'Choisir soit -EnableDailyReports (rapports non qualifies), soit -EnableQualifiedForecast (production qualifiee). Aucune tache modifiee.'
}
$NyxPython = Join-Path $PSScriptRoot '.venv-annual\Scripts\python.exe'
$NyxPythonw = Join-Path $PSScriptRoot '.venv-annual\Scripts\pythonw.exe'
$NyxRunner = Join-Path $PSScriptRoot 'run_nyx_annual_scheduled.py'
if (-not (Test-Path -LiteralPath $NyxPython -PathType Leaf)) { throw 'Executer Setup-NYXAnnualCPU.ps1 auparavant.' }
if ((Get-TimeZone).Id -notin @('Romance Standard Time', 'W. Europe Standard Time')) {
    throw 'Ces horaires exigent le fuseau Windows Paris/CET-CEST. Corriger le fuseau avant installation.'
}
$NyxPrincipal = New-ScheduledTaskPrincipal -UserId ([Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
$NyxSettings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Hours 12)
$NyxCapture = New-ScheduledTaskAction -Execute $NyxPythonw -Argument ('"' + $NyxRunner + '" --action capture') -WorkingDirectory $PSScriptRoot
$NyxTriggers = @(foreach ($NyxMinute in 0,15,30,45,55) { New-ScheduledTaskTrigger -Daily -At ([datetime]::Today.AddHours(7).AddMinutes($NyxMinute)) })
Register-ScheduledTask -TaskName 'NYX CWE PIT' -Action $NyxCapture -Trigger $NyxTriggers -Settings $NyxSettings -Principal $NyxPrincipal -Description 'Capture JAO, hydro, echanges et actualise leurs historiques avant 08h Paris' -Force
if ($EnableQualifiedForecast) {
    Push-Location -LiteralPath $PSScriptRoot
    try { & $NyxPython -c "from chronos2_hourly.nyx_annual_cpu_live import verify_activation; verify_activation(); print('Qualification complete valide')" }
    finally { Pop-Location }
    if ($LASTEXITCODE -ne 0) { throw 'Prevision automatique non installee : qualification complete absente. La capture reste installee.' }
    $NyxForecast = New-ScheduledTaskAction -Execute $NyxPythonw -Argument ('"' + $NyxRunner + '" --action forecast') -WorkingDirectory $PSScriptRoot
    $NyxTrigger = New-ScheduledTaskTrigger -Daily -At ([datetime]::Today.AddHours(8).AddMinutes(5))
    Register-ScheduledTask -TaskName 'NYX CWE CPU production' -Action $NyxForecast -Trigger $NyxTrigger -Settings $NyxSettings -Principal $NyxPrincipal -Description 'Saturn, reentrainement CPU et quatre previsions annuelles qualifiees' -Force
}
if ($EnableDailyReports) {
    $NyxReports = Join-Path $PSScriptRoot 'NYXAnnualReports.ps1'
    if (-not (Test-Path -LiteralPath $NyxReports -PathType Leaf)) { throw 'NYXAnnualReports.ps1 absent. Mettre a jour le depot.' }
    # Reuse the host that ran this installer and its execution policy.
    $NyxPowerShell = Join-Path $PSHOME 'pwsh.exe'
    if (-not (Test-Path -LiteralPath $NyxPowerShell -PathType Leaf)) {
        $NyxPowerShell = Join-Path $PSHOME 'powershell.exe'
    }
    $NyxReportsAction = New-ScheduledTaskAction -Execute $NyxPowerShell -Argument ('-NoProfile -NonInteractive -WindowStyle Hidden -File "' + $NyxReports + '"') -WorkingDirectory $PSScriptRoot
    $NyxReportsTrigger = New-ScheduledTaskTrigger -Daily -At ([datetime]::Today.AddHours(8).AddMinutes(5))
    Register-ScheduledTask -TaskName 'NYX CWE CPU rapports' -Action $NyxReportsAction -Trigger $NyxReportsTrigger -Settings $NyxSettings -Principal $NyxPrincipal -Description 'Preparation CPU et quatre rapports annuels non qualifies, sans activation de production' -Force
}
Write-Host 'Le poste doit rester allume, eveille, connecte au reseau, avec la session ouverte.'
