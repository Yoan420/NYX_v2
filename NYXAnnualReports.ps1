<# Prepare et produit les rapports annuels CPU non qualifies pour demain Paris. #>
[CmdletBinding()]
param(
    [ValidatePattern('^$|^\d{4}-\d{2}-\d{2}$')]
    [string]$DeliveryDay = '',
    [string]$PythonExecutable = ''
)
$ErrorActionPreference = 'Stop'
$NyxRoot = $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($PythonExecutable)) {
    $PythonExecutable = Join-Path $NyxRoot '.venv-annual\Scripts\python.exe'
}
if (-not (Test-Path -LiteralPath $PythonExecutable -PathType Leaf)) {
    throw 'Python annuel absent. Executer Setup-NYXAnnualCPU.ps1 ou fournir -PythonExecutable.'
}
$NyxPrepare = Join-Path $NyxRoot 'NYXAnnualCPU.ps1'
$NyxReports = Join-Path $NyxRoot 'run_nyx_annual_cpu_reports.py'
foreach ($NyxEntry in @($NyxPrepare, $NyxReports)) {
    if (-not (Test-Path -LiteralPath $NyxEntry -PathType Leaf)) {
        throw "Entree annuelle absente : $NyxEntry. Mettre a jour le depot avant de lancer les rapports."
    }
}
Push-Location -LiteralPath $NyxRoot
$NyxTranscriptStarted = $false
try {
    if (-not $DeliveryDay) {
        $NyxDayResult = @(& $PythonExecutable -c "from datetime import datetime, timedelta; from zoneinfo import ZoneInfo; print((datetime.now(ZoneInfo('Europe/Paris')).date() + timedelta(days=1)).isoformat())")
        if ($LASTEXITCODE -ne 0 -or $NyxDayResult.Count -ne 1 -or $NyxDayResult[0] -notmatch '^\d{4}-\d{2}-\d{2}$') {
            throw 'Impossible de determiner demain dans le fuseau Europe/Paris avec Python annuel.'
        }
        $DeliveryDay = $NyxDayResult[0]
    }
    $NyxBundle = Join-Path $NyxRoot ('runs\live\nyx_annual_cpu\' + $DeliveryDay)
    $NyxOutput = Join-Path $NyxRoot ('runs\nyx_annual_cpu_reports\' + $DeliveryDay)
    if (Test-Path -LiteralPath $NyxOutput) {
        throw "La sortie existe deja et reste immuable : $NyxOutput. Conserver ce rapport et consulter son manifeste avant toute nouvelle tentative."
    }
    $NyxLogDirectory = Join-Path $NyxRoot ('runs\logs\nyx_annual_cpu_reports\' + $DeliveryDay)
    New-Item -ItemType Directory -Path $NyxLogDirectory -Force | Out-Null
    $NyxAttempt = [datetime]::UtcNow.ToString('yyyyMMddTHHmmssZ') + '_' + [guid]::NewGuid().ToString('N')
    $NyxLogPath = Join-Path $NyxLogDirectory ('reports_' + $NyxAttempt + '.log')
    $NyxReportLogPath = Join-Path $NyxLogDirectory ('reports_' + $NyxAttempt + '.report.log')
    Start-Transcript -LiteralPath $NyxLogPath -NoClobber | Out-Null
    $NyxTranscriptStarted = $true
    Write-Host "Rapports annuels CPU non qualifies - livraison $DeliveryDay"
    Write-Host "Journal : $NyxLogPath"
    Write-Host 'Etape 1/2 : preparation des sources et modeles annuels CPU.'
    & $NyxPrepare -Action prepare -DeliveryDay $DeliveryDay -PythonExecutable $PythonExecutable
    $NyxPrepareExitCode = $LASTEXITCODE
    if ($NyxPrepareExitCode -ne 0) {
        throw "Preparation bloquee (code $NyxPrepareExitCode). Les rapports ne sont pas lances. Journal : $NyxLogPath"
    }
    Write-Host 'Etape 2/2 : generation des rapports non qualifies.'
    Write-Host "Journal detaille des rapports : $NyxReportLogPath"
    & $PythonExecutable -u $NyxReports --bundle $NyxBundle --delivery-day $DeliveryDay --output $NyxOutput --log-file $NyxReportLogPath
    $NyxReportExitCode = $LASTEXITCODE
    if ($NyxReportExitCode -ne 0) {
        throw "Rapports bloques (code $NyxReportExitCode). Journal : $NyxReportLogPath"
    }
    Write-Host "Rapports non qualifies : $NyxOutput"
    $global:LASTEXITCODE = $NyxReportExitCode
} finally {
    if ($NyxTranscriptStarted) { Stop-Transcript | Out-Null }
    Pop-Location
}
