<# Lance les recettes annuelles seuils pour demain, a toute heure. #>
[CmdletBinding()]
param(
    [ValidatePattern('^$|^\d{4}-\d{2}-\d{2}$')]
    [string]$DeliveryDay = '',
    [string]$PythonExecutable = '',
    [string]$ResumeRun = '',
    [switch]$NoOpen
)
$ErrorActionPreference = 'Stop'
if (-not $PythonExecutable) {
    $PythonExecutable = Join-Path $PSScriptRoot '.venv-annual\Scripts\python.exe'
}
if (-not (Test-Path -LiteralPath $PythonExecutable -PathType Leaf)) {
    throw 'Python annuel absent. Executer Setup-NYXAnnualCPU.ps1 ou fournir -PythonExecutable.'
}
$NyxResult = Join-Path $PSScriptRoot ('runs\logs\nyx_annual_anytime\launch_' + [guid]::NewGuid().ToString('N') + '.json')
$NyxArguments = @('-u', (Join-Path $PSScriptRoot 'run_nyx_annual_anytime.py'), '--result-file', $NyxResult)
if ($DeliveryDay) { $NyxArguments += @('--delivery-day', $DeliveryDay) }
if ($ResumeRun) { $NyxArguments += @('--resume-run', $ResumeRun) }
Push-Location -LiteralPath $PSScriptRoot
try {
    & $PythonExecutable @NyxArguments
    $NyxCode = $LASTEXITCODE
    if (Test-Path -LiteralPath $NyxResult -PathType Leaf) {
        $NyxDiagnostic = Get-Content -LiteralPath $NyxResult -Raw -Encoding UTF8 | ConvertFrom-Json
    }
    if ($NyxCode -ne 0) {
        throw ("Calcul bloque. Diagnostic : {0}. Dossier de reprise : {1}. Cause : {2}" -f $NyxResult, $NyxDiagnostic.run_directory, $NyxDiagnostic.error)
    }
    Write-Host "Rapports termines : $($NyxDiagnostic.index)"
    if (-not $NoOpen) { Invoke-Item -LiteralPath $NyxDiagnostic.index }
    $global:LASTEXITCODE = $NyxCode
} finally { Pop-Location }
