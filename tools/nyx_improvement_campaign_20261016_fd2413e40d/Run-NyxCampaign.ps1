[CmdletBinding()]
param(
    [ValidateSet('Screen','Replay','Challengers','Archive','Annual','Summary')]
    [string]$Phase = 'Screen',
    [string]$AnnualRepo = (Join-Path $env:USERPROFILE 'NYX_CWE_CPU'),
    [string]$AppRepo = (Join-Path $env:USERPROFILE 'chronos2_v1'),
    [string]$BaseSession = '',
    [string]$SessionDir = '',
    [switch]$NewSession
)
$ErrorActionPreference = 'Stop'
$NyxPython = Join-Path $AnnualRepo '.venv-annual\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $NyxPython -PathType Leaf)) { throw 'Python annuel absent : corriger -AnnualRepo vers le depot existant.' }
$NyxRoot = Join-Path $AnnualRepo 'runs\nyx_improvement_campaign'
New-Item -ItemType Directory -Path $NyxRoot -Force | Out-Null
$NyxLogDir = Join-Path $NyxRoot 'launcher_logs'
New-Item -ItemType Directory -Path $NyxLogDir -Force | Out-Null
$NyxLog = Join-Path $NyxLogDir ((Get-Date -Format 'yyyyMMdd_HHmmss') + '_' + $Phase.ToLowerInvariant() + '_' + [guid]::NewGuid().ToString('N').Substring(0,6) + '.log')
$NyxArgs = @('-u', (Join-Path $PSScriptRoot 'campaign.py'), '--phase', $Phase, '--annual-repo', $AnnualRepo, '--app-repo', $AppRepo)
if ($BaseSession) { $NyxArgs += @('--base-session', $BaseSession) }
if ($SessionDir) { $NyxArgs += @('--session-dir', $SessionDir) }
if ($NewSession) { $NyxArgs += '--new-session' }
$NyxTranscript = $false
try {
    Start-Transcript -Path $NyxLog -Append | Out-Null
    $NyxTranscript = $true
    Write-Host ('Journal : ' + $NyxLog)
    & $NyxPython @NyxArgs
    if ($LASTEXITCODE -ne 0) { throw ('Campagne interrompue ou refusee. Journal : ' + $NyxLog) }
    $NyxLast = Join-Path $NyxRoot 'last_campaign.json'
    if (Test-Path -LiteralPath $NyxLast -PathType Leaf) {
        $NyxSaved = Get-Content -LiteralPath $NyxLast -Raw -Encoding UTF8 | ConvertFrom-Json
        $NyxReportDir = if ($SessionDir) { $SessionDir } else { [string]$NyxSaved.session_dir }
        $NyxIndex = Join-Path $NyxReportDir 'index.html'
        if (Test-Path -LiteralPath $NyxIndex -PathType Leaf) { Start-Process -FilePath $NyxIndex | Out-Null }
    }
} finally {
    if ($NyxTranscript) { Stop-Transcript | Out-Null }
}
