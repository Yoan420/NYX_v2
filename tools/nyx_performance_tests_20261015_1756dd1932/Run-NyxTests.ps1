[CmdletBinding()]
param(
    [ValidateSet('Pilot', 'Replay', 'Archive')][string]$Phase = 'Pilot',
    [string]$AnnualRepo = (Join-Path $env:USERPROFILE 'NYX_CWE_CPU'),
    [string]$AppRepo = (Join-Path $env:USERPROFILE 'chronos2_v1'),
    [string]$SessionDir = '',
    [ValidateRange(1,128)][int]$Threads = 8
)
$ErrorActionPreference = 'Stop'
$NyxPython = Join-Path $AnnualRepo '.venv-annual\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $NyxPython -PathType Leaf)) { throw 'Python annuel absent : corriger -AnnualRepo vers le depot existant.' }
$NyxTestRoot = Join-Path $AnnualRepo 'runs\nyx_performance_tests'
New-Item -ItemType Directory -Path $NyxTestRoot -Force | Out-Null
$NyxLastSession = Join-Path $NyxTestRoot 'last_session.json'
if (-not $SessionDir) {
    if ($Phase -eq 'Pilot') {
        $SessionDir = Join-Path $NyxTestRoot ((Get-Date -Format 'yyyyMMdd_HHmmss') + '_' + [guid]::NewGuid().ToString('N').Substring(0,6))
    } else {
        if (-not (Test-Path -LiteralPath $NyxLastSession -PathType Leaf)) { throw 'Lancer d abord la phase Pilot, ou fournir -SessionDir.' }
        $NyxSavedSession = Get-Content -LiteralPath $NyxLastSession -Raw -Encoding UTF8 | ConvertFrom-Json
        $SessionDir = [string]$NyxSavedSession.session_dir
    }
}
$SessionDir = [IO.Path]::GetFullPath($SessionDir)
$NyxSessionFile = Join-Path $SessionDir 'session.json'
if (Test-Path -LiteralPath $NyxSessionFile -PathType Leaf) {
    $NyxSession = Get-Content -LiteralPath $NyxSessionFile -Raw -Encoding UTF8 | ConvertFrom-Json
    if ([string]$NyxSession.tools_dir -ne $PSScriptRoot) { throw 'Cette session appartient a une autre version des scripts de test.' }
    $AnnualRepo = [string]$NyxSession.annual_repo
    $AppRepo = [string]$NyxSession.app_repo
    $Threads = [int]$NyxSession.threads
    $NyxPython = Join-Path $AnnualRepo '.venv-annual\Scripts\python.exe'
} else {
    if ($Phase -ne 'Pilot') { throw 'session.json absent : lancer Pilot pour initialiser cette session.' }
    New-Item -ItemType Directory -Path $SessionDir -Force | Out-Null
    $NyxSession = [pscustomobject]@{ session_dir=$SessionDir; annual_repo=$AnnualRepo; app_repo=$AppRepo; tools_dir=$PSScriptRoot; threads=$Threads; seed_run='' }
    $NyxSession | ConvertTo-Json | Set-Content -LiteralPath $NyxSessionFile -Encoding UTF8
    $NyxSession | ConvertTo-Json | Set-Content -LiteralPath $NyxLastSession -Encoding UTF8
}
function Invoke-NyxResearchPython {
    param([string[]]$Arguments)
    & $NyxPython -u @Arguments
    if ($LASTEXITCODE -ne 0) { throw ('Test interrompu ou refuse (code ' + $LASTEXITCODE + '). Consulter le journal de cette session.') }
}
$NyxLog = Join-Path $SessionDir ('console_' + $Phase.ToLowerInvariant() + '_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '.log')
$NyxTranscriptStarted = $false
try {
    Start-Transcript -Path $NyxLog -Append | Out-Null
    $NyxTranscriptStarted = $true
    Write-Host ('Dossier des tests : ' + $SessionDir)
    Write-Host ('Phase : ' + $Phase)
    if ($Phase -eq 'Pilot') {
        $NyxInventoryDir = Join-Path $SessionDir 'inventory'
        $NyxInventoryArgs = @((Join-Path $PSScriptRoot 'replay.py'), 'inventory', '--annual-repo', $AnnualRepo, '--app-repo', $AppRepo, '--output', $NyxInventoryDir)
        if ([string]$NyxSession.seed_run) { $NyxInventoryArgs += @('--seed-run', [string]$NyxSession.seed_run) }
        Invoke-NyxResearchPython -Arguments $NyxInventoryArgs
        $NyxInventory = Get-Content -LiteralPath (Join-Path $NyxInventoryDir 'inventory.json') -Raw -Encoding UTF8 | ConvertFrom-Json
        $NyxSession.seed_run = [string]$NyxInventory.seed_run
        $NyxSession | ConvertTo-Json | Set-Content -LiteralPath $NyxSessionFile -Encoding UTF8
        $NyxSession | ConvertTo-Json | Set-Content -LiteralPath $NyxLastSession -Encoding UTF8
        $NyxInputs = Join-Path $SessionDir 'reference_inputs'
        if (-not (Test-Path -LiteralPath (Join-Path $NyxInputs 'OOF.manifest.json') -PathType Leaf)) {
            Invoke-NyxResearchPython -Arguments @((Join-Path $PSScriptRoot 'reference_pilot.py'), '--annual-repo', $AnnualRepo, '--app-repo', $AppRepo, '--seed-run', $NyxSession.seed_run, '--output', $NyxInputs)
        }
        $NyxResults = Join-Path $SessionDir 'reference_study'
        if (-not (Test-Path -LiteralPath (Join-Path $NyxResults 'report.html') -PathType Leaf)) {
            Invoke-NyxResearchPython -Arguments @((Join-Path $PSScriptRoot 'calibration.py'), '--input', (Join-Path $NyxInputs 'OOF.csv'), '--output', $NyxResults, '--scope', 'reference_component_pilot', '--component-columns', 'reference_price,nyx_baseline,test2_price', '--control-column', 'nyx_baseline', '--warmup-days', '45', '--holdout-days', '30')
        }
        Start-Process -FilePath (Join-Path $NyxResults 'report.html') | Out-Null
        Write-Host 'Pilote termine : composants historiques sauvegardes, sans rejeu des etats de fondation.'
        Write-Host ('Origines disponibles pour le rejeu final : ' + $NyxInventory.conditional_j1_available_days)
    } elseif ($Phase -eq 'Replay') {
        $NyxReplayDir = Join-Path $SessionDir 'final_replay'
        $NyxSeedRun = [string]$NyxSession.seed_run
        $NyxInventoryDir = Join-Path $SessionDir 'inventory'
        Invoke-NyxResearchPython -Arguments @((Join-Path $PSScriptRoot 'replay.py'), 'inventory', '--annual-repo', $AnnualRepo, '--seed-run', $NyxSeedRun, '--output', $NyxInventoryDir)
        $NyxInventory = Get-Content -LiteralPath (Join-Path $NyxInventoryDir 'inventory.json') -Raw -Encoding UTF8 | ConvertFrom-Json
        if ([int]$NyxInventory.conditional_j1_available_days -lt 1) { throw 'Aucune origine finale complete disponible. Le pilote precedent reste consultable ; inventory.json donne la couverture manquante.' }
        Write-Host 'Recalcul des references et des experts prix sur chaque origine disponible. Les origines deja scellees seront verifiees et reutilisees.'
        Invoke-NyxResearchPython -Arguments @((Join-Path $PSScriptRoot 'replay.py'), 'replay', '--annual-repo', $AnnualRepo, '--seed-run', $NyxSeedRun, '--output', $NyxReplayDir, '--days', '0', '--threads', [string]$Threads, '--implementation-root', (Join-Path $PSScriptRoot 'annual_extension'))
        $NyxResults = Join-Path $SessionDir 'final_replay_study'
        if (-not (Test-Path -LiteralPath (Join-Path $NyxResults 'report.html') -PathType Leaf)) {
            Invoke-NyxResearchPython -Arguments @((Join-Path $PSScriptRoot 'final_study.py'), '--input', (Join-Path $NyxReplayDir 'OOF.csv'), '--output', $NyxResults, '--annual-repo', $AnnualRepo, '--app-repo', $AppRepo, '--calibrate-when-ready')
        }
        Start-Process -FilePath (Join-Path $NyxResults 'report.html') | Out-Null
    } else {
        $NyxSuffix = (Get-Date -Format 'yyyyMMdd_HHmmss') + '_' + [guid]::NewGuid().ToString('N').Substring(0,6)
        $NyxFreshInputs = Join-Path $SessionDir ('archive_inputs_' + $NyxSuffix)
        Invoke-NyxResearchPython -Arguments @((Join-Path $PSScriptRoot 'reference_pilot.py'), '--annual-repo', $AnnualRepo, '--app-repo', $AppRepo, '--seed-run', ([string]$NyxSession.seed_run), '--output', $NyxFreshInputs)
        $NyxArchiveDir = Join-Path $SessionDir ('archive_' + $NyxSuffix)
        Invoke-NyxResearchPython -Arguments @((Join-Path $PSScriptRoot 'replay.py'), 'collect', '--annual-repo', $AnnualRepo, '--app-repo', $AppRepo, '--output', $NyxArchiveDir, '--labels', (Join-Path $NyxFreshInputs 'report_labels.csv'))
        $NyxArchiveAudit = Get-Content -LiteralPath (Join-Path $NyxArchiveDir 'OOF.manifest.json') -Raw -Encoding UTF8 | ConvertFrom-Json
        if ([int]$NyxArchiveAudit.scorable_rows -gt 0) {
            $NyxResults = Join-Path $NyxArchiveDir 'study'
            Invoke-NyxResearchPython -Arguments @((Join-Path $PSScriptRoot 'final_study.py'), '--input', (Join-Path $NyxArchiveDir 'OOF.csv'), '--output', $NyxResults, '--annual-repo', $AnnualRepo, '--app-repo', $AppRepo, '--calibrate-when-ready')
            Start-Process -FilePath (Join-Path $NyxResults 'report.html') | Out-Null
        } else {
            Write-Host 'Previsions archivees. Les journees encore sans prix EPEX complets restent dans pending_forecasts.csv.'
        }
        Write-Host ('Archive J+1 / J+2 : ' + $NyxArchiveDir)
    }
    Write-Host ('Resultats : ' + $SessionDir)
    Write-Host ('Journal : ' + $NyxLog)
} finally {
    if ($NyxTranscriptStarted) { Stop-Transcript | Out-Null }
}
