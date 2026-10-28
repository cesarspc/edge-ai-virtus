# Compara recortes usando solo val; muestra etapas y tqdm en la terminal actual.
[CmdletBinding()]
param([string]$Resume, [int]$PauseAfterBatches = 0,
      [string]$Config = 'configs/experiments/crops_v002.yaml')
$ErrorActionPreference = 'Stop'
Write-Host '[INICIO] Comparacion de recortes con C congelado...' -ForegroundColor Cyan
$repoDirectory = Split-Path -Parent $PSScriptRoot
$previousProgressSetting = [Environment]::GetEnvironmentVariable('VIRTUS_PROGRESS', 'Process')
Push-Location $repoDirectory
try {
    $pythonExecutable = Join-Path $repoDirectory '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $pythonExecutable)) { throw 'No existe el entorno .venv.' }
    $processFile = Join-Path $repoDirectory 'runs\baseline_v002.process.json'
    if (Test-Path -LiteralPath $processFile) {
        $record = Get-Content -LiteralPath $processFile -Raw | ConvertFrom-Json
        if ($record.pid) {
            $activeProcess = Get-Process -Id $record.pid -ErrorAction SilentlyContinue
            if ($activeProcess -and [Math]::Abs(($activeProcess.StartTime.ToUniversalTime() - [datetime]::Parse($record.started_utc).ToUniversalTime()).TotalSeconds) -lt 10) {
                throw 'Ya existe una tarea de entrenamiento o recortes activa.'
            }
        }
    }
    $studyArguments = @('-u', 'scripts/train/evaluate_crop_sensitivity.py', '--config', $Config)
    if ($Resume) { $studyArguments += @('--resume', $Resume) }
    if ($PauseAfterBatches -gt 0) { $studyArguments += @('--pause-after-batches', "$PauseAfterBatches") }
    $shellProcess = Get-Process -Id $PID
    New-Item -ItemType Directory -Path 'runs' -Force | Out-Null
    @{pid=$PID;started_utc=$shellProcess.StartTime.ToUniversalTime().ToString('o');config=$Config} | ConvertTo-Json | Set-Content -LiteralPath $processFile -Encoding UTF8
    try {
        $env:VIRTUS_PROGRESS = '1'
        & $pythonExecutable @studyArguments
        if ($LASTEXITCODE -ne 0) { throw "Estudio interrumpido (exit $LASTEXITCODE); los bloques completos se conservan." }
        Write-Host 'Estudio guardado. Revisa la carpeta indicada por Python.' -ForegroundColor Green
    } finally {
        @{pid=$null;config=$Config} | ConvertTo-Json | Set-Content -LiteralPath $processFile -Encoding UTF8
    }
} finally {
    [Environment]::SetEnvironmentVariable('VIRTUS_PROGRESS', $previousProgressSetting, 'Process')
    Pop-Location
}
