# Ejecuta solo el entrenamiento, en la terminal actual, con barras de tqdm.
[CmdletBinding()]
param([switch]$Smoke, [string]$Resume, [string]$InitCheckpoint, [switch]$PauseAfterEpoch,
      [string]$Config = 'configs/train/ma_resnet18_v002.yaml', [switch]$SkipInstall, [string]$Device)
$ErrorActionPreference = 'Stop'
Write-Host '[1/4] Iniciando lanzador de entrenamiento...' -ForegroundColor Cyan
$repoDirectory = Split-Path -Parent $PSScriptRoot
$previousProgressSetting = [Environment]::GetEnvironmentVariable('VIRTUS_PROGRESS', 'Process')
Push-Location $repoDirectory
try {
    Write-Host "Repositorio: $repoDirectory"
    Write-Host "Configuracion: $Config"
    $pythonExecutable = Join-Path $repoDirectory '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $pythonExecutable)) {
        throw 'Primero prepara el entorno con scripts/run_baseline.ps1 -DatasetOnly.'
    }
    if (-not (Test-Path -LiteralPath $Config)) { throw "No existe la configuracion: $Config" }
    Write-Host '[2/4] Entorno encontrado; comprobando corridas activas...' -ForegroundColor Cyan
    $processFile = Join-Path $repoDirectory 'runs\baseline_v002.process.json'
    if (Test-Path -LiteralPath $processFile) {
        $record = Get-Content -LiteralPath $processFile -Raw | ConvertFrom-Json
        if ($record.pid) {
            $activeProcess = Get-Process -Id $record.pid -ErrorAction SilentlyContinue
            if ($activeProcess -and [Math]::Abs(($activeProcess.StartTime.ToUniversalTime() - [datetime]::Parse($record.started_utc).ToUniversalTime()).TotalSeconds) -lt 10) {
                throw 'Ya hay un entrenamiento activo. Puedes verlo con .\.venv\Scripts\python.exe scripts/train/watch_training.py.'
            }
        }
    }
    if (-not $SkipInstall) {
        Write-Host '[3/4] Preparando dependencias...' -ForegroundColor Cyan
        & $pythonExecutable -m pip install -r requirements-train.txt
        if ($LASTEXITCODE -ne 0) { throw 'No se pudieron instalar las dependencias de entrenamiento.' }
    } else { Write-Host '[3/4] Reutilizando dependencias instaladas.' -ForegroundColor Cyan }
    $trainingArguments = @('-u', 'scripts/train/train_ma.py', '--config', $Config)
    if ($Smoke) { $trainingArguments += '--smoke' }
    if ($Resume) { $trainingArguments += @('--resume', $Resume) }
    if ($InitCheckpoint) { $trainingArguments += @('--init-checkpoint', $InitCheckpoint) }
    if ($Device) { $trainingArguments += @('--device', $Device) }
    if ($PauseAfterEpoch) { $trainingArguments += '--pause-after-epoch' }
    $shellProcess = Get-Process -Id $PID
    New-Item -ItemType Directory -Path 'runs' -Force | Out-Null
    @{pid=$PID;started_utc=$shellProcess.StartTime.ToUniversalTime().ToString('o');config=$Config} | ConvertTo-Json | Set-Content -LiteralPath $processFile -Encoding UTF8
    try {
        $env:VIRTUS_PROGRESS = '1'
        Write-Host '[4/4] Iniciando Python; se mostrara cada etapa y su progreso...' -ForegroundColor Cyan
        & $pythonExecutable @trainingArguments
        if ($LASTEXITCODE -ne 0) { throw "Entrenamiento interrumpido (exit $LASTEXITCODE)." }
        Write-Host 'Python termino correctamente. Revisa la carpeta de corrida indicada.' -ForegroundColor Green
    } finally {
        @{pid=$null;config=$Config} | ConvertTo-Json | Set-Content -LiteralPath $processFile -Encoding UTF8
    }
} finally {
    [Environment]::SetEnvironmentVariable('VIRTUS_PROGRESS', $previousProgressSetting, 'Process')
    Pop-Location
}
