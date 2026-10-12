# Ejecuta el pipeline v002; falla en la primera etapa incompleta. La descarga se reanuda.
[CmdletBinding()]
param([switch]$DatasetOnly, [switch]$SmokeOnly, [switch]$SkipInstall)
$ErrorActionPreference = 'Stop'
if ($DatasetOnly -and $SmokeOnly) { throw 'DatasetOnly y SmokeOnly son mutuamente excluyentes.' }
$repoDirectory = Split-Path -Parent $PSScriptRoot
Push-Location $repoDirectory
try {
    $venvPython = Join-Path $repoDirectory '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $venvPython)) {
        & python -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw 'No se pudo crear el entorno virtual.' }
    }
    function Invoke-Python {
        param([string[]]$Arguments)
        & $venvPython -u @Arguments
        if ($LASTEXITCODE -ne 0) { throw "Falló Python (exit $LASTEXITCODE): $Arguments" }
    }
    Invoke-Python -Arguments @('-c', 'import sys; assert sys.version_info[:2] == (3, 12), "Se requiere Python 3.12"')
    if (-not $SkipInstall) {
        Invoke-Python -Arguments @('-m', 'pip', 'install', '-r', 'requirements.txt')
        if (-not $DatasetOnly) {
            Invoke-Python -Arguments @('-m', 'pip', 'install', '-r', 'requirements-train.txt')
        }
    }
    Invoke-Python -Arguments @('scripts/data/acquire_datasets.py', 'fungitastic')
    Invoke-Python -Arguments @('scripts/data/build_dataset.py', '--config', 'configs/dataset_v002.yaml', '--prepare-images')
    Invoke-Python -Arguments @('scripts/data/download_images.py', '--version', 'v002', '--images-csv', 'data/interim/v002/images.csv', '--out', 'data/raw/images/fungitastic_v002')
    Invoke-Python -Arguments @('scripts/data/build_dataset.py', '--config', 'configs/dataset_v002.yaml')
    Invoke-Python -Arguments @('scripts/data/build_dataset.py', '--config', 'configs/dataset_v002.yaml', '--check')
    if (-not $DatasetOnly) {
        Invoke-Python -Arguments @('scripts/train/check_env.py', '--config', 'configs/train/ma_resnet18_v002.yaml')
        Invoke-Python -Arguments @('scripts/train/train_ma.py', '--config', 'configs/train/ma_resnet18_v002.yaml', '--smoke')
        if (-not $SmokeOnly) {
            Invoke-Python -Arguments @('scripts/train/train_ma.py', '--config', 'configs/train/ma_resnet18_v002.yaml')
        }
    }
} finally {
    Pop-Location
}
