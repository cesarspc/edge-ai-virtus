param(
    [string]$RepoRoot = (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)),
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path $RepoRoot).Path
$Upstream = Join-Path $RepoRoot "tmp\FungiTastic-upstream"
$SavePath = Join-Path $RepoRoot "data\raw\fungitastic"
$LogPath = Join-Path $RepoRoot "fungitastic_climatic_download.log"
$DownloadScript = Join-Path $Upstream "dataset\download.py"
$Archive = Join-Path $SavePath "FungiTastic\climatic.zip"
$ManifestPath = Join-Path (Split-Path -Parent $Archive) "climatic_download_manifest.json"
$ClimateCsvs = @(
    (Join-Path (Split-Path -Parent $Archive) "FungiTastic-Climatic-Average.csv"),
    (Join-Path (Split-Path -Parent $Archive) "FungiTastic-Climatic-Timeseries.csv")
)

New-Item -ItemType Directory -Force -Path $SavePath | Out-Null
Push-Location $RepoRoot
Start-Transcript -Path $LogPath -Append | Out-Null
try {
    Write-Host "[1/3] Comprobando herramientas..."
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        throw "No se encontró git. Instala Git for Windows y vuelve a ejecutar este script."
    }
    $Wget = Get-Command wget.exe -ErrorAction SilentlyContinue
    $Curl = Get-Command curl.exe -ErrorAction SilentlyContinue
    if (-not $Wget -and -not $Curl) { throw "Se requiere wget.exe o curl.exe para descargar el paquete climático." }

    if (-not (Test-Path $DownloadScript)) {
        Write-Host "[2/3] Descargando el script oficial de FungiTastic (sin imágenes)..."
        git clone --depth 1 https://github.com/BohemianVRA/FungiTastic.git $Upstream
        if ($LASTEXITCODE -ne 0) { throw "No se pudo clonar el repositorio oficial (git exit $LASTEXITCODE)." }
    } else {
        Write-Host "[2/3] Script oficial ya disponible; se reutiliza."
    }

    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Archive) | Out-Null
    if ((Test-Path $Archive) -and (Test-Path $ManifestPath)) {
        $savedManifest = Get-Content -Raw -LiteralPath $ManifestPath | ConvertFrom-Json
        $savedHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $Archive).Hash.ToLowerInvariant()
        if ($savedHash -ne $savedManifest.sha256) { throw "El SHA-256 del paquete no coincide con el manifiesto: $ManifestPath" }
        $extracted = @($ClimateCsvs | Where-Object { (Test-Path $_) -and (Get-Item -LiteralPath $_).Length -gt 0 }).Count
        if ($extracted -eq $ClimateCsvs.Count) {
            Write-Host "Checkpoint válido: paquete descargado y extraído anteriormente. SHA-256: $savedHash"
            Write-Host "No se requiere repetir la descarga. Log: $LogPath"
            return
        }
        Write-Host "El ZIP está verificado, pero falta la extracción. Se extraerá sin volver a descargar."
    }

    if (-not (Test-Path $ManifestPath)) {
        Write-Host "[3/3] Descargando únicamente el paquete climático; el cliente mostrará el progreso. Si se interrumpe, vuelve a ejecutar este script para continuar."
        # The current official directory publishes climaticData.zip, while download.py
        # still looks for the local filename climatic.zip before extracting it.
        $ClimateUrl = "https://cmp.felk.cvut.cz/datagrid/FungiTastic/shared/download/climaticData.zip"
        if ($Wget) {
            $UpstreamArchive = Join-Path (Split-Path -Parent $Archive) "climaticData.zip"
            & wget.exe --continue --progress=bar:force -P (Split-Path -Parent $Archive) $ClimateUrl
            if ($LASTEXITCODE -eq 0 -and (Test-Path $UpstreamArchive)) {
                Move-Item -LiteralPath $UpstreamArchive -Destination $Archive -Force
            }
        } else {
            & curl.exe --fail --location --continue-at - --output $Archive --progress-bar $ClimateUrl
        }
        if ($LASTEXITCODE -ne 0) { throw "La descarga falló (exit $LASTEXITCODE). Puedes reanudar ejecutando este script de nuevo." }
    }

    Write-Host "Extrayendo con el script oficial de FungiTastic..."
    & $Python $DownloadScript --climatic --keep_zip --save_path $SavePath
    if ($LASTEXITCODE -ne 0) { throw "El descargador oficial falló (Python exit $LASTEXITCODE). Revisa $LogPath" }
    if (-not (Test-Path $Archive)) { throw "Terminó el comando, pero no aparece el archivo esperado: $Archive" }
    foreach ($csv in $ClimateCsvs) {
        if (-not (Test-Path $csv) -or (Get-Item -LiteralPath $csv).Length -eq 0) {
            throw "Falta un archivo climático extraído o está vacío: $csv"
        }
    }

    $hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $Archive).Hash.ToLowerInvariant()
    $manifest = [ordered]@{
        source = "https://cmp.felk.cvut.cz/datagrid/FungiTastic/shared/download/climaticData.zip"
        downloaded_at_utc = [DateTime]::UtcNow.ToString("o")
        archive = $Archive
        bytes = (Get-Item -LiteralPath $Archive).Length
        sha256 = $hash
    }
    $manifest | ConvertTo-Json | Set-Content -LiteralPath $ManifestPath -Encoding utf8
    Write-Host "Descarga lista. Checkpoint: $ManifestPath"
    Write-Host "SHA-256: $hash"
    Write-Host "Log: $LogPath"
} finally {
    Stop-Transcript | Out-Null
    Pop-Location
}
