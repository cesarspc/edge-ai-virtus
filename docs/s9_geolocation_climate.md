# Experimentos B y C: geolocalización y clima

## Diseño acordado

- **B — solo geolocalización:** clasificador tabular que recibe latitud y longitud. No recibe imágenes, elevación, hábitat, sustrato ni fecha. Se usa una fila por observación para que las observaciones con varias fotografías no pesen más.
- **C — imagen + geolocalización + clima:** MobileNetV3-Large visual con una rama tabular para las coordenadas y las variables climáticas asociadas a la observación.
- Ambos usan FungiTastic v002, las mismas clases oficiales y train/val. `val` selecciona el checkpoint. `test` no se toca durante el desarrollo.

Las coordenadas se codifican con seno/coseno de latitud y longitud. La normalización se ajusta únicamente con `train`. Coordenadas ausentes o inválidas se imputan con la media de train; los indicadores de ausencia/calidad no se entregan al modelo, para que B use exclusivamente la localización.

## Obtener solo el paquete climático

El helper `scripts/data/download_fungitastic_climate.ps1` clona el descargador oficial en `tmp` y ejecuta `download.py --climatic --keep_zip`. El índice vigente publica el ZIP como `climaticData.zip`, aunque el descargador oficial busca el nombre local `climatic.zip`; el helper guarda el archivo con el nombre local esperado. Muestra el progreso de curl/wget, conserva el ZIP, escribe un log y guarda un manifiesto con SHA-256. Si se interrumpe, volver a ejecutar el mismo comando continúa la descarga.

```powershell
Set-Location D:\Dev\Virtus\edge-ai-virtus
powershell -ExecutionPolicy Bypass -File scripts\data\download_fungitastic_climate.ps1 -Python .\.venv\Scripts\python.exe
```

El paquete y su extracción se guardan bajo `data/raw/fungitastic/FungiTastic/`. El ZIP se conserva como artefacto de origen; no se vuelven a bajar imágenes ni metadatos. La descarga validada tiene 545,926,654 bytes y SHA-256 `39391ba78f378b88b36356f935811cac32bd4726fe6c0ff2321f7f6c228008d8`.

## Verificación y alineación

`scripts/data/inspect_fungitastic_climate.py` registra el esquema de los dos CSV y `scripts/data/verify_climate_coverage.py` comprueba hash, IDs repetidos, cobertura y valores numéricos. El resultado para v002 quedó en `data/raw/fungitastic/FungiTastic/climatic_coverage_v002.json`: 19 variables BIOCLIM promedio y 912 columnas de serie mensual entre 2000 y 2018, sin IDs repetidos. La cobertura de los archivos es 99,8893 % y 99,8972 % respectivamente.

La fuente no trae datos para 28 observaciones del promedio BIOCLIM (17 train, 10 val, 1 test) ni para 26 de la serie mensual (15 train, 10 val, 1 test). Los faltantes quedan enumerados en el informe; no se eliminan ni se sustituyen por datos externos. La tabla `data/interim/s9_climate_v002.csv` mantiene las 25.290 observaciones y marca `climate_available`; su manifiesto guarda hashes y conteos. Para entrenar C, imputación y escalado deben ajustarse usando solo train y se añade la máscara de disponibilidad.

El experimento C usa las 19 variables promedio `Bio1`–`Bio19` (normal climática), no las 912 columnas mensuales. La serie mensual queda preservada para un experimento posterior que defina una ventana temporal con respecto a `event_date`.

```powershell
.\.venv\Scripts\python.exe scripts\data\inspect_fungitastic_climate.py
.\.venv\Scripts\python.exe scripts\data\verify_climate_coverage.py
.\.venv\Scripts\python.exe scripts\data\prepare_climate_v002.py
```

## Ejecutar B

```powershell
Set-Location D:\Dev\Virtus\edge-ai-virtus
.\.venv\Scripts\python.exe scripts\train\train_s8.py --config configs\train\s9_b_geolocation_v002.yaml --mode geolocation --device cuda
```

Se puede pausar con `Ctrl+C` después del último checkpoint de época y reanudar con `--resume "runs\<carpeta_de_B>\last.pt"`. El entrenamiento deja barra de lotes, `last.pt`, `best.pt`, historial y métricas de validación.

## Ejecutar C

```powershell
Set-Location D:\Dev\Virtus\edge-ai-virtus
.\.venv\Scripts\python.exe scripts\train\train_s8.py --config configs\train\s9_c_image_geo_climate_v002.yaml --mode geo_climate --device cuda
```

C usa MobileNetV3-Large inicializada con el checkpoint C de S5 y suma una rama tabular con coordenadas y las 19 variables BIOCLIM. La rama tabular empieza en cero. Las estadísticas de imputación y escala climática se ajustan únicamente con los registros climáticos disponibles en train; las 17 ausencias de train se imputan con esas estadísticas y se marcan con una variable de disponibilidad. Val selecciona el checkpoint y permite pruebas de sensibilidad con metadatos ausentes/permutados. No se carga ni evalúa test.

B y C comparten semilla, partición, clases, optimizador, lote, épocas máximas, parada temprana y selección por F1 macro de observación en val. B es deliberadamente un baseline de coordenadas solamente; por eso B frente a C mide el cambio completo al añadir imágenes y clima, no el efecto aislado del clima. Para aislar el aporte climático se deberá comparar C contra imagen+geolocalización con la misma configuración.

## C: paso previo al entrenamiento

La estructura y las claves de `climatic.zip` deben inspeccionarse antes de implementar la unión. Después se documentará la transformación que vincule cada serie/raster a una observación usando coordenadas y fecha, la ventana temporal y las unidades. La normalización de variables climáticas también se ajustará solo con train. No se debe crear una fila climática duplicada o emparejar por cercanía sin una regla trazable.

La comparación B/C debe compartir seed, particiones, clases, regla de selección y política de parada. El resultado principal debe calcularse a nivel de observación (F1 macro), junto con el soporte por clase y la cobertura de coordenadas/datos climáticos. Los análisis de entradas ausentes o permutadas se hacen sobre `val`; el test queda reservado a una comparación final previamente definida.
