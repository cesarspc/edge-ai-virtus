# Baseline v002: particiones oficiales y primer modelo

Requiere Python 3.12 y `curl` en Windows. Ejecutar desde la raíz:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_baseline.ps1
```

Opciones: `-DatasetOnly` genera el dataset sin entrenar; `-SmokeOnly` termina tras la prueba rápida;
`-SkipInstall` reutiliza el entorno preparado. Cada etapa se detiene si falla. Repetir la orden reanuda la descarga.
En equipos con GPU NVIDIA, instalar la distribución CUDA de PyTorch apropiada en `.venv` antes de ejecutar.

Se adquiere solo FungiTastic. `v002` usa 10 géneros y los CSV `Train`, `ClosedSet-Val` y `ClosedSet-Test`;
no incluye OpenSet, DNA, Mini/FewShot, DF20, GBIF ni otras fuentes. No hay `test_external`.
Las proporciones y conteos son los oficiales después del filtrado de géneros, sin muestreo para balancear val/test.
El desbalance se describe por clase; la selección utiliza macro-F1 por observación.

El flujo tiene dos etapas de construcción: `--prepare-images` escribe candidatos en `data/interim/v002/`
para descargarlos; la construcción final valida todos los archivos descargados y publica `data/processed/v002/`
y `data/versions/v002/`. Los candidatos no publican un manifiesto y nunca reemplazan tablas finales.
Se conserva la inmutabilidad de `v001` y de cualquier versión ya publicada.

La curaduría compara SHA-256 entre particiones: test tiene prioridad sobre val, y val sobre train.
Solo se eliminan las copias de particiones inferiores; las copias dentro de una misma partición se conservan.
Si una observación pierde todas sus imágenes, se retira de la tabla de observaciones y se recalculan los conteos.
Las similitudes perceptuales (Hamming <= 4) se reportan sin excluirlas automáticamente.
Las imágenes decodificables cuyo lado menor está por debajo de 128 px se conservan, se registran en
`small_images.csv` y se redimensionan por los transforms; esto preserva también las fotografías panorámicas oficiales de test.

Los artefactos versionados incluyen `image_exclusions.csv`, `near_duplicates.csv`, `site_day_overlaps.csv`
y `image_integrity.csv`. El último registra los hashes de las imágenes efectivamente usadas y forma parte de
la huella del dataset. El entrenamiento verifica los bytes locales contra esos hashes.
Los cruces sitio-día se cuantifican sin cambiar las particiones oficiales; deben citarse como limitación si existen.

```powershell
.\.venv\Scripts\python.exe scripts/data/build_dataset.py --config configs/dataset_v002.yaml --check
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

La prueba `--smoke` entrena tres lotes y evalúa solo val. `--eval-only` también evalúa solo val.
El entrenamiento completo tiene hasta 30 épocas y parada temprana tras cinco épocas sin mejorar val.
Solo evalúa test al terminar con el mejor checkpoint. Resultados: `runs/<run_id>/` (ignorado por Git),
con configuración, huella del dataset, historial, `best.pt`, métricas, predicciones y `completed.json`.
Un directorio sin `completed.json` corresponde a una ejecución incompleta.

## Primera ejecución local (2026-10-01)

### Ver el progreso o entrenar en una terminal visible

Desde la raíz del repositorio, el visor muestra la corrida existente con una barra Python,
pérdida y ETA de train para la época actual. No inicia otra corrida; Ctrl+C solo cierra el visor:

```powershell
.\.venv\Scripts\python.exe scripts/train/watch_training.py
```

Para seguir el log sin el visor:

```powershell
Get-Content .\runs\baseline_v002.stdout.log -Tail 20 -Wait
```

Para una nueva corrida en primer plano, una vez terminada la actual:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/train_visible.ps1
```

El script instala las dependencias de entrenamiento y usa los datos locales ya preparados;
no vuelve a descargar imágenes. Añadir `-Smoke` ejecuta solo la prueba rápida.
Train, validación y evaluación final muestran barras `tqdm` con tiempo y velocidad.
Ctrl+C en el entrenamiento lo interrumpe; para continuar se usa `--resume` con el último `last.pt`.
El lanzador rechaza otra corrida mientras el proceso registrado siga activo.
Las barras internas nuevas se aplican a las siguientes ejecuciones; el visor puede usarse ahora.

### Pausar y reanudar

Las nuevas corridas guardan `last.pt` de forma atómica al completar cada época y su validación.
Contiene pesos actuales, optimizador, scheduler, GradScaler, estados aleatorios (Python, NumPy,
Torch, CUDA y generador del DataLoader), historial, mejor checkpoint y contador de parada temprana.
Se recrean los workers por época para restaurar sus semillas al reanudar. La restauración exige
la misma configuración, clases, huella del dataset y tipo de dispositivo; no garantiza identidad
numérica entre hardware o versiones de bibliotecas distintos.

Para entrenar una época y pausar antes de la evaluación final:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/train_visible.ps1 -PauseAfterEpoch
```

Para pedir una pausa mientras una corrida nueva ya está ejecutándose, desde otra terminal
crea este archivo en **su** directorio (reemplaza `<run_id>`):

```powershell
New-Item -ItemType File -Path .\runs\<run_id>\pause.request
```

La pausa sucede al terminar la época y validación actuales; el visor mostrará `paused`.
También puedes usar Ctrl+C: se conserva la última época guardada y se pierde el avance parcial
de la época interrumpida. Antes de completar la primera época todavía no existe `last.pt`.

Para continuar (reemplaza `<run_id>` por la corrida pausada):

```powershell
powershell -ExecutionPolicy Bypass -File scripts/train_visible.ps1 -Resume "runs/<run_id>/last.pt"
```

O directamente con Python:

```powershell
.\.venv\Scripts\python.exe scripts/train/train_ma.py --config configs/train/ma_resnet18_v002.yaml --resume "runs/<run_id>/last.pt"
```

La continuación crea otro directorio con el historial y mejor checkpoint heredados y registra
`resumed_from` en `run.json`. No se reanuda una corrida completada o que ya haya empezado la
evaluación de test; una interrupción durante test requiere revisión y no dispara otra evaluación automática.
`best.pt` antiguo no contiene el estado del optimizador y se rechaza como checkpoint de reanudación.
La primera corrida local arrancó antes de implementar esta función. Las corridas posteriores
utilizaron `last.pt`; la corrida final consolidada registra la cadena de reanudaciones.

El entorno Python 3.12.10 y las dependencias quedaron instalados en `.venv`.
La publicación y su reconstrucción con `--check` producen la misma huella:
`144323e165ecfbe86d29a3484e56fd91b8e8cb1dce1eef5f1d1355fea7bedb90`.

| Partición | Observaciones | Imágenes |
|---|---:|---:|
| train | 17.799 | 30.172 |
| val | 3.452 | 6.290 |
| test | 4.039 | 7.159 |

No hay duplicados exactos ni cruces sitio-día entre particiones. Se reporta un par con
similitud perceptual train/test y se conservan 33 fotografías pequeñas. Morchella tiene
solo 11 observaciones de val y 20 de test; interpretar sus métricas con ese soporte.
Los cinco tests del código pasaron y `check_env.py` indicó LISTO, sin imágenes faltantes.
La prueba real quedó en `runs/20261001T225420063305Z_ma_resnet18_v002_smoke/`, con
`completed.json`, checkpoint, historial y métricas de val; cero evaluaciones de test.

## Cierre y consolidación (2026-10-04)

La corrida `runs/20261004T133123181686Z_ma_resnet18_v002/` terminó correctamente en CPU,
tras 16 épocas y parada temprana. El checkpoint seleccionado es de la época 11.
`completed.json` confirma una sola evaluación final de test.

| Unidad | Macro-F1 val | Macro-F1 test | Exactitud test |
|---|---:|---:|---:|
| Imagen | 0,8997 | 0,8875 | 91,09 % |
| Observación | 0,9545 | 0,9463 | 95,79 % |

El checkpoint, la configuración efectiva, el manifiesto del dataset, las predicciones,
las métricas y el historial están conservados en
`artifacts/baselines/resnet18_v002/20261004T133123181686Z_ma_resnet18_v002/`.
`archive_manifest.json` registra SHA-256 de cada copia y comprueba igualdad con el origen.
La corrida original permanece sin cambios.

Las tablas y listas de errores están en `reports/baseline_v002/`; las seis gráficas de
curvas, matrices de confusión y F1 por género están en `reports/figures/baseline_v002_*.png`.
La galería de fotografías de errores queda solo en local, fuera de Git.
Hay 499 errores por imagen y 140 por observación en val. Morchella tiene F1 de observación
1,0000 en val con 11 casos, por lo que debe interpretarse con su soporte.

Para reproducir la consolidación, sin entrenamiento ni inferencia:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-report.txt
.\.venv\Scripts\python.exe scripts/train/consolidate_baseline.py
.\.venv\Scripts\python.exe scripts/reports/update_baseline_report.py
```

El último comando actualiza en su mismo archivo la sección de resultados del informe LaTeX,
con datos numéricos y curvas autocontenidas. La inspección detallada de errores utiliza val;
test se presenta como resultado final descriptivo y no se utiliza para ajustar hiperparámetros.
