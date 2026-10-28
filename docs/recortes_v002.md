# Recortes sobre MobileNetV3-Large v002

Esta campaña trata la segmentación como recortes geométricos de fotografías. No modifica v002 ni asigna nuevas particiones; cada vista mantiene la identidad de su foto y observación. Los porcentajes son áreas previas al procesamiento habitual (Resize y CenterCrop), no porcentajes del lado ni del área final que ve el modelo.

## Estado actual: comparación completada

Las dos continuaciones finalizaron por parada temprana en la época 4, con el mejor checkpoint en la época 1. Las tres épocas siguientes no mejoraron el F1 macro de val; no es necesario ejecutar las seis restantes del máximo de diez.

- Control completado: `runs/crops_v002/20261028T141000472345Z_cropping_mobilenet_v3_large_v002_control_seed42`.
- Central 75 % completado: `runs/crops_v002/20261028T145527817405Z_cropping_mobilenet_v3_large_v002_center75_seed42`.
- Informe consolidado: `reports/crops_v002/completed_seed42/RESULTS.md`.
- Candidato por validación: `reports/crops_v002/completed_seed42/selection.json`.

| Entrenamiento | F1 macro obs., inferencia original | F1 macro obs., inferencia central 75 % |
|---|---:|---:|
| Control | 0,9773 | 0,9738 |
| Con recortes | 0,9661 | 0,9770 |

Se conserva el procesamiento original. La mejora del modelo recortado al comparar ambos con inferencia central 75 % sigue siendo no concluyente (Δ F1 +0,00316; IC 95 % −0,00101 a +0,00768). Esta conclusión se limita a una semilla y esta receta de recorte.

Los mejores pesos y buffers son exactamente los del piloto, aunque los archivos `.pt` reserializados tienen otro hash. Esa igualdad se verificó tensor por tensor, junto con receta, datos, clases y predicciones finales. Se reutilizó la inferencia anterior; las curvas nuevas cubren las cuatro épocas y se recalcularon las comparaciones desde los CSV guardados. La auditoría está en `reuse_audit.json`.

Los parámetros siguieron cambiando sin mejorar val. En la época 4, el cambio L2 del extractor respecto a C fue 2,756 % en control y 2,799 % con recortes; las estadísticas de BatchNorm cambiaron 3,722 % y 5,106 %, respectivamente. Son valores de la última época, distintos de los del mejor checkpoint de la época 1.

La continuación de control queda registrada como candidato de esta comparación; no reemplaza la selección histórica de S5. Test permanece reservado. El estudio posterior de las semillas 42, 43 y 44 obtuvo F1 macro medio 0,9732 y desviación estándar 0,0037; no confirmó una mejora uniforme frente a C (0,9717). No es necesario prolongar estas corridas.

La comprobación adicional con semillas 43 y 44 está completada, con el protocolo y resultados descritos en [control_stability_v002.md](control_stability_v002.md). Mantiene el candidato 42 preseleccionado y estudia únicamente el control con encuadre original.

## Sensibilidad con C congelado: completada

Resultados: `reports/crops_v002/20261024T161742286751Z_frozen_C/RESULTS.md`.

Se evaluaron las mismas 6 290 fotos y 3 452 observaciones de val con un único checkpoint C. El control reprodujo todas las decisiones originales; los hashes de todos los parámetros y buffers fueron idénticos antes y después de la inferencia. No se consultó test.

| Procesamiento | F1 macro por observación | F1 macro por foto | Vistas por foto |
|---|---:|---:|---:|
| Control actual | 0,9717 | 0,9188 | 1 |
| Central 75 % | 0,9662 | 0,9274 | 1 |
| Central 50 % | 0,9660 | 0,9132 | 1 |
| Cinco regiones al 50 % | 0,9661 | 0,9254 | 5 |

Los tres intervalos pareados de diferencia frente a C incluyen cero. Se conserva el control como procesamiento recomendado: no hay evidencia clara de mejora de la métrica principal. Central 75 % merece un piloto de adaptación porque mejora el F1 por foto y usa una sola vista; esto no prueba una mejora por observación tras entrenar.

Morchella tiene solo 11 observaciones de val. El recorte central del 75 % añade un falso positivo para ese género y su F1 pasa de 1 a 0,9565, aunque conserva el recall. Geastrum también pierde precisión. Estas clases explican parte de la caída macro; una mayor exactitud global no basta para afirmar que el método mejora.

Las cinco vistas se promedian por foto y luego por observación. Los intervalos usan 10 000 remuestreos pareados por observación, semilla 42, sin corrección múltiple; no incluyen la variación de semillas de entrenamiento ni la dependencia por sitio-día. Los tiempos guardados son de forward en GPU y excluyen lectura/transformaciones; no son latencia Android.

La galería `gallery_train/gallery_train.png` usa una observación de train por género, seleccionada con semilla 42. Las fotos y sus recortes permanecen locales, excluidos de Git; `gallery_index.csv` identifica las fuentes. Las figuras estadísticas sí pueden versionarse.

## Scripts disponibles

- `scripts/crops_visible.ps1`: evaluación congelada con barras, bloques atómicos por lote y reanudación.
- `scripts/train/analyze_crop_experiments.py`: recalcula tablas, intervalos y figuras desde las probabilidades guardadas, sin inferencia.
- `scripts/train/preview_crop_policy.py`: galería de train, con filas guardadas y reutilizables.
- `scripts/train/train_ma.py`: admite inicialización nueva desde un checkpoint y registro de cambios de pesos.
- `scripts/train/evaluate_crop_pair.py`: comparación de validación 2 × 2, con bloques por lote y bootstrap reanudables.
- `scripts/train/plot_crop_weights.py`: figura de distancias desde los CSV, con porcentajes y eje desde cero; no ejecuta inferencia.
- `scripts/train/consolidate_crop_runs.py`: consolidación de corridas terminadas; permite reutilizar inferencia solo si los mejores estados y la receta coinciden exactamente.

Desde la raíz del repositorio, un estudio congelado nuevo se inicia así:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/crops_visible.ps1
```

Para continuar un estudio interrumpido:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/crops_visible.ps1 -Resume "reports/crops_v002/<estudio>"
```

La reanudación exige los mismos datos, pesos, código, configuración y entorno. Si cambian, debe abrirse un estudio nuevo. El estudio completado conserva una copia de sus fuentes originales para auditoría. Rehacer sus gráficos no requiere el entorno de inferencia original:

```powershell
.\.venv\Scripts\python.exe -u scripts/train/analyze_crop_experiments.py --study reports/crops_v002/20261024T161742286751Z_frozen_C
```

## Ajuste pareado y cambios de pesos

Las configuraciones nuevas son:

- `configs/train/cropping_mobilenet_v3_large_v002_control.yaml`.
- `configs/train/cropping_mobilenet_v3_large_v002_center75.yaml`.

Ambas inicializan exactamente desde `runs/20261024T031650621510Z_s5_mobilenet_v3_large_v002_c/best.pt`, con optimizador, scheduler, historial y numeración nuevos. Usan semilla 42, las mismas fotos y conteos de clase, AdamW, tasa 0,0001, lote 64, warmup de una época, máximo 10 épocas y paciencia 3. Solo cambia el recorte adicional. El orden de lotes se controla con la misma semilla; CUDA no se declara determinista.

Una época por rama constituye el piloto inicial. Se solicita pausa al terminar esa época para conservar `last.pt`; no equivale a completar la campaña de diez épocas ni a un resultado definitivo. Las dos ramas deben compararse también con un mismo procesamiento de inferencia para distinguir adaptación y encuadre.

```powershell
powershell -ExecutionPolicy Bypass -File scripts/train_visible.ps1 -SkipInstall -Config configs/train/cropping_mobilenet_v3_large_v002_control.yaml -PauseAfterEpoch
powershell -ExecutionPolicy Bypass -File scripts/train_visible.ps1 -SkipInstall -Config configs/train/cropping_mobilenet_v3_large_v002_center75.yaml -PauseAfterEpoch
```

Para continuar cualquiera de ellas, se pasa su configuración original y su `last.pt` con `-Resume`, sin `-PauseAfterEpoch`. La corrida de continuación se guarda en una carpeta nueva y hereda todo el historial. Se imprimirá esa ruta.

La opción `-InitCheckpoint` del lanzador (o `--init-checkpoint` de Python) permite crear otra corrida desde pesos comunes. No se combina con `-Resume`: reanudar restaura también optimizador, scheduler, RNG, paciencia e historial. La referencia de pesos y sus mediciones se incluyen en `last.pt`, por lo que no dependen de mantener abierta la sesión anterior.

Cada corrida guarda:

- `initial.pt` y `initialization.json`: referencia común y hashes de origen.
- `last.pt`, `best.pt`, `history.csv` y `progress.json`.
- `parameter_drift.csv`: normas por tensor respecto a C, época, lotes y actualizaciones reales del optimizador.
- `parameter_drift_summary.csv`: grupos backbone, cabeza clasificadora, estadísticas de BatchNorm y contadores.

La distancia relativa es `||W_t − W_0||₂ / (||W_0||₂ + 1e-12)`. Las estadísticas de BatchNorm son buffers, no pesos entrenables. Una distancia mayor no demuestra una mejora ni permite atribuir por sí sola un significado visual a cada capa. El desbalance sigue ponderado con las fotos originales de train; los recortes no alteran los pesos de pérdida por género.

La verificación de integridad permite `data.integrity_workers: 8` en estas configuraciones. Lee y calcula nuevamente cada SHA-256; no omite imágenes ni reutiliza hashes basados solo en fecha o tamaño. Las campañas anteriores conservan su configuración.

## Evaluación y ampliación

El piloto de una época por rama y su evaluación 2 × 2 están completados. Ambas ramas realizaron 471 lotes y 471 actualizaciones de AdamW, y quedaron pausadas con `last.pt`:

- Control: `runs/crops_v002/20261024T164558048644Z_cropping_mobilenet_v3_large_v002_control_seed42`.
- Central 75 %: `runs/crops_v002/20261024T165707710689Z_cropping_mobilenet_v3_large_v002_center75_seed42`.

Resultados completos: `reports/crops_v002/pilot_seed42/RESULTS.md`.

| Modelo después del ajuste | Inferencia original | Inferencia central 75 % |
|---|---:|---:|
| Continuación de control | 0,9773 | 0,9738 |
| Ajuste con recorte central 75 % | 0,9661 | 0,9770 |

Valores: F1 macro por observación en las mismas 3 452 observaciones de val. Con inferencia original, la diferencia recortado − control fue −0,01119 (IC 95 % pareado: −0,02699 a −0,00085). Con inferencia central 75 %, fue +0,00316 (−0,00101 a +0,00768): no hay evidencia clara de una ventaja por entrenar con recortes en esa comparación. Son intervalos exploratorios sin corrección múltiple y un piloto de una sola semilla.

El modelo adaptado funciona mejor cuando recibe el recorte con el que fue ajustado; esto no justifica sustituir el procesamiento original. La continuación de control también cambia los pesos y alcanza una mayor puntuación que C en val. Esa diferencia preliminar no sustituye una campaña completa ni una evaluación final.

| Estado respecto a C | Control: cambio L2 relativo | Central 75 %: cambio L2 relativo |
|---|---:|---:|
| Parámetros del extractor | 1,025 % | 0,997 % |
| Parámetros de la cabeza | 1,272 % | 1,232 % |
| Estadísticas de BatchNorm (buffers) | 1,522 % | 3,398 % |

Estas son distancias de cada rama respecto a la inicialización común, no distancias directas entre las dos ramas. Los buffers de BatchNorm cambian más con recortes; esto describe cambios de estadísticas internas, sin demostrar una mejora de clasificación. Se conservaron también mediciones por tensor y por fila de la capa de salida para cada género.

Los siguientes comandos se usaron para continuar desde el piloto. Se conservan como historial: las continuaciones ya finalizaron y no es necesario volver a ejecutarlos.

```powershell
powershell -ExecutionPolicy Bypass -File scripts/train_visible.ps1 -SkipInstall -Config configs/train/cropping_mobilenet_v3_large_v002_control.yaml -Resume "runs/crops_v002/20261024T164558048644Z_cropping_mobilenet_v3_large_v002_control_seed42/last.pt"

powershell -ExecutionPolicy Bypass -File scripts/train_visible.ps1 -SkipInstall -Config configs/train/cropping_mobilenet_v3_large_v002_center75.yaml -Resume "runs/crops_v002/20261024T165707710689Z_cropping_mobilenet_v3_large_v002_center75_seed42/last.pt"
```

La ampliación conservó el límite de diez épocas y la paciencia de tres, contando la época del piloto, y finalizó en la cuarta. Las fuentes exactas usadas por ambos entrenamientos del piloto se archivaron en `reports/crops_v002/pilot_seed42/training_source/`; los mensajes posteriores de preparación del primer lote no alteran la receta ni el formato de reanudación.

El evaluador pareado recibe las carpetas de las dos ramas y un destino nuevo:

```powershell
.\.venv\Scripts\python.exe -u scripts/train/evaluate_crop_pair.py --control "runs/crops_v002/20261024T164558048644Z_cropping_mobilenet_v3_large_v002_control_seed42" --cropped "runs/crops_v002/20261024T165707710689Z_cropping_mobilenet_v3_large_v002_center75_seed42" --output reports/crops_v002/pilot_seed42
```

Produce métricas de ambos modelos con inferencia original y central 75 %, comparación pareada a encuadre común, curvas de distancia de pesos y registros a épocas/lotes comunes. Repetir el mismo comando retoma sus bloques terminados. Si AMP omite pasos de AdamW, el análisis indica si coinciden las actualizaciones reales.

Después de continuar un entrenamiento, debe indicarse su nueva carpeta de corrida y un destino de análisis nuevo: el evaluador rechaza reutilizar bloques de otros pesos o código. Tanto el lanzador como el pipeline imprimen la etapa desde el inicio y un aviso periódico mientras se prepara el primer lote en Windows.

La figura final del piloto usa barras para una sola época, con eje desde cero y los buffers separados de los parámetros. Puede regenerarse sin leer imágenes ni modelos:

```powershell
.\.venv\Scripts\python.exe -u scripts/train/plot_crop_weights.py --study reports/crops_v002/pilot_seed42
```

La comparación de la semilla 42 y el estudio adicional de las semillas 43 y 44 están cerrados. Se mantiene el encuadre original y se conserva el control 42 como candidato preseleccionado de la nueva campaña. Los resultados de estabilidad están en `reports/crops_v002/control_stability/results/RESULTS.md`. Los ajustes exploratorios no reemplazan automáticamente `reports/s5/selected_model.json`. El test permanece reservado para una selección final explícita.

Para reproducir la consolidación sin inferencia:

```powershell
.\.venv\Scripts\python.exe -u scripts/train/consolidate_crop_runs.py --control runs/crops_v002/20261028T141000472345Z_cropping_mobilenet_v3_large_v002_control_seed42 --cropped runs/crops_v002/20261028T145527817405Z_cropping_mobilenet_v3_large_v002_center75_seed42 --reuse-evaluation reports/crops_v002/pilot_seed42 --output reports/crops_v002/completed_seed42
```

El consolidador rechaza la reutilización si cambia cualquier peso, buffer, clase, receta o código de inferencia. En ese caso corresponde ejecutar `evaluate_crop_pair.py` sobre los checkpoints nuevos y un directorio de salida nuevo.
