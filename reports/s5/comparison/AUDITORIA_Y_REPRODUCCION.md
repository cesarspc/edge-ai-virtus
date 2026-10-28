# Auditoría y reproducción del informe de S5 y S6

## Fase 0: hallazgos antes de la reescritura

Se inspeccionaron las corridas, sus reanudaciones y los informes históricos. El árbol y los archivos por carpeta se guardan en `artifact_tree.json` y `run_inventory.csv` (reports/s5/comparison/).

| Ensayo | Tasa inicial | Lote train / val | Ponderación | Entorno efectivo | Fuente de los ajustes |
|---|---:|---:|---|---|---|
| A | 0,0003 | 64 / 128 | none | CPU FP32 | (runs/20261012T161041257797Z_s5_mobilenet_v3_large_v002_a/run.json) |
| B | 0,0001 | 64 / 64 | none | GPU AMP | (runs/20261024T022849507889Z_s5_mobilenet_v3_large_v002_b/run.json) |
| C | 0,0003 | 64 / 64 | inverse_sqrt | GPU AMP | (runs/20261024T031650621510Z_s5_mobilenet_v3_large_v002_c/run.json) |
| D | 0,0001 | 64 / 64 | inverse_sqrt | GPU AMP | (runs/20261024T132723373718Z_s5_mobilenet_v3_large_v002_d/run.json) |

El lote de A se reconstruyó de la versión histórica del código cuyo SHA-256 con CRLF coincide con run.json; se conserva la comprobación (reports/s5/comparison/historical_code_audit.json).

El resto de la receta se conserva entre ensayos: arquitectura, entrada, aumentos, optimizador, scheduler, suavizado, semilla y ausencia de capas congeladas; la tabla LaTeX y el CSV de configuración contienen cada valor y su fuente (reports/s5/comparison/configuration.csv).

Las corridas válidas comparten huella v002, orden de géneros y todos los identificadores de validación. Se verificaron los bytes de metadatos contra el manifiesto y se reprodujeron las matrices de confusión a partir de los CSV (reports/s5/comparison/analysis_manifest.json).

Las predicciones por observación no existían como archivos separados: se reconstruyeron promediando probabilidades por observación, antes de argmax; se entregan cinco CSV derivados, uno por modelo (scripts/train/metrics.py; reports/s5/comparison/predictions_observation_*_val.csv).

Ningún MobileNet tiene test guardado; ResNet-18 tiene una evaluación histórica y su exposición se declara. No se ejecutó inferencia nueva sobre test (reports/s5/comparison/run_inventory.csv; reports/s5/comparison/analysis_manifest.json).

### Corridas usadas y resultados de validación

| Modelo | F1 observación [IC 95 %] | F1 foto | Acierto observación, % | Mejor / total | Horas de épocas + val | Fuente |
|---|---|---:|---:|---:|---:|---|
| ResNet-18 | 0,955 [0,945; 0,963] | 0,900 | 95,9 | 11 / 16 | 14,80 | (runs/20261004T133123181686Z_ma_resnet18_v002/metrics_val.json; runs/20261004T133123181686Z_ma_resnet18_v002/history.csv) |
| MobileNet A | 0,971 [0,956; 0,980] | 0,925 | 98,0 | 19 / 24 | 14,04 | (runs/20261012T161041257797Z_s5_mobilenet_v3_large_v002_a/metrics_val.json; runs/20261012T161041257797Z_s5_mobilenet_v3_large_v002_a/history.csv) |
| MobileNet B | 0,970 [0,956; 0,979] | 0,915 | 97,7 | 6 / 11 | 0,76 | (runs/20261024T022849507889Z_s5_mobilenet_v3_large_v002_b/metrics_val.json; runs/20261024T022849507889Z_s5_mobilenet_v3_large_v002_b/history.csv) |
| MobileNet C | 0,972 [0,964; 0,979] | 0,919 | 97,7 | 16 / 21 | 1,44 | (runs/20261024T031650621510Z_s5_mobilenet_v3_large_v002_c/metrics_val.json; runs/20261024T031650621510Z_s5_mobilenet_v3_large_v002_c/history.csv) |
| MobileNet D | 0,962 [0,947; 0,972] | 0,913 | 97,4 | 7 / 12 | 0,84 | (runs/20261024T132723373718Z_s5_mobilenet_v3_large_v002_d/metrics_val.json; runs/20261024T132723373718Z_s5_mobilenet_v3_large_v002_d/history.csv) |

Las horas no incluyen preflight, espera entre sesiones ni evaluación final, y no suman otra vez las historias de origen al reanudar. D usa su carpeta final de continuación (reports/s5/comparison/validation_results.csv; reports/s5/comparison/run_inventory.csv).

### Corridas excluidas

- runs/20261023T005149730394Z_s5_mobilenet_v3_large_v002_b: runs\20261023T005149730394Z_s5_mobilenet_v3_large_v002_b\predictions_val.csv: 6272/6290 rows contain non-finite probabilities (reports/s5/comparison/run_inventory.csv).
- runs/20261023T020544245443Z_s5_mobilenet_v3_large_v002_c: runs\20261023T020544245443Z_s5_mobilenet_v3_large_v002_c\predictions_val.csv: 6272/6290 rows contain non-finite probabilities (reports/s5/comparison/run_inventory.csv).
- runs/20261024T015337199324Z_s5_mobilenet_v3_large_v002_d: runs\20261024T015337199324Z_s5_mobilenet_v3_large_v002_d\predictions_val.csv: 6272/6290 rows contain non-finite probabilities (reports/s5/comparison/run_inventory.csv).

Se excluyen por probabilidades no finitas, aunque exista completed.json. No se interpretan sus métricas como rendimiento del modelo. Las carpetas se conservaron (reports/s5/development_results.json).

## Comparación y selección

Bootstrap percentil ordinario por observación: 10 000 remuestreos, semilla 42, etiquetas fijas. Los modelos comparten los mismos índices en cada remuestreo; no hay ajuste por comparaciones múltiples (reports/s5/comparison/analysis_manifest.json).

| Comparación | Diferencia F1 | IC 95 % pareado | Incluye cero |
|---|---:|---|---|
| A menos R | 0,0160 | [0,0005; 0,0281] | No |
| B menos A | -0,0008 | [-0,0173; 0,0158] | Sí |
| B menos R | 0,0152 | [0,0011; 0,0265] | No |
| C menos A | 0,0011 | [-0,0091; 0,0157] | Sí |
| C menos R | 0,0171 | [0,0077; 0,0270] | No |
| D menos A | -0,0089 | [-0,0254; 0,0078] | Sí |
| D menos R | 0,0071 | [-0,0081; 0,0192] | Sí |
| C menos B | 0,0019 | [-0,0084; 0,0161] | Sí |
| C menos D | 0,0100 | [-0,0011; 0,0248] | Sí |
| D menos B | -0,0081 | [-0,0155; -0,0012] | No |

Fuente de todas las diferencias: (reports/s5/comparison/bootstrap_paired.csv).

Se selecciona C por la mayor estimación de F1 macro por observación. Su diferencia con A no se distingue del ruido; no se afirma superioridad concluyente. Los MobileNet tienen idéntico coste de arquitectura y no existe benchmark Android que los desempatase. La selección operativa conserva el criterio principal y queda congelada con hashes (reports/s5/selected_model.json; reports/s5/comparison/profile_ratios.json).

## Cambios realizados

- Reescritura del LaTeX con A/B/C/D, fuentes locales y lista de pendientes (reports/week_05_06_results.tex).
- Curvas sin etiquetas flotantes, con mejores vueltas en la leyenda; diferencias pareadas, métricas por género y tablas de soporte (reports/s5/comparison/).
- Taxonomía del ganador y ResNet-18 con F1 y exactitud e IC; confusiones dirigidas y todas las especies difíciles según el criterio (reports/s6/comparison/).
- Se distingue la exposición histórica a test y se entrega el comando final sin ejecutarlo (scripts/train/evaluate_s5_final.py).
- El lector y selector rechazan probabilidades no finitas; el evaluador final respeta el lote configurado. No se altera la data ni las particiones (scripts/train/pipeline/predictions.py; scripts/train/summarize_s5.py; scripts/train/evaluate_s5_final.py).

## Cifras que cambiaron y por qué

- ResNet-18 val pasa de 0,954 a 0,955 en las tablas a tres decimales: el valor exacto sigue siendo 0,9545427706 (runs/20261004T133123181686Z_ma_resnet18_v002/metrics_val.json).
- Los IC de ResNet val y A val se recalcularon con el método común; se reportan los nuevos extremos en la tabla de resultados. La amplitud mayor de A se relaciona con su error en Morchella, un género de soporte muy pequeño (reports/s5/comparison/bootstrap_genus_uncertainty.csv).
- La razón de velocidad histórica es 1,7928058711, redondeada a 1,793, y la reducción de tamaño 61,9228247881 %, redondeada a 61,923 %. No son medidas Android (reports/s5/comparison/profile_ratios.json).
- Se reemplaza A como candidato por C siguiendo la estimación principal, con empate no resuelto estadísticamente frente a A; se añade B/C/D y se retiran sus corridas corruptas de la selección (reports/s5/development_results.json; reports/s5/comparison/bootstrap_paired.csv).
- La lista de especies usa el criterio en al menos una evaluación disponible, y añade los soportes de val y test; ya no conserva únicamente las tres especies del PDF original (reports/s6/comparison/species_difficult.csv).

## Supuestos y límites

- Se conserva la taxonomía de FungiTastic, incluida Polyporaceae para Ganoderma; cambiar esa clasificación cambiaría los aciertos familiares (data/taxonomy/v002/genus_taxonomy.csv).
- La observación es la unidad de remuestreo. Los IC son condicionales a los modelos y no cubren entrenamiento ni dependencia de sitio-día; los CSV redondeados reproducen las matrices guardadas (reports/s5/comparison/analysis_manifest.json).
- A usa CPU FP32 y B/C/D CUDA AMP. Una sola semilla por receta impide atribuir causalmente toda diferencia A-C a la ponderación (reports/s5/comparison/configuration.csv).
- La explicación por aspecto visual se presenta como hipótesis; los conteos de fotos no demuestran esa causa (reports/s6/comparison/species_difficult.csv).
- El test de ResNet-18 ya fue expuesto: el futuro test de C será una nueva evaluación de otro modelo sobre una partición conocida históricamente, no una partición ciega durante todo el proyecto (runs/20261004T133123181686Z_ma_resnet18_v002/completed.json).

## Alcance y trabajo futuro

- Test de C: métricas, predicciones, IC, taxonomía, confusiones y especies (no existe final_test/completed.json en la corrida seleccionada).
- Teléfono Android concreto, runtime/formato final, latencia, memoria y tamaño del despliegue (reports/s5/architecture_profile.json).
- Réplicas independientes con otras semillas (reports/s5/comparison/validation_results.csv).
- Verificación de clonación limpia en Windows con finales de línea reproducibles; el checkout actual sí coincide con el manifiesto (reports/s5/comparison/git_windows_audit.json).
- Modelo exacto de CPU de las corridas A y ResNet (sus run.json registran CPU, sin identificar el procesador).
- Transcripciones completas de terminal de B/C/D y tiempo de pared con preparación/evaluación final; history.csv solo contiene el tiempo de épocas más val (reports/s5/comparison/run_inventory.csv).
- Descriptores o revisión morfológica para contrastar el aspecto diferente de las especies difíciles (reports/s6/comparison/species_difficult.csv).

## Reproducción

Desde D:\Dev\Virtus\edge-ai-virtus, en PowerShell:

```powershell
.venv\Scripts\python.exe -u scripts\reports\analyze_s5_s6.py --resamples 10000 --seed 42
.venv\Scripts\python.exe scripts\reports\build_s5_s6_tables.py
```

El primer comando reanuda automáticamente el checkpoint del bootstrap. Si cambian las predicciones o el método, exige --restart. Las tablas se escriben por separado y son regenerables (scripts/reports/analyze_s5_s6.py; scripts/reports/build_s5_s6_tables.py).

Compilación del PDF, con Tectonic instalado (se usó el ejecutable auxiliar local tmp/tectonic/tectonic.exe):

```powershell
tectonic --keep-logs --outdir reports reports/week_05_06_results.tex
```

La selección ya está congelada. El siguiente comando se entrega para ejecutar deliberadamente el test pendiente una vez; NO se ejecutó durante esta revisión:

```powershell
.venv\Scripts\python.exe -u scripts\train\evaluate_s5_final.py
```

El marcador de campaña bloquea repeticiones automáticas. No se debe borrarlo para repetir una evaluación (scripts/train/evaluate_s5_final.py).

Comprobaciones numéricas y de reanudación: registro en (reports/s5/comparison/validation_checks.txt). Los CSV y JSON de comparison son la evidencia de esta revisión; los exports anteriores de reports/s6 describen el estado previo y no sustituyen las nuevas tablas.
