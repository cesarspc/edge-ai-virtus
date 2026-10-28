# Estabilidad del ajuste de control

Estudio completado: se repitió únicamente el control con encuadre original, usando semillas 43 y 44 además de la 42 ya completada. Las dos repeticiones usaron CUDA de forma secuencial, con máximo diez épocas, paciencia tres, lote 64, tasa 0,0001 y pesos de pérdida `inverse_sqrt` calculados sobre las fotos originales de train.

Las tres corridas parten de los mismos pesos C. Se reinician optimizador y scheduler en cada repetición. Este estudio mide variación del ajuste posterior condicionado al checkpoint C ya seleccionado; no mide la variación completa de entrenar desde ImageNet ni cambia la versión o las particiones del dataset.

## Resultado completado

| Semilla | Épocas ejecutadas | Mejor época | F1 macro por observación |
|---|---:|---:|---:|
| 42 | 4 | 1 | 0,9773 |
| 43 | 5 | 2 | 0,9700 |
| 44 | 9 | 6 | 0,9723 |

Las tres finalizaron por parada temprana. La media de F1 macro fue **0,9732**, con desviación estándar muestral **0,0037** y rango **0,9700–0,9773**. La referencia C antes del ajuste obtuvo **0,9717**.

La mejora observada con la semilla 42 no se reprodujo de forma uniforme: la 43 quedó por debajo del punto de C y todos los intervalos pareados frente a C incluyen cero. Este estudio no permite afirmar una mejora concluyente del ajuste sobre C. La desviación entre semillas y los intervalos por observación describen incertidumbres diferentes.

Morchella (11 observaciones), Hericium (66) y Geastrum (79) mostraron la mayor variación de F1. Hay 40 observaciones mal clasificadas por las tres semillas y 67 con predicciones diferentes entre semillas. Estos casos quedan identificados para revisión; no se eliminaron ni modificaron imágenes o etiquetas.

La semilla 42 sigue siendo el candidato preseleccionado. La selección histórica de S5 no cambió y no se consultó test. El informe, las curvas, errores y hashes están en [results/RESULTS.md](../reports/crops_v002/control_stability/results/RESULTS.md).

Corridas nuevas completadas:

- `runs/crops_v002/20261028T161559294606Z_cropping_mobilenet_v3_large_v002_control_seed43`.
- `runs/crops_v002/20261028T163842294290Z_cropping_mobilenet_v3_large_v002_control_seed44`.

## Protocolo fijado antes de las repeticiones

Archivo: `configs/experiments/control_stability_v002.yaml`.

- Referencia preseleccionada: control, semilla 42, `runs/crops_v002/20261028T141000472345Z_cropping_mobilenet_v3_large_v002_control_seed42/best.pt`.
- Nuevas configuraciones: `configs/train/cropping_mobilenet_v3_large_v002_control_seed43.yaml` y `cropping_mobilenet_v3_large_v002_control_seed44.yaml`.
- Solo cambian semilla, nombre de corrida y metadatos de semilla. Se verifican la receta, los hashes del pipeline y el checkpoint común antes de lanzar.
- Las repeticiones se usan para describir estabilidad. Una semilla con mayor F1 no reemplaza automáticamente a la 42 y no se construye un ensemble.
- La media y desviación estándar se calculan sobre los F1 de los tres mejores checkpoints seleccionados con val. Son estadísticas de validación y la desviación estándar entre semillas no es un intervalo de confianza.
- Test permanece reservado.

## Ejecución, progreso y continuación

Desde la raíz del repositorio:

```powershell
$env:VIRTUS_PROGRESS = '1'
.\.venv\Scripts\python.exe -u scripts/train/run_control_stability.py
```

El lanzador conserva la corrida completa de la semilla 42 y ejecuta 43 y 44 una a la vez cuando faltan. Con la campaña actual ya completada, repetirlo reutiliza las tres corridas y no vuelve a entrenarlas; esa reutilización se verificó tras el cierre. Cada entrenamiento usa el lanzador visible existente, con dispositivo CUDA explícito y dependencias ya instaladas. El bloqueo de campaña y el marcador del lanzador evitan ejecuciones duplicadas.

El estado global se guarda en `reports/crops_v002/control_stability/campaign.json` y `progress.json`. Los procesos de entrenamiento guardan `best.pt`, `last.pt`, historia y cambios de pesos por época. La barra de la época avanza por lote y el arranque del primer lote imprime avisos periódicos.

Para reanudar una interrupción, repetir el mismo comando. Las semillas terminadas se reutilizan; las incompletas continúan desde su último `last.pt`. Si se interrumpe antes del primer guardado, esa semilla vuelve a comenzar desde C. No se seleccionan corridas por su puntuación: se conserva la primera corrida completa que cumple el protocolo.

Para pausar al finalizar una época, crear `pause.request` dentro de la carpeta de la corrida activa y esperar al guardado. El lanzador terminará antes de iniciar la siguiente semilla. Repetir el comando retoma el último checkpoint. Los lotes parciales de una época interrumpida no se consideran un punto de reanudación.

## Análisis final

Con las tres semillas completadas, el análisis puede regenerarse así:

```powershell
.\.venv\Scripts\python.exe -u scripts/train/analyze_control_stability.py
```

El análisis usa únicamente las predicciones y métricas finales de val que ya guardó cada entrenamiento. Verifica su cohorte, clases, matrices de confusión, inicialización común, pesos de pérdida y receta. No ejecuta inferencia nueva.

Resultados en `reports/crops_v002/control_stability/results/`:

- `results_by_seed.csv`, `summary.json`: F1, exactitud, épocas, media, desviación estándar muestral y rango entre las tres semillas.
- `paired_comparisons.csv`: diferencias pareadas de cada semilla frente a C y de 43/44 frente a 42; bootstrap de observaciones con 10 000 remuestreos y semilla 42.
- `per_genus_stability.csv`, `observation_errors.csv`, `error_overlap.json`: soporte, variación por género y errores compartidos.
- `parameter_drift_summary.csv`, `selected_checkpoint_drift.csv`: trayectoria de pesos y distancia del checkpoint elegido respecto a C, separando parámetros y buffers. Las épocas seleccionadas pueden diferir entre semillas.
- `drift_at_common_batches.csv`: comparación a épocas y lotes comunes, con comprobación de las actualizaciones reales del optimizador.
- `figures/`: curvas de entrenamiento, comparación de validación y trayectoria de pesos.
- `selection.json`: candidato 42 preservado, acompañado del estudio de semillas.
- `analysis_manifest.json`: fuentes y hashes; `bootstrap_state.npz`: guardado del remuestreo que permite reanudar.

Los intervalos por observación no incluyen toda la incertidumbre por semillas o por sitio-día y no corrigen comparaciones múltiples. La referencia C y cada mejor checkpoint ya fueron seleccionados con val. Tres semillas permiten describir variación observada, con especial atención a géneros de soporte pequeño; no garantizan generalización.

La selección histórica de S5 y los informes del piloto y del ajuste pareado se conservan.
