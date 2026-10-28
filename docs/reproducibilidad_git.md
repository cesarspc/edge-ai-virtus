# Resultados y checkpoints publicados

Los checkpoints se versionan con Git LFS. Tras clonar el repositorio, ejecutar `git lfs install` y `git lfs pull` para disponer de los pesos completos. Un puntero LFS de texto no es un checkpoint cargable por PyTorch.

Se publican las corridas terminadas con predicciones válidas, incluyendo las que finalizaron por parada temprana. Los pilotos pausados, las corridas con probabilidades no finitas, las fotografías, los entornos virtuales y las cachés permanecen locales. No se borraron esos archivos locales al publicar.

Las reglas de `.gitattributes` conservan los bytes de los metadatos, fuentes y snapshots usados por las auditorías SHA-256. Algunas diferencias de fin de línea respecto a commits anteriores son necesarias para que los hashes registrados coincidan con los archivos publicados. No cambia el contenido semántico de las tablas ni las particiones de v002.

## Consultar los resultados

- Comparación de recortes: `reports/crops_v002/completed_seed42/RESULTS.md`.
- Estabilidad con tres semillas: `reports/crops_v002/control_stability/results/RESULTS.md`.
- Guías: `docs/recortes_v002.md` y `docs/control_stability_v002.md`.

Las tablas, predicciones finales y figuras pueden consultarse sin ejecutar entrenamiento ni inferencia. Las fuentes archivadas sirven para auditoría; no constituyen proyectos independientes ejecutables desde sus carpetas de snapshot.

## Regenerar una evaluación

La inferencia requiere reconstruir los metadatos y disponer de las fotografías locales de v002, con los hashes publicados y el entorno de entrenamiento. Véanse `docs/baseline_v002.md` y las guías de adquisición. No se incluyen fotografías del dataset en Git.

Para repetir la comparación 2 × 2 a partir de corridas completas publicadas, con los datos ya preparados:

```powershell
$env:VIRTUS_PROGRESS = '1'
.\.venv\Scripts\python.exe -u scripts/train/evaluate_crop_pair.py --control runs/crops_v002/20261028T141000472345Z_cropping_mobilenet_v3_large_v002_control_seed42 --cropped runs/crops_v002/20261028T145527817405Z_cropping_mobilenet_v3_large_v002_center75_seed42 --output reports/crops_v002/reproduced_seed42
```

Esto usa solo val y crea un informe nuevo. Los comandos históricos del consolidador que verifican los checkpoints de los pilotos originales requieren aquellos archivos locales pausados, que no se publican. Sus predicciones y pruebas de equivalencia de estados están conservadas en los informes; la evaluación nueva puede usar directamente los mejores checkpoints de las corridas completas.

El análisis de estabilidad utiliza las predicciones finales ya guardadas de las tres corridas completas. Mantiene la semilla 42 como candidato preseleccionado y no consulta test. Los marcadores de campaña permiten reutilizar corridas terminadas; no es necesario volver a entrenarlas para consultar sus resultados.
