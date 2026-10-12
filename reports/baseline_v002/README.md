# ResNet-18 v002: consolidación final

Corrida: `20261004T133123181686Z_ma_resnet18_v002`. Finalizó el 4 de octubre de 2026.
Se completaron **16 épocas**, se seleccionó la **época 11** y se realizó **una evaluación de test**.
Los gráficos y análisis utilizan los archivos guardados. No se entrenó ni se ejecutó otra inferencia.

## Rendimiento del checkpoint seleccionado

| Conjunto | Unidad | Soporte | Macro-F1 (0–1) | Exactitud |
|---|---|---:|---:|---:|
| Val | Imagen | 6.290 | 0,8997 | 92,07 % |
| Val | Observación | 3.452 | 0,9545 | 95,94 % |
| Test | Imagen | 7.159 | 0,8875 | 91,09 % |
| Test | Observación | 4.039 | 0,9463 | 95,79 % |

La selección usa macro-F1 de observación en val. Las épocas 12–16 no superaron la época 11.
La época 16 obtuvo mejor F1 por imagen, pero no mejoró el criterio establecido.

## Gráficas

- [Curvas de pérdida, macro-F1 y tasa de aprendizaje](../figures/baseline_v002_training_curves.png).
- [Matriz por imagen: conteos](../figures/baseline_v002_confusion_image_counts.png) y [porcentajes por fila](../figures/baseline_v002_confusion_image_percent.png).
- [Matriz por observación: conteos](../figures/baseline_v002_confusion_observation_counts.png) y [porcentajes por fila](../figures/baseline_v002_confusion_observation_percent.png).
- [F1 por género y soporte](../figures/baseline_v002_f1_by_genus.png).

En las matrices, las filas corresponden al género real y las columnas al predicho.
Los porcentajes se calculan respecto al soporte de cada fila; los conteos son enteros.

## Errores de validación

Hay **499 errores por imagen** y **140 por observación**. Las listas completas,
con identificadores, etiquetas y probabilidades guardadas, están en
[validation_image_errors.csv](validation_image_errors.csv) y
[validation_observation_errors.csv](validation_observation_errors.csv).
El promedio de probabilidades, exportadas con seis decimales, reprodujo exactamente
la matriz de observaciones guardada. No se volvió a utilizar el modelo.

| Género real | Predicho | Errores por observación | Soporte real |
|---|---|---:|---:|
| Amanita | Lycoperdon | 22 | 904 |
| Ganoderma | Trametes | 14 | 335 |
| Trametes | Ganoderma | 10 | 756 |
| Trametes | Lycoperdon | 9 | 756 |
| Hericium | Trametes | 5 | 66 |

La revisión visual de ejemplos muestra tomas parciales de superficies, ejemplares deteriorados
y una imagen de microscopía. Algunos errores reciben probabilidades altas. Son ejemplos
para revisión experta; no demuestran errores de etiquetado ni permiten cuantificar la
causa del error. No se modificaron etiquetas.
La galería `validation_examples_local_only.png` permanece solo en local y se excluye de Git.

## Clases con soporte reducido

| Género | Observaciones val | Aciertos val | F1 obs. val | Recall val | Intervalo Wilson 95 % del recall |
|---|---:|---:|---:|---:|---|
| Morchella | 11 | 11 | 1,0000 | 100,00 % | 74,12 %–100,00 % |
| Hericium | 66 | 59 | 0,9219 | 89,39 % | 79,69 %–94,77 % |
| Geastrum | 79 | 76 | 0,9325 | 96,20 % | 89,42 %–98,70 % |

Los intervalos aproximados corresponden al recall bajo un supuesto binomial, no al F1.
No ajustan posibles dependencias entre hallazgos. Morchella presenta cuatro imágenes
incorrectas de 21 en val, pese a acertar las 11 observaciones al combinar sus fotografías.
El soporte de test de Morchella es de 20 observaciones, con recall 90,00 % y F1 0,9474.
Las métricas de test se describen como resultados finales, sin usarlas para ajustar el modelo.

## Conservación y reproducibilidad

Se conservaron `best.pt`, configuración efectiva, manifiesto del dataset, historial,
métricas, predicciones y marcadores de finalización en:

`artifacts/baselines/resnet18_v002/20261004T133123181686Z_ma_resnet18_v002/`

SHA-256 de `best.pt`:
`a5edc643662266708263899c2fdd6977f0dabac160739ba7ccfde82603af50e3`.

La igualdad de las copias con los originales quedó verificada. La carpeta de corrida
no se modificó. [preservation_manifest.json](preservation_manifest.json) registra los hashes
del archivo de conservación; [consolidation_manifest.json](consolidation_manifest.json)
registra entradas, generador y salidas del análisis.

El informe formal actualizado está en `reports/dataset_v002_training_report.tex`,
con curvas autocontenidas, historial completo y matrices numéricas.
La compilación del editor integrado sigue sin confirmarse por un fallo de directorios del runtime.

Para regenerar los artefactos:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-report.txt
.\.venv\Scripts\python.exe scripts/train/consolidate_baseline.py
.\.venv\Scripts\python.exe scripts/reports/update_baseline_report.py
```
