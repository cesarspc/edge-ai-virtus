# Estabilidad del control: semillas 42, 43 y 44

Las tres repeticiones ajustan el mismo checkpoint C con la misma receta y encuadre original. Solo cambia la semilla. Train y val conservan las particiones oficiales de v002.

| Semilla | Épocas ejecutadas | Mejor época | F1 macro obs. | F1 macro foto | Exactitud obs. |
|---|---:|---:|---:|---:|---:|
| 42 | 4 | 1 | 0.9773 | 0.9332 | 0.9803 |
| 43 | 5 | 2 | 0.9700 | 0.9281 | 0.9812 |
| 44 | 9 | 6 | 0.9723 | 0.9219 | 0.9806 |

F1 macro por observación: media **0.9732**, desviación estándar muestral **0.0037**, rango **0.9700–0.9773** (tres semillas). La desviación estándar no es un intervalo de confianza.
Referencia C antes del ajuste: **0.9717** en val.
Las mejores puntuaciones de las semillas 43 no superan el punto de referencia C. La mejora frente a C observada en la semilla 42 no apareció en las tres repeticiones; debe comunicarse junto con la dispersión entre semillas.

| Semilla | Referencia | Δ F1 macro | IC 95 % pareado por observación |
|---|---|---:|---|
| 42 | C | +0.0056 | [-0.0002, +0.0116] |
| 43 | C | -0.0017 | [-0.0144, +0.0076] |
| 43 | 42 | -0.0073 | [-0.0198, +0.0013] |
| 44 | C | +0.0006 | [-0.0122, +0.0099] |
| 44 | 42 | -0.0050 | [-0.0172, +0.0035] |

| Género con mayor variación | Soporte de val | F1 mínimo | F1 máximo | DE entre semillas |
|---|---:|---:|---:|---:|
| Morchella | 11 | 0.9565 | 1.0000 | 0.0251 |
| Hericium | 66 | 0.9206 | 0.9538 | 0.0166 |
| Geastrum | 79 | 0.9576 | 0.9811 | 0.0136 |

La semilla 42 se mantiene como candidato fijado antes de ejecutar las repeticiones. No se elige la semilla con mayor puntuación ni se combinan las probabilidades para crear un ensemble.
Cada fila utiliza su mejor checkpoint seleccionado con val. Las diferencias frente a C son comparaciones de validación, no resultados de test ni estimaciones independientes de la selección previa.
Errores: 40 observaciones fallan en las tres semillas; 67 reciben predicciones distintas entre semillas. Véanse `observation_errors.csv` y `per_genus_stability.csv`.
Los intervalos usan bootstrap pareado por observación, 10000 remuestreos, semilla 42, sin corrección múltiple. No representan la incertidumbre total por semillas o sitios-día.
El estudio describe variación del ajuste posterior condicionada a una inicialización C ya seleccionada. Con tres semillas no se afirma estabilidad universal ni se mide la variación de entrenar desde ImageNet con otras inicializaciones.
Las distancias de pesos de los checkpoints seleccionados se guardan por separado de las trayectorias completas. Sus épocas pueden ser distintas; `drift_at_common_batches.csv` permite comparar épocas/lotes comunes e indica si coinciden los pasos reales del optimizador. Una distancia mayor no implica mejor clasificación.
No se ejecutó inferencia nueva: se verificaron y analizaron los archivos finales de val ya guardados. No se consultó test ni se sustituyó la selección histórica de S5.

Figuras: [Validación](figures/validation_stability.png) · [Entrenamiento](figures/training_curves.png) · [Pesos](figures/weight_trajectories.png).
Trazabilidad: `analysis_manifest.json`. Candidato preservado: `selection.json`.
