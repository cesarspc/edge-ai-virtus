# Comparación completada: control y recortes

Finalización de las ramas:

- control: 4 épocas ejecutadas de un máximo de 10; mejor época 1; finalizó por parada temprana.
- cropped: 4 épocas ejecutadas de un máximo de 10; mejor época 1; finalizó por parada temprana.

Cuando se activa la parada temprana, no es necesario completar el límite de épocas de la corrida.

| Entrenamiento | Inferencia | Épocas ejecutadas | Mejor época | F1 macro obs. | F1 macro foto |
|---|---|---:|---:|---:|---:|
| control | original | 4 | 1 | 0.9773 | 0.9332 |
| control | center75 | 4 | 1 | 0.9738 | 0.9350 |
| cropped | original | 4 | 1 | 0.9661 | 0.9200 |
| cropped | center75 | 4 | 1 | 0.9770 | 0.9346 |

| Inferencia común | Δ F1 recortado − control | IC 95 % pareado |
|---|---:|---|
| original | -0.0112 | [-0.0270, -0.0009] |
| center75 | +0.0032 | [-0.0010, +0.0077] |

Se conserva el procesamiento original y se registra la continuación de control como candidato de esta comparación por su mayor F1 macro observado. No hay ventaja concluyente por entrenar con recortes cuando ambos modelos reciben el recorte central del 75 %.
El modelo adaptado pierde rendimiento con el encuadre original. Estas conclusiones se limitan a esta receta y una semilla.

La inferencia del piloto se reutilizó tras comparar exactamente todos los parámetros y buffers de los mejores checkpoints. Sus archivos .pt tienen hashes diferentes por serialización, pero su estado del modelo es idéntico. Las predicciones propias de las corridas completadas reproducen las matrices guardadas.
Los intervalos se recalculan desde probabilidades guardadas: bootstrap pareado por observación, 10 000 remuestreos, semilla 42, sin corrección múltiple ni ajuste por sitio-día o variación entre semillas.

Las curvas de pesos incluyen el historial completo. La tabla de filas de la cabeza por género corresponde al mejor checkpoint seleccionado, no a los pesos de la última época.
La pérdida de train disminuye mientras el F1 de val cae respecto al mejor punto. Aumentar épocas con esta receta no aportó una mejora observada.
Las fuentes, identificadores, clases y particiones oficiales de v002 se mantienen. No se ejecutó inferencia nueva ni test. La selección histórica de S5 permanece intacta.

Figuras: [Entrenamiento](figures/training_curves.png) · [Cambios de pesos](figures/weight_drift.png) · [Matrices](figures/confusion_observations.png).
Auditoría y trazabilidad: `reuse_audit.json`. Candidato de validación: `selection.json`.
