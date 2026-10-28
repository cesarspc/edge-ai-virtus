# Piloto pareado de ajuste con recortes

Ambas ramas parten exactamente de C. Optimizador e historial reiniciados. Solo validación.

| Entrenamiento | Inferencia | Mejor época | F1 macro obs. | F1 macro foto |
|---|---|---:|---:|---:|
| control | original | 1 | 0.9773 | 0.9332 |
| control | center75 | 1 | 0.9738 | 0.9350 |
| cropped | original | 1 | 0.9661 | 0.9200 |
| cropped | center75 | 1 | 0.9770 | 0.9346 |

Comparar filas con la misma inferencia permite estudiar el cambio de entrenamiento. Comparar columnas de un mismo modelo mide el efecto del encuadre.

| Inferencia común | Δ F1 cropped − control | IC 95 % pareado |
|---|---:|---|
| original | -0.0112 | [-0.0270, -0.0009] |
| center75 | +0.0032 | [-0.0010, +0.0077] |

Los CSV de cambios de pesos separan backbone, cabeza y buffers de BatchNorm; indican épocas, lotes y actualizaciones reales de AdamW.
Una mayor distancia de pesos no implica mejor clasificación. La comparación a lotes comunes distingue también posibles pasos omitidos por AMP.
Este es un piloto de una semilla desde una inicialización ya seleccionada; una época comprueba el mecanismo, no estima el potencial final del ajuste.
Los intervalos son bootstrap pareado por observación, 10 000 remuestreos, semilla 42, exploratorios sin corrección múltiple.
Las ramas conservan last.pt y pueden continuar con su configuración original. No se ha consultado test ni sustituido C.
