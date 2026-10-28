# Sensibilidad a recortes: MobileNetV3-Large v002

Validación oficial únicamente. Mismo checkpoint C congelado en todas las variantes.

| Política | F1 macro observación | F1 macro foto | Exactitud obs. | Vistas/foto |
|---|---:|---:|---:|---:|
| control | 0.9717 | 0.9188 | 0.9768 | 1 |
| center75 | 0.9662 | 0.9274 | 0.9771 | 1 |
| center50 | 0.9660 | 0.9132 | 0.9751 | 1 |
| multi5 | 0.9661 | 0.9254 | 0.9794 | 5 |

| Recorte frente a control | Δ F1 macro obs. | IC 95 % pareado | Aciertos ganados | Aciertos perdidos |
|---|---:|---|---:|---:|
| center75 | -0.0055 | [-0.0182, +0.0039] | 30 | 29 |
| center50 | -0.0057 | [-0.0192, +0.0042] | 37 | 43 |
| multi5 | -0.0056 | [-0.0224, +0.0063] | 40 | 31 |

Procesamiento recomendado en esta etapa: **control**.

Los recortes cambian las predicciones. Los parámetros y los buffers de BatchNorm se verificaron sin cambios.
Las cinco regiones se promedian por foto; luego las fotos se promedian por observación. No se crean muestras independientes.
El porcentaje de área corresponde al recorte adicional previo a Resize y CenterCrop.

Bootstrap pareado por observación: 10000 remuestreos, semilla 42, IC percentil; comparaciones exploratorias sin corrección múltiple.
El tiempo de forward excluye lectura y transformaciones; no representa latencia en Android.
Estos resultados no prueban el efecto de reentrenar. Un ajuste debe contrastarse con una continuación de control desde el mismo C.
No se ejecutó test ni se reemplazó la selección histórica de S5.

Figuras: [F1 e intervalos](figures/validation_f1.png) · [Matrices de confusión](figures/confusion_observations.png).
Galería de train: `gallery_train/gallery_train.png` (fotos locales, excluidas de Git).
