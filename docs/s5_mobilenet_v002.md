# S5: arquitectura móvil, ajustes y desbalance

Estado: campaña en curso, actualizado el 12 de octubre de 2026. Alcance: S5.
Destino móvil indicado por el usuario: **Android**; modelo de teléfono aún sin especificar.

## Punto de partida revisado

ResNet-18 completó 16 épocas; seleccionó la época 11 y evaluó test una vez.
La elección usa macro-F1 por observación en val: 0,9545. Las épocas 12–16 no la superaron.
Se conserva el resultado como referencia; la nueva selección se realiza exclusivamente con val.

Datos: `v002`, FungiTastic, diez géneros, particiones oficiales y misma huella publicada.
Se mantienen entrada 224, transformaciones, batch 64, AdamW, weight decay 0,05,
label smoothing 0,1, semilla 42, máximo 30 épocas y paciencia 5.
No se reconstruye ni modifica el dataset para este cambio de arquitectura.

## Arquitectura

Se utiliza **MobileNetV3-Large**, incluida en el alcance previo del repositorio.
La cabeza se adapta a diez géneros y se ajusta toda la red. Se fijan explícitamente
pesos `IMAGENET1K_V1`, evitando que `DEFAULT` cambie la receta de preentrenamiento.
El procesamiento de evaluación permanece igual al de la baseline (resize 224 y crop 224)
para esta comparación controlada. La receta estándar de esos pesos usa resize 256;
esta elección de 224 es deliberada y debe describirse en la comparación.

[Documentación oficial de MobileNetV3-Large](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_large.html).

## Caracterización preliminar de arquitectura

Medición local con diez salidas, pesos aleatorios, FP32, batch 1, tensor sintético
1×3×224×224, CPU del PC, cuatro hilos, cinco calentamientos y treinta mediciones.
No utiliza fotografías de train, val o test. No mide el tiempo de entrenamiento.

| Arquitectura | Parámetros | Estado serializado | Mediana CPU | P90 CPU |
|---|---:|---:|---:|---:|
| ResNet-18 | 11.181.642 | 44,80 MB | 37,10 ms | 59,17 ms |
| MobileNetV3-Large | 4.214.842 | 17,06 MB | 20,70 ms | 45,25 ms |

La reducción de parámetros es del 62,31 %. El estado serializado no incluye optimizador
ni mide la memoria máxima de ejecución. Las latencias excluyen decodificación,
redimensionamiento y normalización; pueden variar por carga del equipo.
**Estas mediciones son de PC, no de Android.** Deben medirse el modelo entrenado,
el runtime y el dispositivo Android concretos antes de afirmar una latencia móvil.
Resultados completos: `reports/s5/architecture_profile.json`.

## Experimentos acotados: cambiar dos factores

| Ensayo | Configuración | LR | Ponderación de clases |
|---|---|---:|---|
| A | `s5_mobilenet_v3_large_v002_a.yaml` | 0,0003 | Ninguna |
| B | `s5_mobilenet_v3_large_v002_b.yaml` | 0,0001 | Ninguna |
| C | `s5_mobilenet_v3_large_v002_c.yaml` | 0,0003 | Inversa de raíz del conteo |
| D | `s5_mobilenet_v3_large_v002_d.yaml` | 0,0001 | Inversa de raíz del conteo |

Primero ejecutar A y revisar val. B permite estudiar un paso de optimización menor;
C estudia la ponderación y D su combinación. Todos utilizan la misma semilla y
las mismas particiones; cuatro ensayos no constituyen una búsqueda exhaustiva ni
una estimación de variabilidad entre semillas.

Para C/D, los pesos de la pérdida se calculan **solo con las imágenes de train**:
`w_c = 1 / sqrt(n_c)` y se normalizan para que el promedio de los diez pesos sea uno.
Se evita ponderación por inversa directa, más extrema para clases pequeñas.
No se duplican fotografías ni se sobremuestrea el DataLoader. `class_balance.json`
registra conteos y pesos efectivos en cada corrida.
Val y test mantienen las proporciones oficiales y no se ponderan en la evaluación.

## Primer ensayo visible

Desde `D:\Dev\Virtus\edge-ai-virtus`, con el entorno existente:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/train_visible.ps1 -Config configs/train/s5_mobilenet_v3_large_v002_a.yaml -SkipInstall
```

La primera ejecución descarga los pesos de MobileNetV3-Large si no están en `.cache/torch`.
Las imágenes ya están disponibles. El lanzador muestra barras y conserva `last.pt`.
El lanzador imprime `[1/4]` inmediatamente y fuerza las barras en stdout mediante
`VIRTUS_PROGRESS=1`, incluso si la terminal no se detecta como interactiva.
La carga de bibliotecas y del modelo informa tiempo transcurrido cada cinco segundos;
la verificación de tablas e imágenes muestra barras con avance. Si no aparece ni
`[1/4]`, el lanzador aún no está ejecutándose en esa terminal.
El checkpoint de entrenamiento sigue guardándose al completar cada época y validación.
Para la prueba rápida opcional, añadir `-Smoke`; sus métricas no sirven para comparar modelos.
Para entrenar una época y revisar antes de continuar, añadir `-PauseAfterEpoch`.
Para continuar, usar la misma configuración y `-Resume "runs/<run_id>/last.pt"`.
Las corridas completas de estos ensayos terminan con métricas de **val únicamente**.

## Revisar e iterar con val

```powershell
.\.venv\Scripts\python.exe scripts/train/summarize_s5.py
```

Produce `reports/s5/development_results.md` y JSON con las corridas completas,
su macro-F1 por imagen y observación, LR, ponderación y hashes. Descarta smoke y
eval-only; rechaza ensayos que hayan usado test, datasets distintos o corridas
duplicadas de un ensayo hasta revisarlas.
Los checkpoints de las corridas pausadas todavía no participan en esta selección.

Revisar `metrics_val.json` y su matriz: F1, precisión, recall y soporte de todos los
géneros, particularmente Morchella, Hericium y Geastrum. Comparar A/B y A/C antes
de atribuir una diferencia a LR o ponderación. Un resultado perfecto de Morchella
con solo once observaciones de val no demuestra robustez general.

## Cerrar el ajuste y evaluar test al final

La selección principal es la mayor macro-F1 de observación en val entre los ensayos
completados. Los empates exactos se resuelven por identificador de corrida;
no se utiliza test para desempatar. Si se decide cerrar tras menos de cuatro ensayos,
se registra cuáles quedaron pendientes y no se afirma que se compararon todos.

Para fijar explícitamente el mejor ensayo completo:

```powershell
.\.venv\Scripts\python.exe scripts/train/summarize_s5.py --select-run "runs/<run_id>"
```

Después de revisar la selección:

```powershell
.\.venv\Scripts\python.exe scripts/train/evaluate_s5_final.py
```

Este último comando evalúa solo test del checkpoint seleccionado; no entrena ni
descarga pesos. Comprueba que checkpoint, configuración efectiva, val y dataset
coincidan con los hashes de `selected_model.json`. Conserva métricas/predicciones
en `runs/<run_id>/final_test/`. Un marcador exclusivo de campaña evita evaluaciones
concurrentes o repeticiones automáticas. Si falla una vez iniciada la evaluación,
se requiere revisar el estado antes de considerar una recuperación.
Después de iniciar test, el entrenamiento de ajuste de esta campaña queda cerrado.
La baseline de ResNet-18 ya tiene sus resultados guardados y no vuelve a evaluarse.

## Estado y pendientes de S5

- Configuraciones A–D, selección por val y evaluación final separada: preparadas.
- Perfil de tamaño y latencia sintética en PC: generado.
- Ensayo A completado: 24 épocas ejecutadas; mejor época 19; macro-F1 de observación en val 0,9706 y macro-F1 por imagen 0,9254. Early stopping activó tras cinco épocas sin superar el mejor resultado.
- Ensayos B, C y D: pendientes. El resumen vigente está en `reports/s5/development_results.md`.
- Test no se ha utilizado en la campaña S5; debe mantenerse reservado hasta seleccionar el ensayo final.
- Perfil en Android: pendiente de modelo de teléfono, runtime y acceso al dispositivo.
- El desarrollo de una aplicación móvil completa y las tareas de S6 no forman parte de este paso.
