# Propuesta: efecto de los recortes sobre MobileNetV3-Large

Estado: diseño original, con sensibilidad y ajuste pareado de la semilla 42 completados. Las dos ramas terminaron por parada temprana en la época 4; los resultados y comandos vigentes están en [recortes_v002.md](recortes_v002.md). Las repeticiones con varias semillas siguen siendo trabajo posterior.

## Objetivo

Determinar cómo cambia la clasificación de los diez géneros al modificar el encuadre de las fotografías y, mediante un ajuste posterior controlado, medir cómo cambian los parámetros aprendidos del clasificador.

La hipótesis es que un encuadre más cercano puede aportar información visual útil. También puede perder el hongo o partes diagnósticas, por lo que una mejora debe demostrarse con validación.

## Punto de partida verificado

- El candidato de referencia es C: `runs/20261024T031650621510Z_s5_mobilenet_v3_large_v002_c/best.pt`, seleccionado con F1 macro por observación en validación. Sus hashes están en `reports/s5/selected_model.json`.
- El clasificador es MobileNetV3-Large con diez salidas y entrada de 224 × 224; el ensayo C usa ponderación `inverse_sqrt` (configs/train/s5_mobilenet_v3_large_v002_c.yaml).
- El entrenamiento actual ya utiliza `RandomResizedCrop`, con área aleatoria entre 50 % y 100 % de la imagen, además de reflexión horizontal y variación de color (scripts/train/pipeline/transforms.py).
- La evaluación actual utiliza `Resize(224)` y `CenterCrop(224)`. Por tanto, el control es el procesamiento actual, que ya puede retirar bordes de fotografías no cuadradas; no se lo debe describir como una fotografía completa sin recortar (scripts/train/pipeline/transforms.py).
- Los pesos por clase se calculan con los conteos de fotos originales de train (scripts/train/pipeline/training.py).
- La política del proyecto usa val para seleccionar y reserva test para la evaluación final. Al revisar esta propuesta no existía el marcador de inicio del test final de S5.

## Qué significa observar los pesos

Hay dos cantidades distintas:

1. **Parámetros aprendidos:** tensores de las capas de MobileNet. Cambian al entrenar. Cambiar un recorte durante inferencia modifica las predicciones, pero no estos tensores.
2. **Pesos de la pérdida por género:** coeficientes `inverse_sqrt`. Se conservarán calculados sobre las mismas fotos originales; generar vistas adicionales no hará que un género tenga más peso de forma artificial.

El número de parámetros permanece igual porque se conserva la arquitectura. Recortar imágenes no comprime la red. Una mayor diferencia numérica entre pesos tampoco demuestra una mejora de clasificación.

## Etapa 1: sensibilidad al encuadre con C congelado

Reutilizar el mismo `best.pt` y las mismas observaciones de val. No realizar pasos de optimización. Verificar la integridad del dataset y cargar el modelo una sola vez para ejecutar todas las vistas.

| Variante propuesta | Imagen que recibe el procesamiento actual | Predicción por foto |
|---|---|---|
| Control | Foto original, sin recorte adicional | Una predicción |
| Central 75 | Recorte central que conserva aproximadamente 75 % del área original | Una predicción |
| Central 50 | Recorte central que conserva aproximadamente 50 % del área original | Una predicción |
| Varias regiones | Cinco recortes de área 50 %: centro y cuatro esquinas | Promedio de las cinco probabilidades |

Los porcentajes se refieren al área del recorte adicional **antes** de las transformaciones habituales, y no al área que finalmente ve la red después del recorte central de evaluación. Para conservar la relación de aspecto, un recorte de fracción de área `q` usa ancho `round(W * sqrt(q))` y alto `round(H * sqrt(q))`; se registran las coordenadas y el área efectiva después del redondeo. El control aplica exactamente las transformaciones existentes.

Las ubicaciones se fijan de antemano para todo el conjunto. Los recortes centrales no localizan por sí mismos el hongo. Se preparará una galería de fotos de train para revisar cómo se conservan sus partes y su contexto. Un recorte centrado en una región marcada manualmente podría estudiarse después como otro protocolo explícito.

El recorte central de la variante de cinco regiones puede reutilizarse para el resultado Central 50, evitando calcularlo dos veces.

Para varias regiones se agregan primero sus probabilidades por fotografía y después las fotografías por observación. Cada fotografía conserva el mismo peso aunque tenga más vistas. La regla no elige un recorte utilizando la etiqueta correcta de la imagen.

### Resultados de esta etapa

- F1 macro por observación, F1 por foto, exactitud y precisión/recall/F1 por género.
- Comparación pareada de cada variante con el control sobre las mismas observaciones.
- Observaciones que pasan de error a acierto y de acierto a error.
- Galería de encuadres y ejemplos de pérdida de partes diagnósticas.
- Tiempo por foto original y consumo de memoria de cada estrategia, incluyendo todas sus vistas.
- Comprobación de que los tensores del modelo son idénticos antes y después de la evaluación: los cambios de esta etapa son de comportamiento, no de pesos.

El control debe reproducir las decisiones de validación guardadas de C. Cualquier discrepancia debe explicarse antes de comparar los recortes. Todos los resultados deben contener probabilidades finitas.

## Etapa 2: ajuste controlado para estudiar cambios de parámetros

Elegir con val la política de recorte que merezca estudiar su aprendizaje. Una política que empeora en la etapa 1 todavía puede beneficiarse de adaptación; su resultado inicial mide sensibilidad al cambio de entrada, no su potencial definitivo tras entrenar. Si ninguna política aporta evidencia útil, no se asume que ampliar el entrenamiento vaya a mejorarla.

Crear dos ramas que partan del **mismo checkpoint de C**, con optimizador y scheduler nuevos:

| Rama | Entrada de entrenamiento |
|---|---|
| Control de ajuste | Procesamiento actual |
| Ajuste con recorte | Recorte adicional elegido y los mismos aumentos del control |

Ambas ramas reciben el mismo número de oportunidades de entrenamiento, las mismas observaciones, el mismo orden de lotes por semilla y la misma receta de optimización. Así se controla el efecto de continuar entrenando y de reiniciar el optimizador.

Receta inicial propuesta para el piloto: tasa `0.0001`, máximo 10 épocas, paciencia 3, lote de entrenamiento y evaluación 64, entrada 224 × 224 y ajuste de todas las capas. Estas son decisiones propuestas para ambas ramas, no valores ya validados. Se conservarían la ponderación de C y los demás ajustes comunes. El límite de lote también aplica al número total de vistas que entra a la GPU.

Para entrenar con varias regiones, seleccionar una vista por fotografía y época mediante la semilla registrada. Esto mantiene un paso de entrenamiento por foto original y evita aumentar el número de actualizaciones solo en esa rama. En validación se usa siempre la agregación de vistas prefijada.

Después del piloto, repetir las dos ramas con tres semillas emparejadas propuestas: 42, 43 y 44. Las tres arrancarían del mismo checkpoint de C; por tanto, medirían variación del ajuste posterior condicionada a esa inicialización, no toda la variación de entrenar MobileNet desde ImageNet.

Evaluar los dos modelos con los dos encuadres:

| Modelo entrenado | Evaluación con procesamiento actual | Evaluación con recorte elegido |
|---|---|---|
| Rama de control | Referencia del ajuste | Efecto de cambiar la vista |
| Rama con recortes | Transferencia al encuadre anterior | Resultado de la adaptación |

Comparar modelos en una misma columna permite estudiar el efecto de aprender con recortes. Comparar columnas con un mismo modelo permite estudiar el efecto del encuadre en inferencia. La evaluación principal sigue siendo por observación.

## Medición de cambios en los pesos

Guardar la inicialización común una vez y calcular al finalizar cada época:

`delta_relativa(capa) = ||W_actual - W_inicial||_2 / (||W_inicial||_2 + epsilon)`

- Registrar capa, época, pasos completados, norma inicial y actual, cambio absoluto y relativo.
- Separar parámetros del extractor visual y de la cabeza clasificadora.
- Registrar por separado las estadísticas de BatchNorm, que son estado del modelo y no parámetros actualizados mediante gradientes.
- Comparar las dos ramas también en épocas o pasos comunes: si la parada temprana ocurre en momentos distintos, sus checkpoints finales tienen diferente número de actualizaciones.
- Relacionar los cambios con F1 y errores por género. La magnitud del cambio es descriptiva; no identifica por sí sola qué característica visual aprende una capa.

Los pesos iniciales pueden mantenerse en CPU para no duplicar su almacenamiento en la memoria de la GPU durante el entrenamiento.

## Datos y evaluación

- Mantener `v002`, sus fuentes, etiquetas, identificadores y particiones oficiales. Cada vista hereda el `image_uid`, `observation_uid` y split originales, con un identificador adicional de vista.
- Aplicar los recortes como transformaciones configuradas de una campaña nueva: `cropping_mobilenet_v3_large_v002`. No sobrescribir fotografías ni la selección histórica de S5.
- Registrar la huella de v002 y una huella independiente de la política de recorte, sus coordenadas y la inicialización. Si se excluyen imágenes o se cambia el contenido publicado, crear una nueva versión conforme a `data/versions/README.md`.
- Calcular la pérdida y los pesos de clase con el conteo de fotografías originales, no con el conteo de recortes.
- Comparar F1 con bootstrap pareado por observación. Los recortes de una misma foto no constituyen muestras independientes.
- Conservar la agregación por observación existente. Informar cambios por género y el coste adicional de procesar múltiples vistas.
- Cerrar la comparación completa antes de consultar test. La nueva campaña debe identificar de forma explícita el modelo y el procesamiento seleccionados para la evaluación final; no se ejecutará automáticamente el evaluador de S5 para cada variante.

## Implementación propuesta

| Archivo o componente | Cambio previsto |
|---|---|
| `scripts/train/pipeline/transforms.py` | Política opcional de recorte adicional, desactivada por defecto; coordenadas y versión registradas. |
| `scripts/train/pipeline/datasets.py` | Conservar la identidad de la foto y su observación al producir una o varias vistas. |
| `scripts/train/pipeline/evaluation.py` | Agregación por vista → foto → observación y controles de valores finitos. |
| `scripts/train/train_ma.py` | Añadir `--init-checkpoint` para iniciar una campaña nueva desde pesos guardados, reiniciando optimizador e historia. |
| `scripts/train_visible.ps1` | Exponer la inicialización y conservar progreso, guardados y reanudación. |
| `scripts/train/pipeline/artifacts.py` | Registrar política de recorte, checkpoint inicial, hashes y relaciones entre vistas y fotografías. |
| `scripts/train/evaluate_crop_sensitivity.py` | Nuevo diagnóstico de la etapa 1; carga y verificación únicas, progreso por variante y lote. |
| `scripts/train/analyze_crop_experiments.py` | Nuevo análisis de métricas pareadas y cambios de parámetros por capa. |
| `configs/train/cropping_*_v002.yaml` | Configuraciones independientes para el control y el ajuste con recortes. |
| `reports/crops_v002/` | Tablas, curvas, galerías, métricas y selección de la nueva comparación. |

`--init-checkpoint` y `--resume` tienen propósitos distintos. El primero abre una comparación nueva con pesos comunes. El segundo continúa una corrida interrumpida y debe seguir exigiendo la misma configuración y dispositivo, como hace actualmente el repositorio.

## Progreso, puntos de guardado y aceptación

- Barra durante la verificación, generación de vistas, inferencia y entrenamiento; imprimir la etapa desde el inicio.
- Guardar resultados parciales de inferencia por bloques y permitir reanudación solo con los mismos hashes de datos, modelo y recorte.
- Mantener `last.pt`, `best.pt` y guardado atómico por época. La medición de parámetros se escribe en un CSV acumulado y vinculado al checkpoint.
- Aceptar la implementación cuando el control reproduzca el procesamiento actual, las vistas conserven su partición, las probabilidades sean finitas y una interrupción pueda retomarse correctamente.
- Verificar que la inicialización sea idéntica entre ramas y que la etapa 1 no modifique parámetros ni estadísticas del modelo.
- Elegir el procesamiento con validación y una regla fija de agregación. Si el intervalo pareado incluye cero, describir la diferencia como no concluyente y valorar el coste de inferencia.
- No establecer de antemano una mejora mínima ni un tiempo total: medir un piloto para decidir si el resultado justifica el coste del experimento.

## Referencias técnicas

- Código y configuración locales indicados en cada apartado.
- [Torchvision: RandomResizedCrop](https://docs.pytorch.org/vision/main/generated/torchvision.transforms.RandomResizedCrop.html): el intervalo de escala corresponde a área de la imagen, no a porcentaje de lado.
- [Torchvision: CenterCrop](https://docs.pytorch.org/vision/main/generated/torchvision.transforms.CenterCrop.html): define el recorte centrado utilizado en el procesamiento actual.
