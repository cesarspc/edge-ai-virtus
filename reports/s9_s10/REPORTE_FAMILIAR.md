# S9 y S10 de Virtus, explicado sin enredos

Fecha del reporte: 13 de noviembre de 2026.

## Qué hicimos

Usamos las respuestas guardadas de MobileNetV3-Large S5-C, la misma versión visual en validación y en la evaluación externa. Entrenamos tres clasificadores de contexto en CPU con 17.799 observaciones de entrenamiento. Cada uno aprende únicamente de GPS; GPS y clima; o GPS, clima y sustrato. Combinamos 90 % de la respuesta fotográfica con 10 % de la respuesta del contexto. Ese peso, la regularización y el resto de las reglas quedaron fijos antes de calcular los resultados. Si no hay contexto utilizable, la respuesta es exactamente la de la foto. No volvimos a entrenar la red de imágenes: es una prueba de combinación final con la parte visual congelada.

## Resultado en 3.452 observaciones de validación

| Modelo | Qué recibe | Aciertos | Errores | Acierto % | F1 por género /100 |
| --- | --- | --- | --- | --- | --- |
| A | Solo foto | 3.372 | 80 | 97,68 | 97,17 |
| B | Foto + GPS | 3.372 | 80 | 97,68 | 97,17 |
| C | Foto + GPS + clima | 3.373 | 79 | 97,71 | 97,20 |
| D | Foto + GPS + clima + sustrato | 3.373 | 79 | 97,71 | 97,19 |

F1 por género da el mismo peso a cada género, aunque uno tenga muchas más observaciones. No es el porcentaje de aciertos.

El clima corrigió 2 errores e introdujo 1: el saldo fue 1 acierto adicional. Al sumar sustrato se corrigieron 2 errores y aparecieron 2; el total de aciertos no aumentó.

Las diferencias son demasiado pequeñas para demostrar una mejora estable en este ensayo. No se cambió el peso del contexto para buscar un resultado favorable.

## S9: clima

S9 añade 19 variables de clima CHELSA: describen condiciones típicas de 1981 a 2010, no el tiempo del día de la foto. Hay clima completo en 17.782 de 17.799 observaciones de train y en 3.442 de 3.452 de val. Los datos faltantes se conservan con máscaras; no eliminamos esos hongos de la comparación. Este producto sustituye explícitamente el ERA5-Land propuesto para este ensayo. No incluye humedad relativa.

C menos B: +0.0277 puntos de F1; intervalo del 95 % [-0.0361, +0.1015] puntos. Incluye cero.

## S10: sustrato

Sustrato significa sobre qué crece el hongo: madera, suelo, hojarasca, etc. No es lo mismo que hábitat, que describe el entorno. En train había 1.811 registros sin detalle, frente a uno en val: todos decían solamente madera. Unificamos madera viva, muerta y sin detalle bajo la misma categoría amplia. La calidad o el año no entran como pistas para predecir el género. Las categorías desconocidas tienen una marca propia.

D menos C: -0.0070 puntos de F1; intervalo del 95 % [-0.0892, +0.0715] puntos. Incluye cero.

Los intervalos se calcularon con 2.000 remuestreos pareados por observación, semilla 42. Describen esta comparación; no corrigen el uso previo de val.

## Evaluación externa

Verificamos nuevamente las huellas digitales de las 9,649 fotografías externas. El conjunto completo tiene 4.895 observaciones: incluye 3.348 de Colombia y 1.547 de MIND.Funga. Dentro de las colombianas, 309 corresponden a la Orinoquía. Son grupos que se contienen entre sí; no se suman. No se usaron para entrenar, elegir pesos ni seleccionar una variante. Ya habían sido evaluados en una fase anterior: esta es una revisión de robustez sobre un conjunto conocido, no un test externo nuevo e intacto.

| Grupo | Observaciones | Foto % | C sin protección % | D sin protección % | D protegido % |
| --- | --- | --- | --- | --- | --- |
| Completo | 4895 | 85,78 | 85,25 | 85,72 | 85,78 |
| Colombia | 3348 | 88,32 | 87,54 | 88,23 | 88,32 |
| Orinoquía | 309 | 77,02 | 76,05 | 77,02 | 77,02 |

La tabla externa no trae las 19 variables CHELSA ni el sustrato necesario para estos modelos. Se dejaron ausentes. Por eso la evaluación externa mide el comportamiento con GPS y datos faltantes; no demuestra que el clima o el sustrato mejoren la identificación fuera de FungiTastic. La ubicación aprendida principalmente en Dinamarca puede ser una pista equivocada en Colombia. La columna sin protección aplica directamente el contexto disponible. La protección geográfica desactiva el contexto fuera de los límites de ubicación aprendidos solo de train (percentiles 0,5 y 99,5). Es una regla previa, no un ajuste a partir de las etiquetas externas; no es un detector de dominio validado. En esta muestra las ubicaciones externas no pasan esa regla; la ruta protegida usa la foto.

## Límites

- Se evalúan diez géneros, no una identificación definitiva de especie.
- Las observaciones de validación ya se usaron para seleccionar el modelo visual. No son una prueba nueva e independiente.
- Las normales climáticas pueden incluir años posteriores a 745 registros antiguos de train; el análisis es retrospectivo.
- Sustrato y hábitat son anotaciones de la fuente; no suponemos que una persona pueda reconocerlos al tomar una foto.
- La microscopía, las fotos con poco detalle y las vistas heterogéneas necesitan revisión específica; se conserva la revisión preliminar de seis casos de S8.
- No se ejecutó un ajuste conjunto en GPU ni una prueba de latencia o funcionamiento real en Android.

## Qué sigue

- Conservar la foto como referencia: este ensayo no justifica dar por ganada la incorporación de clima o sustrato.
- En el equipo con GPU, repetir A/B/C/D con el mismo protocolo y varias semillas; no mezclar B solo GPS con B foto + GPS.
- Conseguir clima externo compatible y sustrato fiable antes de afirmar una mejora multimodal en Colombia.
- En Android, foto y fecha se capturan localmente; GPS depende de permiso y señal. Clima remoto llega al sincronizar. Sustrato debe permitir 'no sé'.

## Reproducir y revisar los datos

```powershell
python scripts/data/prepare_s8_s9_context.py
python scripts/train/run_s9_s10_frozen.py --external
python scripts/reports/build_s9_s10_report.py
```

Resultados completos: `reports/s9_s10/frozen_context/`. Configuración fija: `configs/experiments/s9_s10_frozen_context.yaml`.
Modelos de contexto: `models/B.json`, `C.json`, `D.json` (categorías, escalas y coeficientes).
Sensibilidades de ausencia/permutación: `validation_summary.csv`. Intervalos: `paired_validation.json` y `paired_external.json`.
Las probabilidades visuales se reutilizaron; no se recalcularon. Los archivos conservan hashes y procedencia. No se evaluó test interno.

Fuentes: [datos climáticos FungiTastic](https://arxiv.org/html/2408.13632v2), [contrato Android](../../docs/android_metadatos.md), [revisión visual](../s9/visual_review/REVIEW.md).
