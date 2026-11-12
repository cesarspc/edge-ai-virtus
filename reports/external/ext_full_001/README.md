# Evaluación externa completa: `test_external` de `v001` (`ext_full_001`)

> Generado por `scripts/train/evaluate_external.py`; no editar a mano.

Conjunto: [`data/versions/ext_full_001/`](../../../data/versions/ext_full_001/DATASET_CARD.md), huella `61ce96b14844c4b5…`, integridad de imágenes `717a5457b9d4d965…`. 4.895 observaciones y 9.649 imágenes evaluadas; se descargaron todas las imágenes.

Modelos entrenados solo con FungiTastic (`v002`, Dinamarca). Prueba de cambio de dominio: no se usó para seleccionar modelos. Unidad estadística: la observación (`mean_prob`). IC95 por bootstrap de observaciones.

## Resultado global

| Modelo | Macro-F1 obs. (10 clases) | IC95 | Exactitud obs. | Exactitud balanceada obs. | Top-3 obs. | Macro-F1 img. | Macro-F1 obs. en FungiTastic |
|---|---:|---|---:|---:|---:|---:|---:|
| `s5_mobilenet_v3_large_v002_c` | 0,766 | [0,747; 0,786] | 0,858 | 0,851 | 0,934 | 0,729 | 0,972 (val) |

## Por fuente y por región

En los grupos donde falta alguna clase, el macro-F1 sobre 10 clases cuenta como 0 las clases ausentes; la columna «clases presentes» es la comparable.

| Modelo | Grupo | Obs. | Clases presentes | Macro-F1 obs. (clases presentes) | IC95 | Exactitud obs. | IC95 |
|---|---|---:|---:|---:|---|---:|---|
| `s5_mobilenet_v3_large_v002_c` | **Colombia** (iNaturalist + MO) | 3348 | 10 | 0,772 | [0,746; 0,797] | 0,883 | [0,873; 0,894] |
| `s5_mobilenet_v3_large_v002_c` | iNaturalist CO | 2364 | 10 | 0,777 | [0,734; 0,812] | 0,929 | [0,918; 0,939] |
| `s5_mobilenet_v3_large_v002_c` | Mushroom Observer CO | 984 | 10 | 0,714 | [0,674; 0,747] | 0,774 | [0,748; 0,800] |
| `s5_mobilenet_v3_large_v002_c` | **MIND.Funga v2** (sin ubicación) | 1547 | 8 | 0,763 | [0,737; 0,787] | 0,803 | [0,784; 0,823] |
| `s5_mobilenet_v3_large_v002_c` | **Orinoquía** | 309 | 8 | 0,721 | [0,636; 0,786] | 0,770 | [0,722; 0,816] |
| `s5_mobilenet_v3_large_v002_c` | Otra región de Colombia | 3029 | 10 | 0,772 | [0,744; 0,796] | 0,895 | [0,883; 0,906] |
| `s5_mobilenet_v3_large_v002_c` | Colombia, sin departamento | 10 | 4 | 0,750 | [0,250; 0,750] | 0,900 | [0,700; 1,000] |

## Por género (nivel de observación)

`s5_mobilenet_v3_large_v002_c`. Soporte y F1 por grupo.

| Género | Soporte Completo | F1 Completo | Soporte Colombia | F1 Colombia | Soporte MIND.Funga | F1 MIND.Funga |
|---|---:|---:|---:|---:|---:|---:|
| Amanita | 1549 | 0,972 | 1445 | 0,977 | 104 | 0,898 |
| Auricularia | 578 | 0,638 | 238 | 0,570 | 340 | 0,684 |
| Ganoderma | 369 | 0,845 | 241 | 0,884 | 128 | 0,767 |
| Geastrum | 420 | 0,931 | 93 | 0,835 | 327 | 0,958 |
| Hericium | 6 | 0,240 | 6 | 0,324 | 0 | — |
| Lycoperdon | 130 | 0,656 | 111 | 0,745 | 19 | 0,390 |
| Morchella | 175 | 0,966 | 175 | 0,974 | 0 | — |
| Pleurotus | 364 | 0,790 | 219 | 0,766 | 145 | 0,821 |
| Trametes | 801 | 0,835 | 554 | 0,852 | 247 | 0,800 |
| Xylaria | 503 | 0,789 | 266 | 0,796 | 237 | 0,782 |


## Orinoquía

Observaciones con departamento Meta, Casanare, Arauca o Vichada (mismo criterio que la descarga GBIF `0007586-260221153910048`). Lista completa con enlace a cada registro y la predicción de cada modelo en [`orinoquia_predictions.csv`](orinoquia_predictions.csv).

### Por género (nivel de observación)

| Género | Soporte | Recall `s5_mobilenet_v3_large_v002_c` | F1 `s5_mobilenet_v3_large_v002_c` |
|---|---:|---:|---:|
| Amanita | 3 | 1,000 | 0,400 |
| Auricularia | 73 | 0,452 | 0,574 |
| Ganoderma | 24 | 0,875 | 0,737 |
| Geastrum | 19 | 0,947 | 0,973 |
| Lycoperdon | 6 | 0,667 | 0,615 |
| Pleurotus | 10 | 1,000 | 0,714 |
| Trametes | 143 | 0,853 | 0,836 |
| Xylaria | 31 | 0,871 | 0,915 |

Sin observaciones en la Orinoquía: Hericium, Morchella.

### Por departamento y fuente (exactitud de observación)

| Departamento | Fuente | Obs. | `s5_mobilenet_v3_large_v002_c` |
|---|---|---:|---:|
| Arauca | iNaturalist CO | 2 | 1,000 |
| Casanare | iNaturalist CO | 78 | 0,667 |
| Casanare | Mushroom Observer CO | 127 | 0,772 |
| Meta | iNaturalist CO | 90 | 0,822 |
| Meta | Mushroom Observer CO | 6 | 1,000 |
| Vichada | iNaturalist CO | 6 | 1,000 |

## Lectura y límites

* Las etiquetas externas son identificaciones de comunidad (iNaturalist grado de investigación; consenso de votos de Mushroom Observer), no curaduría experta como FungiTastic: la caída frente a FungiTastic mezcla cambio de dominio con ruido de etiqueta.
* MIND.Funga v2 no tiene coordenadas, fecha ni país; sus etiquetas vienen del nombre de carpeta y mezcla imágenes neotropicales de varios orígenes (una imagen por observación). Se reporta aparte de Colombia y no entra en ningún grupo regional.
* La Orinoquía tiene 8 de los 10 géneros y Trametes concentra cerca de la mitad de sus observaciones; las clases con pocas observaciones tienen intervalos amplios.
* En la Orinoquía, iNaturalist CO: 176 observaciones en 152 sitio-días, 41 comparten sitio-día; Mushroom Observer CO: 133 observaciones en 71 sitio-días, 90 comparten sitio-día (sitio-día de `v001`: misma fecha y celda de ~1 km). El bootstrap remuestrea observaciones, no sitio-días, así que los IC95 de ese grupo pueden ser optimistas.
* Una sola semilla por modelo; las diferencias entre modelos cuyos IC95 se solapan no son concluyentes.
