# Evaluación externa en Colombia (`ext_co_001`)

> Generado por `scripts/train/evaluate_external.py`; no editar a mano.

Conjunto: [`data/versions/ext_co_001/`](../../../data/versions/ext_co_001/DATASET_CARD.md), huella `05c6a18e3217914b…`, integridad de imágenes `13883633afb519d7…`. 3.348 observaciones y 8.102 imágenes evaluadas; se descargaron todas las imágenes.

Modelos entrenados solo con FungiTastic (`v002`, Dinamarca). Prueba de cambio de dominio: no se usó para seleccionar modelos. Unidad estadística: la observación (`mean_prob`). IC95 por bootstrap de observaciones.

## Resultado global

| Modelo | Macro-F1 obs. (10 clases) | IC95 | Exactitud obs. | Exactitud balanceada obs. | Top-3 obs. | Macro-F1 img. | Macro-F1 obs. en FungiTastic |
|---|---:|---|---:|---:|---:|---:|---:|
| `ma_resnet18_v002` | 0,726 | [0,698; 0,752] | 0,843 | 0,788 | 0,953 | 0,668 | 0,946 (test) |
| `s5_mobilenet_v3_large_v002_c` | 0,772 | [0,746; 0,797] | 0,883 | 0,832 | 0,952 | 0,717 | 0,972 (val) |

## Por fuente y por región

En los grupos donde falta alguna clase, el macro-F1 sobre 10 clases cuenta como 0 las clases ausentes; la columna «clases presentes» es la comparable.

| Modelo | Grupo | Obs. | Clases presentes | Macro-F1 obs. (clases presentes) | IC95 | Exactitud obs. | IC95 |
|---|---|---:|---:|---:|---|---:|---|
| `ma_resnet18_v002` | iNaturalist CO | 2364 | 10 | 0,712 | [0,665; 0,748] | 0,877 | [0,864; 0,890] |
| `ma_resnet18_v002` | Mushroom Observer CO | 984 | 10 | 0,654 | [0,588; 0,705] | 0,759 | [0,732; 0,786] |
| `ma_resnet18_v002` | **Orinoquía** | 309 | 8 | 0,645 | [0,569; 0,707] | 0,670 | [0,621; 0,725] |
| `ma_resnet18_v002` | Otra región | 3029 | 10 | 0,729 | [0,698; 0,756] | 0,860 | [0,848; 0,872] |
| `ma_resnet18_v002` | Sin departamento | 10 | 4 | 1,000 | [0,500; 1,000] | 1,000 | [1,000; 1,000] |
| `s5_mobilenet_v3_large_v002_c` | iNaturalist CO | 2364 | 10 | 0,777 | [0,734; 0,812] | 0,929 | [0,918; 0,939] |
| `s5_mobilenet_v3_large_v002_c` | Mushroom Observer CO | 984 | 10 | 0,714 | [0,674; 0,747] | 0,774 | [0,748; 0,800] |
| `s5_mobilenet_v3_large_v002_c` | **Orinoquía** | 309 | 8 | 0,721 | [0,636; 0,786] | 0,770 | [0,722; 0,816] |
| `s5_mobilenet_v3_large_v002_c` | Otra región | 3029 | 10 | 0,772 | [0,744; 0,796] | 0,895 | [0,883; 0,906] |
| `s5_mobilenet_v3_large_v002_c` | Sin departamento | 10 | 4 | 0,750 | [0,250; 0,750] | 0,900 | [0,700; 1,000] |

## Comparación pareada

Diferencia de macro-F1 de observación respecto de `ma_resnet18_v002`, con los mismos remuestreos de observaciones para los dos modelos (2.000, semilla 0).

| Modelo | Grupo | Diferencia | IC95 pareado |
|---|---|---:|---|
| `s5_mobilenet_v3_large_v002_c` | Colombia (10 clases) | +0,046 | [+0,027; +0,067] |
| `s5_mobilenet_v3_large_v002_c` | **Orinoquía** (clases presentes) | +0,075 | [+0,015; +0,133] |

## Orinoquía

Observaciones con departamento Meta, Casanare, Arauca o Vichada (mismo criterio que la descarga GBIF `0007586-260221153910048`). Lista completa con enlace a cada registro y la predicción de cada modelo en [`orinoquia_predictions.csv`](orinoquia_predictions.csv).

### Por género (nivel de observación)

| Género | Soporte | Recall `ma_resnet18_v002` | F1 `ma_resnet18_v002` | Recall `s5_mobilenet_v3_large_v002_c` | F1 `s5_mobilenet_v3_large_v002_c` |
|---|---:|---:|---:|---:|---:|
| Amanita | 3 | 1,000 | 0,462 | 1,000 | 0,400 |
| Auricularia | 73 | 0,493 | 0,526 | 0,452 | 0,574 |
| Ganoderma | 24 | 0,875 | 0,712 | 0,875 | 0,737 |
| Geastrum | 19 | 0,947 | 0,947 | 0,947 | 0,973 |
| Lycoperdon | 6 | 0,667 | 0,258 | 0,667 | 0,615 |
| Pleurotus | 10 | 1,000 | 0,645 | 1,000 | 0,714 |
| Trametes | 143 | 0,622 | 0,718 | 0,853 | 0,836 |
| Xylaria | 31 | 0,839 | 0,897 | 0,871 | 0,915 |

Sin observaciones en la Orinoquía: Hericium, Morchella.

### Por departamento y fuente (exactitud de observación)

| Departamento | Fuente | Obs. | `ma_resnet18_v002` | `s5_mobilenet_v3_large_v002_c` |
|---|---|---:|---:|---:|
| Arauca | iNaturalist CO | 2 | 0,000 | 1,000 |
| Casanare | iNaturalist CO | 78 | 0,577 | 0,667 |
| Casanare | Mushroom Observer CO | 127 | 0,724 | 0,772 |
| Meta | iNaturalist CO | 90 | 0,689 | 0,822 |
| Meta | Mushroom Observer CO | 6 | 0,833 | 1,000 |
| Vichada | iNaturalist CO | 6 | 0,500 | 1,000 |

## Lectura y límites

* Las etiquetas externas son identificaciones de comunidad (iNaturalist grado de investigación; consenso de votos de Mushroom Observer), no curaduría experta como FungiTastic: la caída frente a FungiTastic mezcla cambio de dominio con ruido de etiqueta.
* La Orinoquía tiene 8 de los 10 géneros y Trametes concentra cerca de la mitad de sus observaciones; las clases con pocas observaciones tienen intervalos amplios.
* En la Orinoquía, iNaturalist CO: 176 observaciones en 152 sitio-días, 41 comparten sitio-día; Mushroom Observer CO: 133 observaciones en 71 sitio-días, 90 comparten sitio-día (sitio-día de `v001`: misma fecha y celda de ~1 km). El bootstrap remuestrea observaciones, no sitio-días, así que los IC95 de ese grupo pueden ser optimistas.
* Una sola semilla por modelo; las diferencias entre modelos cuyos IC95 se solapan no son concluyentes.
