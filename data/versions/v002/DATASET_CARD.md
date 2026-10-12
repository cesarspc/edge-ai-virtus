# Ficha del dataset `v002`

> Archivo generado por `scripts/data/build_dataset.py`; no editar a mano.

* **Huella de contenido:** `144323e165ecfbe86d29a3484e56fd91b8e8cb1dce1eef5f1d1355fea7bedb90`
* **Configuración:** `configs/dataset_v002.yaml` (SHA-256 `68dfbb94ee76552a…`)
* **Esquema:** `data/schemas/metadata_schema_v1.yaml`
* **Clases (10, nivel genus):** 0=Amanita, 1=Auricularia, 2=Ganoderma, 3=Geastrum, 4=Hericium, 5=Lycoperdon, 6=Morchella, 7=Pleurotus, 8=Trametes, 9=Xylaria
* **Observaciones:** 25.290 · **Imágenes:** 43.621

## Observaciones por género y partición

| Género | train | val | test |
|---|---|---|---|
| Amanita | 3.752 | 904 | 1.315 |
| Auricularia | 1.443 | 288 | 319 |
| Ganoderma | 1.832 | 335 | 346 |
| Geastrum | 779 | 79 | 182 |
| Hericium | 494 | 66 | 51 |
| Lycoperdon | 2.166 | 408 | 407 |
| Morchella | 256 | 11 | 20 |
| Pleurotus | 1.465 | 251 | 319 |
| Trametes | 3.654 | 756 | 791 |
| Xylaria | 1.958 | 354 | 289 |

## Imágenes por género y partición

| Género | train | val | test |
|---|---|---|---|
| Amanita | 6.508 | 1.709 | 2.393 |
| Auricularia | 2.106 | 411 | 475 |
| Ganoderma | 3.567 | 738 | 653 |
| Geastrum | 1.350 | 157 | 377 |
| Hericium | 826 | 114 | 101 |
| Lycoperdon | 3.336 | 686 | 660 |
| Morchella | 405 | 21 | 34 |
| Pleurotus | 2.690 | 495 | 616 |
| Trametes | 6.626 | 1.421 | 1.384 |
| Xylaria | 2.758 | 538 | 466 |

## Fuentes

| Fuente | Papel | Observaciones | Imágenes |
|---|---|---|---|
| `fungitastic` | development | 25.290 | 43.621 |

Fuentes excluidas: `df20` (96,5 % de observaciones compartidas con FungiTastic; fuga entre sus particiones originales.); `fungitastic_open_mini_fewshot_dna` (OpenSet contiene ClosedSet; Mini/FewShot se solapan y DNA es un protocolo distinto.); `gbif` (Agregador de fuentes; riesgo de doble conteo.); `inaturalist` (Fuera de la primera baseline de una sola fuente.); `mushroom_observer` (Fuera de la primera baseline de una sola fuente.); `mind_funga_v2` (Fuera de la primera baseline de una sola fuente.); `sib_colombia` (Recursos auditados sin imágenes.).

## Controles de identidad y cruces de sitio-día

* `groups_in_more_than_one_split`: 0
* `observations_with_images_in_more_than_one_split`: 0
* `images_in_more_than_one_split`: 0
* `observation_uid_duplicated`: 0
* `image_uid_duplicated`: 0

Se preservan las particiones oficiales. Los cruces de sitio-día se reportan sin reasignar datos.

Imágenes excluidas por SHA-256 entre particiones: 0.
Pares perceptualmente similares entre particiones: 1; no se excluyen automáticamente.
Imágenes por debajo del umbral de tamaño: 33; conservadas y reportadas.
Ver `image_exclusions.csv`, `near_duplicates.csv` y `site_day_overlaps.csv`.

## Licencias de las imágenes

| Licencia | Alcance | Imágenes |
|---|---|---|
| CC-BY-NC-SA | dataset | 43.621 |

Más detalle: `cleaning_log.csv`, `split_summary.csv`, `env_coverage.csv`, `manifest.json`.
