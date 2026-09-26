# Ficha del dataset `v001`

> Archivo generado por `scripts/data/build_dataset.py`; no editar a mano.

* **Huella de contenido:** `6704c19c0fb8b7f92932c7a586b10d8e34449f0f13c211d1e1a08eb154c3dec7`
* **Configuración:** `configs/dataset_v001.yaml` (SHA-256 `39f06c7d1f2039ca…`)
* **Esquema:** `data/schemas/metadata_schema_v1.yaml`
* **Clases (10, nivel genus):** 0=Amanita, 1=Auricularia, 2=Ganoderma, 3=Geastrum, 4=Hericium, 5=Lycoperdon, 6=Morchella, 7=Pleurotus, 8=Trametes, 9=Xylaria
* **Observaciones:** 30.269 · **Imágenes:** 53.518

## Observaciones por género y partición

| Género | train | val | test | test_external |
|---|---|---|---|---|
| Amanita | 4.209 | 858 | 952 | 1.549 |
| Auricularia | 1.437 | 304 | 311 | 578 |
| Ganoderma | 1.756 | 360 | 399 | 369 |
| Geastrum | 730 | 171 | 148 | 420 |
| Hericium | 428 | 100 | 83 | 6 |
| Lycoperdon | 2.120 | 426 | 442 | 130 |
| Morchella | 214 | 34 | 40 | 175 |
| Pleurotus | 1.398 | 287 | 358 | 364 |
| Trametes | 3.640 | 767 | 801 | 801 |
| Xylaria | 1.826 | 372 | 403 | 503 |

## Imágenes por género y partición

| Género | train | val | test | test_external |
|---|---|---|---|---|
| Amanita | 7.473 | 1.508 | 1.767 | 3.259 |
| Auricularia | 2.063 | 473 | 458 | 960 |
| Ganoderma | 3.451 | 739 | 775 | 829 |
| Geastrum | 1.365 | 301 | 260 | 564 |
| Hericium | 716 | 182 | 143 | 18 |
| Lycoperdon | 3.354 | 664 | 685 | 305 |
| Morchella | 337 | 50 | 74 | 440 |
| Pleurotus | 2.590 | 554 | 675 | 771 |
| Trametes | 6.654 | 1.367 | 1.429 | 1.518 |
| Xylaria | 2.660 | 527 | 575 | 985 |

## Fuentes

| Fuente | Papel | Observaciones | Imágenes |
|---|---|---|---|
| `fungitastic` | development | 25.374 | 43.869 |
| `mushroom_observer_co` | external_test | 984 | 3.081 |
| `inaturalist_co` | external_test | 2.364 | 5.021 |
| `mind_funga_v2` | external_test | 1.547 | 1.547 |

Fuentes excluidas: `df20` (96,5 % de sus observaciones ya están en FungiTastic y su partición original filtra observaciones (Semana 2, §7.3)); `gbif` (44,7 % de los registros colombianos con imagen provienen de iNaturalist y otros de Mushroom Observer (doble conteo)); `sib_colombia` (los tres recursos IPT no contienen fotografías); `inaturalist_open_data` (las tablas completas (13 y 20 GB) no se descargaron; se usa la API por género).

## Controles de fuga (todos deben ser 0)

* `groups_in_more_than_one_split`: 0
* `observations_with_images_in_more_than_one_split`: 0
* `images_in_more_than_one_split`: 0
* `observation_uid_duplicated`: 0
* `image_uid_duplicated`: 0

## Licencias de las imágenes

| Licencia | Alcance | Imágenes |
|---|---|---|
| CC-BY | image | 696 |
| CC-BY-NC | dataset | 1.547 |
| CC-BY-NC | image | 4.111 |
| CC-BY-NC-SA | dataset | 43.869 |
| CC-BY-NC-SA | image | 103 |
| CC-BY-SA | image | 3.137 |
| CC0 | image | 55 |

Más detalle: `cleaning_log.csv`, `split_summary.csv`, `env_coverage.csv`, `manifest.json`.
