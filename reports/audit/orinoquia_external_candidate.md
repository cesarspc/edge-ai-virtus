# Orinoquía (GBIF GBIF0007586-260221153910048) como candidata a `test_external`

> Generado por `scripts/data/audit_orinoquia_external.py`; fuente `data/raw/gbif_orinoquia/dwca`.

Descarga filtrada: **1.319 ocurrencias** y **3.063 registros multimedia** (158 géneros de macromicetos de la Orinoquía).
Intersección con las 10 clases de `configs/dataset_v002.yaml`: **158 ocurrencias**, de las cuales **78 tienen al menos una imagen con URL** y suman **247 imágenes**.

Clases con soporte > 0: **8/10**. Clases con menos de 10 observaciones: Amanita, Ganoderma, Geastrum, Lycoperdon, Pleurotus. Clases vacías: Hericium, Morchella.

## Soporte por clase

| `label_id` | Género | Ocurrencias | Observaciones con imagen | Imágenes |
|---:|---|---:|---:|---:|
| 0 | Amanita | 1 | 1 | 1 |
| 1 | Auricularia | 46 | 39 | 95 |
| 2 | Ganoderma | 6 | 4 | 12 |
| 3 | Geastrum | 5 | 4 | 13 |
| 4 | Hericium | 0 | 0 | 0 |
| 5 | Lycoperdon | 61 | 1 | 2 |
| 6 | Morchella | 0 | 0 | 0 |
| 7 | Pleurotus | 2 | 1 | 5 |
| 8 | Trametes | 20 | 17 | 79 |
| 9 | Xylaria | 17 | 11 | 40 |

## Procedencia, licencia y cobertura

* Rango de años: 2014–2025.
* Departamentos: Casanare (39); Meta (38); Vichada (1).
* Licencias declaradas en la ocurrencia: CC_BY_NC_4_0 (62); CC_BY_4_0 (16).
* Licencias de las imágenes (extensión multimedia, la que rige para usarlas): by-nc/4.0 (182); by/4.0 (60); by-nc-nd/4.0 (5).
* Datasets de origen en GBIF:
  * `50c9509d-22c7-4a22-a47d-8c48425ef4a7` (73 ocurrencias) — iNaturalist Research-grade Observations, **fuente ya auditada aparte en `v001`**
  * `239f2462-dbfb-4a47-a5f0-1b7b23b91221` (5 ocurrencias) — Biodiversidad Fúngica del Caño Siete Machos: valor ecoturístico en el área de influencia del PNN Sierra de la Macarena, Colombia

* Solapamiento con el `test_external` ya publicado en `v001` (iNaturalist CO + Mushroom Observer CO), por `occurrenceID`: **63 de 78** observaciones. Si se usan las dos fuentes hay que deduplicar.

Uso real en la prueba externa: [`data/versions/ext_co_001/`](../../data/versions/ext_co_001/DATASET_CARD.md) toma la Orinoquía de las fuentes de `v001` por departamento (más amplia que esta descarga) y lista en `gbif_orinoquia_exclusions.csv` por qué ninguno de los registros de esta descarga ausentes de `v001` es utilizable.

## Lectura

El conjunto es utilizable (coordenadas, fecha y licencia completas), pero su tamaño para estas 10 clases es de 78 observaciones, con 5 clases por debajo de 10 observaciones. No sostiene un macro-F1 de 10 clases; sirve como sonda regional con soporte por clase e intervalos de confianza, o como subgrupo dentro de una prueba externa colombiana más amplia.
