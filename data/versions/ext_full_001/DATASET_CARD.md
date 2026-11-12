# Conjunto de evaluación externa `ext_full_001` (test_external completo de `v001`)

> Generado por `scripts/data/build_external_full.py`; no editar a mano.

* **Huella de contenido:** `61ce96b14844c4b50bcc5aad7306ac6445f2b971596361a8990794e9935b2715`
* **Derivado de:** `v001` (`test_external` completo, huella `6704c19c0fb8b7f9…`), sin cambiar etiquetas, imágenes ni filtros.
* **Parte colombiana:** copiada sin cambios de [`ext_co_001`](../ext_co_001/DATASET_CARD.md) (departamento y región, incluida la Orinoquía).
* **Observaciones:** 4.895 · **Imágenes:** 9.649

| Fuente | Observaciones | Imágenes | Etiqueta | Región |
|---|---:|---:|---|---|
| `inaturalist_co` | 2.364 | 5.021 | comunidad, grado de investigación | departamento |
| `mushroom_observer_co` | 984 | 3.081 | consenso de votos ≥ 1 | departamento |
| `mind_funga_v2` | 1.547 | 1.547 | nombre de carpeta | `no_location` (sin coordenadas, fecha ni país) |

MIND.Funga v2 (https://doi.org/10.17632/sfrbdjvxcc.2): imágenes de macrohongos neotropicales de origen mixto; una imagen por observación. Cada imagen conserva el SHA-256 publicado por la fuente (`sha256_reported`).

## Observaciones por género y fuente

| Género | iNaturalist CO | Mushroom Observer CO | MIND.Funga | Total |
|---|---:|---:|---:|---:|
| Amanita | 1.332 | 113 | 104 | 1.549 |
| Auricularia | 129 | 109 | 340 | 578 |
| Ganoderma | 67 | 174 | 128 | 369 |
| Geastrum | 18 | 75 | 327 | 420 |
| Hericium | 5 | 1 | 0 | 6 |
| Lycoperdon | 51 | 60 | 19 | 130 |
| Morchella | 171 | 4 | 0 | 175 |
| Pleurotus | 16 | 203 | 145 | 364 |
| Trametes | 489 | 65 | 247 | 801 |
| Xylaria | 86 | 180 | 237 | 503 |

## Uso

Solo evaluación de cambio de dominio. Imágenes: `python scripts/data/download_external_images.py --set ext_full_001`. Evaluación: `python scripts/train/evaluate_external.py --set ext_full_001 --run runs/<run_id>`.
