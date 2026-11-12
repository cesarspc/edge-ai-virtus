# Conjunto de evaluación externa `ext_co_001` (Colombia)

> Generado por `scripts/data/build_external_set.py`; no editar a mano.

* **Huella de contenido:** `05c6a18e3217914b09fbb0fcca9be206a43a5e113e7691a83ed6612bfc636410`
* **Derivado de:** `v001` (`test_external`, huella `6704c19c0fb8b7f9…`), sin cambiar etiquetas, imágenes ni filtros.
* **Fuentes:** `inaturalist_co`, `mushroom_observer_co`. Excluida: `mind_funga_v2` (sin coordenadas, fecha ni país: imágenes neotropicales de origen mixto con etiqueta de carpeta; no es una prueba colombiana ni se puede ubicar por región).
* **Clases (10):** 0=Amanita, 1=Auricularia, 2=Ganoderma, 3=Geastrum, 4=Hericium, 5=Lycoperdon, 6=Morchella, 7=Pleurotus, 8=Trametes, 9=Xylaria
* **Observaciones:** 3.348 · **Imágenes:** 8.102

## Región

`orinoquia` = departamentos Arauca, Casanare, Meta, Vichada, el mismo criterio de la descarga GBIF `0007586-260221153910048`. `other` = otro departamento de Colombia. `unknown` = la ubicación no nombra un departamento.

* iNaturalist: lugar de nivel administrativo 10 dentro de Colombia entre los `place_ids` de la observación.
* Mushroom Observer: departamento nombrado en la ubicación MO (por ejemplo «Yopal, Casanare, Colombia»).

| Fuente | Orinoquía | Otra región | Sin departamento |
|---|---:|---:|---:|
| `inaturalist_co` | 176 | 2.188 | 0 |
| `mushroom_observer_co` | 133 | 841 | 10 |

### Observaciones por género y región

| Género | Orinoquía | Otra región | Sin departamento |
|---|---:|---:|---:|
| Amanita | 3 | 1.436 | 6 |
| Auricularia | 73 | 164 | 1 |
| Ganoderma | 24 | 215 | 2 |
| Geastrum | 19 | 74 | 0 |
| Hericium | 0 | 6 | 0 |
| Lycoperdon | 6 | 105 | 0 |
| Morchella | 0 | 175 | 0 |
| Pleurotus | 10 | 209 | 0 |
| Trametes | 143 | 410 | 1 |
| Xylaria | 31 | 235 | 0 |

## Observaciones de la Orinoquía

**309 observaciones** (857 imágenes). Lista completa, con enlace a cada registro, en [`orinoquia_observations.csv`](orinoquia_observations.csv).

| Departamento | iNaturalist | Mushroom Observer | Total |
|---|---:|---:|---:|
| Arauca | 2 | 0 | 2 |
| Casanare | 78 | 127 | 205 |
| Meta | 90 | 6 | 96 |
| Vichada | 6 | 0 | 6 |

De ellas, 63 aparecen también en la descarga GBIF `0007586-260221153910048` (columna `gbif_id`). Esa descarga (24-02-2026) no contiene ningún registro de Mushroom Observer, aunque el 28-10-2026 la API de GBIF devolvía 172 registros de MO en Colombia con el mismo filtro de departamentos; la causa no se determinó. Por eso la Orinoquía se identifica aquí por departamento en cada fuente y la descarga GBIF solo sirve de verificación cruzada.

## Registros de la descarga GBIF de la Orinoquía que no entran

La descarga tiene 15 registros con imagen en las 10 clases que no están en `v001`. Ninguno es utilizable; motivo por registro en [`gbif_orinoquia_exclusions.csv`](gbif_orinoquia_exclusions.csv):

* iNaturalist: grado 'needs_id' (v001 exige research grade) (5)
* iNaturalist la identifica en otro género; el árbol de GBIF la ubica en Trametes. v001 usa la taxonomía de iNaturalist (5)
* licencia de imagen CC-BY-NC-ND no admitida por allowed_licenses (se excluye ND) (5)

## Uso

Solo evaluación de cambio de dominio: no se usa para seleccionar modelos ni hiperparámetros. Las imágenes se descargan con `python scripts/data/download_external_images.py --set ext_co_001` y los modelos se evalúan con `python scripts/train/evaluate_external.py --set ext_co_001 --run runs/<run_id>`. El conjunto completo (Colombia + MIND.Funga) es [`ext_full_001`](../ext_full_001/DATASET_CARD.md).
