# `data/raw/` — datos originales (solo en local)


| Carpeta | Contenido | Origen |
|---|---|---|
| `fungitastic/` | `metadata.zip` (CSV de todos los subconjuntos) + instantáneas de la API/HEAD | servidor oficial `cmp.felk.cvut.cz`, ficha pública de Kaggle |
| `df20/` | `DF20-metadata.zip` (CSV de entrenamiento y prueba pública) | servidor oficial `ptak.felk.cvut.cz` |
| `mind_funga_v2/` | Instantánea JSON de la ficha de Mendeley Data (v2) y directorio central del ZIP oficial | API pública de Mendeley Data (se accede con `curl`; con `requests` Cloudflare responde 403) |
| `inaturalist_open_data/` | `taxa.csv.gz` completo y fragmentos iniciales (solo esquema) de `observations`/`photos` | bucket S3 público `inaturalist-open-data` |
| `inaturalist_api/` | Instantáneas JSON de consultas pequeñas (fungi en Colombia) | `api.inaturalist.org/v1` |
| `gbif/` | Instantáneas JSON de consultas a la API de ocurrencias y del registro | `api.gbif.org/v1` |
| `sib_*/` (3 carpetas) | Archivos Darwin Core (DwC-A) de los tres recursos IPT del SiB Colombia | `ipt.biodiversidad.co` |
| `mushroom_observer/` | Volcados CSV nocturnos oficiales | `mushroomobserver.org/*.csv` |
| `sources_documentation/` | Estado HTTP de las URL de documentación suministradas | GET de comprobación |

Cada archivo está registrado con su tamaño, SHA256, URL de origen y fecha en
[`data/manifests/raw_files.csv`](../manifests/raw_files.csv). Las consultas remotas (URL, parámetros,
fecha y hash de la respuesta) están en [`data/manifests/remote_queries.csv`](../manifests/remote_queries.csv),
y hay un manifiesto por fuente en [`data/manifests/sources/`](../manifests/sources/).

## Cómo obtenerlos (reproducir la adquisición)

Desde la raíz del repositorio:

```bash
pip install -r requirements.txt
python scripts/data/acquire_datasets.py            # todas las fuentes (~0,7 GB en total)
python scripts/data/acquire_datasets.py df20 gbif  # solo algunas fuentes
```

El script no vuelve a descargar ni sobrescribe archivos existentes y reutiliza consultas remotas guardadas.

Las imágenes no se descargan con este script.