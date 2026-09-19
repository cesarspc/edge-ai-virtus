# edge-ai-virtus

Investigación universitaria sobre la **identificación multimodal de macromicetos mediante inteligencia artificial**:
imágenes, geolocalización, variables climáticas, sustrato y otros metadatos. En fases posteriores se evaluarán modelos
visuales y multimodales y su ejecución en dispositivos móviles.

La investigación se organiza por semanas. Estado actual:

| Semana | Fase | Estado | Entregables |
|---|---|---|---|
| 2 | Auditoría y curaduría (inventario de datasets, faltantes, duplicados, desbalance) | Hecha (sin transformar datos) | [`reports/week_02_dataset_audit.md`](reports/week_02_dataset_audit.md) |
| 3 en adelante | Curaduría efectiva, particiones, modelos, multimodalidad, móvil | Pendiente | — |

## Estructura del repositorio

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/                     # datos originales SOLO en local (ignorados por Git); README versionado
│   └── manifests/
│       ├── dataset_inventory.csv   # una fila por fuente: estado, cifras reportadas vs. verificadas, licencias
│       ├── raw_files.csv           # archivos locales con tamaño y SHA-256
│       ├── remote_queries.csv      # consultas remotas (URL, parámetros, fecha, hash de la respuesta)
│       └── sources/*.yaml          # un manifiesto de reproducibilidad por fuente
├── scripts/data/
│   ├── acquire_datasets.py      # adquisición metadata-first (idempotente)
│   ├── audit_datasets.py        # auditoría: tamaño, faltantes, duplicados, desbalance, inventario
│   ├── adapters.py              # un adaptador por fuente (lectura de solo lectura)
│   ├── audit_core.py            # métricas comunes a todos los datasets
│   ├── inventory.py             # construcción del inventario y de los manifiestos YAML
│   └── common.py                # rutas, HTTP, hashes y manifiestos
├── notebooks/01_dataset_audit.ipynb   # vista reproducible de la auditoría
└── reports/
    ├── week_02_dataset_audit.md       # informe de la Semana 2
    ├── audit/                         # resultados cuantitativos (CSV/JSON pequeños)
    └── figures/                       # figuras del informe
```

## Instalación

Requiere Python 3.12 (probado en Windows 11).

```bash
pip install -r requirements.txt
```

`curl` debe estar disponible en el `PATH` (solo lo usa la adquisición de MIND.Funga, ver el informe).
No se instalan frameworks de aprendizaje profundo en esta fase.

## Dónde están los datos y cómo obtenerlos

Los datos originales viven en `data/raw/` y **no se versionan** (tamaño y licencias). Ver
[`data/raw/README.md`](data/raw/README.md). La Semana 2 usa una estrategia *metadata-first*: se descargan metadatos, índices,
taxonomías y respuestas de APIs (~0,7 GB en total), **no imágenes**.

```bash
python scripts/data/acquire_datasets.py              # todas las fuentes
python scripts/data/acquire_datasets.py df20 gbif    # solo algunas
```

El script no sobrescribe archivos existentes y registra cada archivo y consulta en `data/manifests/`.

## Ejecutar la auditoría

```bash
python scripts/data/audit_datasets.py        # ~3 min; regenera reports/audit/, el inventario y los manifiestos YAML
```

Para ejecutar el notebook de forma técnica:

```bash
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/01_dataset_audit.ipynb
```

## Dónde consultar los resultados

* Informe: [`reports/week_02_dataset_audit.md`](reports/week_02_dataset_audit.md)
* Inventario de fuentes: [`data/manifests/dataset_inventory.csv`](data/manifests/dataset_inventory.csv)
* Tablas cuantitativas: [`reports/audit/`](reports/audit/) (`size_summary.csv`, `missing_values.csv`, `duplicates.csv`,
  `overlaps.csv`, `imbalance_summary.csv`, `class_distribution.csv`, `metadata_matrix.csv`, `remote_counts.csv`, `audit_facts.json`)

## Convenciones

* Toda la documentación está en español; el código y los nombres de archivo/columna, en inglés.
* Las cifras se etiquetan como `SOURCE_REPORTED` (declaradas por la fuente), `LOCAL_VERIFIED` (calculadas sobre archivos
  locales), `DERIVED` (deducidas en memoria) o `NOT_VERIFIED` (no comprobadas).
* La Semana 2 no modifica datos: no elimina, corrige, imputa, deduplica ni particiona.

## Nota sobre OneDrive

Si el repositorio está en una carpeta sincronizada con OneDrive, `data/raw/` (~0,7 GB tras la Semana 2) también se
sincronizará. Conviene excluirla de la sincronización o mover el repositorio antes de descargar imágenes.
