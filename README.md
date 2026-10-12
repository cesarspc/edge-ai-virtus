# edge-ai-virtus

## Primera baseline con particiones oficiales (v002)

Ejecutar en PowerShell desde la raíz: `powershell -ExecutionPolicy Bypass -File scripts/run_baseline.ps1`.
Usa únicamente FungiTastic (`Train`, `ClosedSet-Val`, `ClosedSet-Test`), verifica hashes, protege test
al excluir copias exactas entre particiones y entrena ResNet-18. La prueba rápida no evalúa test.
Opciones y protocolo: [docs/baseline_v002.md](docs/baseline_v002.md). La documentación de v001 sigue abajo
como historial de las fases anteriores.

Investigación universitaria sobre la **identificación multimodal de macromicetos mediante inteligencia artificial**:
imágenes, geolocalización, variables climáticas, sustrato y otros metadatos. En fases posteriores se evaluarán modelos
visuales y multimodales y su ejecución en dispositivos móviles.

La investigación se organiza por semanas. Estado actual:

| Semana | Fase | Estado | Entregables |
|---|---|---|---|
| 2 | Auditoría y curaduría (inventario de datasets, faltantes, duplicados, desbalance) | Hecha (sin transformar datos) | [`reports/week_02_dataset_audit.md`](reports/week_02_dataset_audit.md) |
| 3 | Limpieza, normalización, particiones, esquema de metadatos ambientales y versionado (`v001`) | Hecha (sin imágenes descargadas) | [`reports/week_03_dataset_curation.md`](reports/week_03_dataset_curation.md), [`data/versions/v001/`](data/versions/v001/) |
| 4 | Entorno de entrenamiento y primer entrenamiento de referencia ($M_A$, ResNet-18, solo FungiTastic) | **Preparado, sin ejecutar** | [`docs/handoff_fase4_entrenamiento.md`](docs/handoff_fase4_entrenamiento.md) |
| 5 en adelante | Candidatas móviles, multimodalidad, prueba externa colombiana, móvil | Pendiente | — |

## Estructura del repositorio

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── requirements-train.txt       # dependencias de entrenamiento (PyTorch, torchvision, scikit-learn)
├── configs/
│   ├── dataset_v001.yaml        # todo lo que define el contenido de la versión v001
│   └── train/ma_resnet18_v001.yaml   # primer entrenamiento de referencia (M_A, ResNet-18)
├── docs/handoff_fase4_entrenamiento.md   # guía paso a paso para continuar
├── data/
│   ├── raw/                     # datos originales SOLO en local (ignorados por Git); README versionado
│   ├── processed/v001/          # tablas completas del dataset procesado (ignoradas por Git; reproducibles)
│   ├── versions/                # versionado: política, CHANGELOG y, por versión, manifiesto, particiones y ficha
│   ├── schemas/metadata_schema_v1.yaml   # esquema de metadatos (procedencia: fuente, método, calidad)
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
│   ├── common.py                # rutas, HTTP, hashes y manifiestos
│   ├── build_dataset.py         # Semana 3: construye, valida y versiona el dataset procesado
│   ├── curation_sources.py      # un cargador por fuente (reglas + metadatos ambientales)
│   ├── cleaning.py              # catálogo de reglas de limpieza/normalización y registro de su efecto
│   ├── splits.py                # particiones agrupadas por sitio-día con hash determinista
│   ├── metadata_schema.py       # carga y validación del esquema
│   └── download_images.py       # fase 4: descarga reanudable de las imágenes FungiTastic de v001
├── scripts/train/
│   ├── check_env.py             # comprueba bibliotecas, GPU, versión del dataset e imágenes
│   ├── train_ma.py              # entrena/evalúa M_A (solo imagen); resultados en runs/ (ignorado)
│   ├── data.py                  # tablas por partición + verificación contra el manifiesto de la versión
│   └── metrics.py               # macro-F1, exactitud balanceada, top-k, por clase, matriz de confusión
├── notebooks/01_dataset_audit.ipynb   # vista reproducible de la auditoría
└── reports/
    ├── week_02_dataset_audit.md       # informe de la Semana 2
    ├── week_03_dataset_curation.md    # informe de la Semana 3 (+ week_03_results.tex/.pdf)
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

## Construir el dataset procesado (Semana 3)

```bash
python scripts/data/acquire_datasets.py inaturalist_genera_co   # instantáneas de iNaturalist por género (Colombia)
python scripts/data/build_dataset.py                            # ~2 min; escribe data/processed/v001 y data/versions/v001
python scripts/data/build_dataset.py --check                    # reconstruye y compara con el manifiesto
```

Política de versiones: [`data/versions/README.md`](data/versions/README.md).

## Siguiente fase: imágenes y primer entrenamiento

Guía completa: [`docs/handoff_fase4_entrenamiento.md`](docs/handoff_fase4_entrenamiento.md). Resumen:

```bash
python scripts/data/download_images.py                                   # ~1,1 GB, reanudable (solo FungiTastic)
pip install -r requirements-train.txt                                    # tras instalar PyTorch según el hardware
python scripts/train/check_env.py --config configs/train/ma_resnet18_v001.yaml
python scripts/train/train_ma.py --config configs/train/ma_resnet18_v001.yaml --smoke
python scripts/train/train_ma.py --config configs/train/ma_resnet18_v001.yaml
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
* Desde la Semana 3 las transformaciones solo ocurren en `build_dataset.py`, nunca sobre `data/raw/`, y cada regla
  aplicada queda contada en `data/versions/<versión>/cleaning_log.csv`.

## Nota sobre OneDrive

Si el repositorio está en una carpeta sincronizada con OneDrive, `data/raw/` (~0,7 GB tras la Semana 2) también se
sincronizará. Conviene excluirla de la sincronización o mover el repositorio antes de descargar imágenes.
