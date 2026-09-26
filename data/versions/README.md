# Versionado del dataset procesado

Cada versión del dataset procesado (`v001`, `v002`, …) queda definida por **tres cosas versionadas en Git** y es
reproducible a partir de los datos crudos registrados en `data/manifests/raw_files.csv`:

| Qué | Dónde | Papel |
|---|---|---|
| Configuración | `configs/dataset_vXXX.yaml` | Clases, fuentes, reglas, umbrales, licencias, semilla (*salt*) y proporciones de partición. |
| Esquema | `data/schemas/metadata_schema_vN.yaml` | Columnas, vocabularios y procedencia (fuente, método, calidad) de los metadatos. |
| Código | `scripts/data/build_dataset.py` y módulos | Reglas de limpieza, normalización y partición. |

La ejecución produce:

* `data/processed/vXXX/observations.csv` e `images.csv` — tablas completas. **No se versionan** (≈ 27 MB, derivan de
  datos con licencias heterogéneas); se regeneran de forma determinista y su SHA-256 queda en el manifiesto.
* `data/versions/vXXX/` — **sí se versiona**:
  * `manifest.json`: huella de contenido, commit de Git, versiones de Python/pandas, SHA-256 de la configuración,
    del esquema, del código, de cada archivo crudo de entrada y de cada salida; conteos y controles de fuga.
  * `splits.csv`: partición de cada observación (`observation_uid → split`, `group_id`, `split_temporal`).
  * `classes.csv`, `cleaning_log.csv`, `split_summary.csv`, `env_coverage.csv`, `DATASET_CARD.md`.

## Reglas

1. **Inmutabilidad.** Una versión publicada no cambia. Si al reconstruir con la misma configuración la huella de
   contenido difiere, `build_dataset.py` se detiene: hay que crear una versión nueva.
2. **Cuándo se crea una versión nueva.** Cualquier cambio que altere el contenido: clases, fuentes, reglas o umbrales,
   esquema, código de limpieza/partición o nuevos datos crudos (p. ej. instantáneas de API más recientes).
   Cambios que no alteran el contenido (documentación, refactorización) no crean versión; `--check` lo demuestra.
3. **Numeración.** `vXXX` con tres dígitos, consecutiva. El esquema lleva su propia versión (`metadata_schema_vN`):
   añadir columnas o valores de vocabulario crea `vN+1`; una versión del dataset declara qué esquema usa.
4. **Estabilidad de particiones.** La asignación a train/val/test depende solo del `group_id` y de la *salt*, así que
   al añadir datos en una versión nueva los grupos existentes conservan su partición (si no cambian la *salt*,
   las proporciones ni la regla de agrupación). Cambiar cualquiera de ellas debe anotarse en el `CHANGELOG.md`.
5. **Publicación.** (a) commit de configuración, esquema y código; (b) `python scripts/data/build_dataset.py
   --rewrite-manifest` para que el manifiesto registre ese commit con el árbol limpio; (c) commit de
   `data/versions/vXXX/`; (d) etiqueta `git tag data-vXXX`.
6. **Verificación.** `python scripts/data/build_dataset.py --check` reconstruye en memoria y compara cada salida con
   el manifiesto (`IDENTICAL` / `DIFFERENT`). Los modelos y experimentos deben citar la versión y su huella de contenido.

Historial: [`CHANGELOG.md`](CHANGELOG.md).
