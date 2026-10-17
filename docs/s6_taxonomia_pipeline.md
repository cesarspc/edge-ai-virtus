# S6: métricas por nivel taxonómico y modularización del pipeline

Estado: implementado el 17 de octubre de 2026. Alcance: S6 (Fase 3, cierre de la línea base visual).
Dataset `v002` (huella `144323e1…`). No se entrenó ni se evaluó ningún modelo nuevo: todo sale de predicciones ya guardadas.

## 1. Taxonomía

`scripts/data/build_taxonomy.py` construye el mapeo género → familia → orden → clase → filo a partir de
`data/raw/fungitastic/metadata.zip`, con el SHA-256 ya registrado en `data/manifests/sources/fungitastic.yaml`,
y de `data/versions/v002/splits.csv`. Escribe en `data/taxonomy/v002/` y deja `v002` intacto.

| Archivo | Contenido |
|---|---|
| `genus_taxonomy.csv` | Una fila por clase: linaje, número de especies y observaciones por partición |
| `observation_species.csv` | Especie de cada una de las 25.290 observaciones (análisis de errores bajo género) |
| `manifest.json` | Hashes de entradas y salidas, reglas y controles |

Controles: las 25.290 observaciones tienen taxonomía completa, ninguna mezcla especies entre sus imágenes, cada
género tiene un único linaje y no aparecen sinónimos de género. Los rangos superiores se toman tal como los
publica FungiTastic, sin ediciones manuales. Niveles resultantes: 10 géneros, 9 familias (Ganoderma y Trametes
son Polyporaceae), 7 órdenes, 3 clases y 2 filos (Morchella y Xylaria son Ascomycota; los otros ocho, Basidiomycota).

```powershell
python scripts/data/build_taxonomy.py           # construye
python scripts/data/build_taxonomy.py --check   # reconstruye y compara hashes
```

## 2. Métricas por nivel

`scripts/train/taxonomic_metrics.py` lee `runs/*/predictions_<split>.csv` de las corridas completas. No necesita
PyTorch, imágenes ni checkpoints. Antes de reportar exige que cada evaluación:

1. use la huella del dataset de la taxonomía;
2. cubra todas las observaciones de su partición;
3. reproduzca a nivel de género las métricas que la corrida guardó (matriz de confusión idéntica).

Reglas:

- **Regla principal:** el taxón predicho es el del género top-1. Es la decisión real del modelo juzgada a un
  nivel superior, y con ella un acierto de género implica acierto en todos los niveles superiores.
- **Regla secundaria (columnas `marginal_*`):** sumar las probabilidades de los géneros de cada taxón. Se reporta
  solo como comparación, porque con *label smoothing* cada género conserva un piso de probabilidad y la suma
  favorece a los taxones con muchos géneros (Agaricomycetes agrupa 7 de los 10).
- **Por observación:** primero se promedian las probabilidades de sus imágenes (`mean_prob`, igual que en el
  entrenamiento) y después se aplica la regla. IC95 por bootstrap percentil sobre observaciones (1.000
  remuestreos, semilla 42).

Resultados principales (macro-F1 por observación; IC95 entre corchetes):

| Modelo | Partición | Género | Familia | Orden | Clase | Filo |
|---|---|---:|---:|---:|---:|---:|
| ResNet-18 | val | 0,9545 [0,945–0,962] | 0,9588 | 0,9623 | 0,9870 | 0,9810 |
| ResNet-18 | test | 0,9463 [0,933–0,957] | 0,9495 | 0,9521 | 0,9611 | 0,9681 |
| MobileNetV3-Large A | val | 0,9706 [0,956–0,981] | 0,9721 | 0,9705 | 0,9788 | 0,9916 |

Tablas completas (también por imagen), F1 por taxón y desglose por especie: [`reports/s6/README.md`](../reports/s6/README.md).

Lectura:

- **Familia y orden suben poco.** En ResNet-18 (val), de 140 errores de género solo 24 quedan dentro de la misma
  familia y 51 dentro del mismo orden; los demás cruzan órdenes. Los que se recuperan son los pares más
  frecuentes: Ganoderma ↔ Trametes (Polyporaceae) y Amanita → Lycoperdon (Agaricales).
- **Macro-F1 no crece de forma monótona al subir de nivel**, aunque la exactitud sí. En clase y filo, el taxón
  raro (Pezizomycetes, que es solo Morchella: 11 observaciones en val y 20 en test) pesa lo mismo que
  Agaricomycetes, que tiene más de 3.000. Su IC95 es ancho; no conviene interpretar diferencias a ese nivel.
- **La separación Ascomycota/Basidiomycota es casi perfecta** (exactitud ≥ 0,99). Quedan entre 11 y 37
  observaciones mal clasificadas de filo según el modelo y la partición, en ambos sentidos (por ejemplo,
  Auricularia → Xylaria).
- **Especies difíciles en ambos modelos** (recall de género < 0,90): *Auricularia mesenterica*, *Hericium
  cirrhatum* y *Trametes cinnabarina*. Tienen pocas observaciones; son candidatas para revisar en la Fase 5.
- **ResNet-18 contra MobileNet solo se puede comparar en val**, porque MobileNet todavía no tiene test (S5
  abierto). La tabla se completa al ejecutar `evaluate_s5_final.py`: el script detecta `runs/<id>/final_test/`
  sin cambios.

```powershell
python scripts/train/taxonomic_metrics.py           # regenera reports/s6/
python scripts/train/taxonomic_metrics.py --check   # recalcula y compara byte a byte (exit 1 si cambia)
```

## 3. Pipeline modular

`scripts/train/train_ma.py` queda como orquestador: argumentos, salvaguardas de test y carpeta de corrida. Las
piezas están en `scripts/train/pipeline/`:

| Módulo | Responsabilidad |
|---|---|
| `data.py` (existente) | Carga de la versión y verificación SHA-256 contra el manifiesto |
| `pipeline/datasets.py` | `ImageTable` (Dataset de PyTorch sobre las tablas por partición) |
| `pipeline/transforms.py` | Transformaciones de entrenamiento y evaluación desde `input` |
| `pipeline/models.py` | Fábrica de arquitecturas (pesos preentrenados y cabeza) |
| `pipeline/training.py` | Semillas, ponderación de clases, optimizador/scheduler, una época, checkpoints y reanudación |
| `pipeline/evaluation.py` | Inferencia, métricas por imagen y observación, CSV de predicciones |
| `pipeline/artifacts.py` | Procedencia (`run.json`), `progress.json` y `RESULTS.md` |
| `pipeline/predictions.py`, `pipeline/taxonomy.py` | Lectura de predicciones y métricas taxonómicas (sin PyTorch) |

Garantías:

- **Misma receta S5.** No cambia ningún hiperparámetro, transformación, regla de selección ni formato de
  checkpoint. `train_ma` sigue exportando `ImageTable`, `build_model`, `build_transforms`, `predict`, `evaluate` y
  `save_predictions`, que usan `evaluate_s5_final.py`, `profile_s5.py` y los tests.
- **Regresión.** Con una configuración S5 C (ponderación `inverse_sqrt`) sobre datos sintéticos, el código nuevo
  reproduce byte a byte las predicciones, métricas, `history.csv` (sin la columna de segundos) y
  `class_balance.json` del código original, tanto en smoke como en una corrida pausada tras la época 1 y
  reanudada. Los `best.pt` y `last.pt` versionados de ResNet-18 y MobileNet A cargan en el código nuevo
  (`strict=True`) y siguen siendo reanudables.
- **Procedencia.** `run.json["code_sha256"]` ahora incluye los módulos de `pipeline/`. Las corridas nuevas no
  comparan ese campo con las antiguas.

Para la Fase 5 (multimodal), el punto de extensión es `pipeline/datasets.py`, que entregaría `(imagen,
metadatos)`, junto con `pipeline/models.py`, que añadiría la rama tabular. El resto del pipeline (entrenamiento,
selección por val, evaluación por observación y métricas taxonómicas) no depende del tipo de entrada.

## 4. Tests

`tests/test_s6_taxonomy_pipeline.py` comprueba:

- que el nivel de género es idéntico a las métricas de entrenamiento y que la exactitud nunca baja al subir de nivel;
- la regla marginal, el macro-F1 propio contra sklearn y el bootstrap determinista;
- que la taxonomía cubre v002;
- que las predicciones guardadas reproducen sus métricas y que `reports/s6/` se reproduce byte a byte;
- que los checkpoints guardados cargan y reanudan.

```powershell
python -m pytest tests -q
```

## 5. Limitaciones y pendientes

- **S5 sigue abierto.** Faltan los ensayos B, C y D y la evaluación final en test de MobileNet. Al cerrarlo,
  basta con volver a ejecutar `taxonomic_metrics.py`.
- **Una sola semilla por modelo.** Los IC95 solo cubren la variabilidad del muestreo de observaciones, no la del
  entrenamiento.
- **Las métricas por nivel son jerárquicas solo a posteriori.** El modelo no se entrenó con pérdida jerárquica;
  podría estudiarse en la Fase 5.
- **Finales de línea.** Con `core.autocrlf=true` (Windows), Git entrega `classes.csv` e `image_integrity.csv` con
  CRLF, y `data.py` los compara byte a byte con el manifiesto (publicado con LF), así que el entrenamiento se
  detendría con "does not match the published dataset". Los scripts de S6 normalizan los finales de línea. Para
  entrenar en esa máquina, añadir `data/versions/** -text` a `.gitattributes` o ejecutar
  `git config core.autocrlf input` y volver a hacer *checkout*. Esto no se ha cambiado.
