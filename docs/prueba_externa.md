# Validación externa: `test_external` completo de `v001`, con Colombia y la Orinoquía por separado

Actualizado: 29 de octubre de 2026. Modelo evaluado: **MobileNetV3-Large S5, configuración C**
(`runs/20261024T031650621510Z_s5_mobilenet_v3_large_v002_c`), el seleccionado en S5 por macro-F1 de `val`
(`reports/s5/selected_model.json`).

El modelo se entrenó solo con FungiTastic (`v002`, Dinamarca). Esta prueba mide su rendimiento con fotografías de
otras fuentes de las mismas 10 clases. Es una medición de cambio de dominio: no se usó ni se usará para seleccionar
modelos o hiperparámetros.

| Resultado | Informe |
|---|---|
| Conjunto completo (Colombia + MIND.Funga) | [`reports/external/ext_full_001/README.md`](../reports/external/ext_full_001/README.md) |
| Solo Colombia, con la Orinoquía | [`reports/external/ext_co_001/README.md`](../reports/external/ext_co_001/README.md) |

## 1. Los conjuntos

Los dos son vistas derivadas del `test_external` de `v001` (Semana 3). Ese `test_external` se construyó y validó
entonces, pero nunca se había evaluado porque sus imágenes no estaban descargadas. Ninguno de los dos cambia
etiquetas, imágenes ni filtros de `v001`.

| Fuente | Observaciones | Imágenes | Etiqueta | Ubicación | `ext_co_001` | `ext_full_001` |
|---|---:|---:|---|---|:---:|:---:|
| iNaturalist CO (grado de investigación) | 2.364 | 5.021 | comunidad | departamento | ✓ | ✓ |
| Mushroom Observer CO (consenso ≥ 1) | 984 | 3.081 | consenso de votos | departamento | ✓ | ✓ |
| MIND.Funga v2 (DOI 10.17632/sfrbdjvxcc.2) | 1.547 | 1.547 | nombre de carpeta | ninguna | | ✓ |
| **Total** | | | | | **3.348 / 8.102** | **4.895 / 9.649** |

* **`ext_full_001`** es exactamente el `test_external` de `v001`: el script comprueba que contiene las mismas
  4.895 observaciones y 9.649 imágenes, ni una más ni una menos. Su parte colombiana se copia sin cambios de
  `ext_co_001`.
* MIND.Funga v2 no tiene coordenadas, fecha ni país. Mezcla imágenes neotropicales de varios orígenes y tiene una
  imagen por observación, así que en todos los resultados va en su propio grupo (`region = no_location`).

Se descargaron las 9.649 imágenes, sin fallos:

* iNaturalist y Mushroom Observer desde la URL de cada imagen.
* MIND.Funga extrayendo solo las 1.547 imágenes necesarias del ZIP oficial de Mendeley (3,2 GB) con 38 peticiones
  HTTP Range (306 MB). Cada imagen se verificó contra el CRC-32 del ZIP y contra el SHA-256 que publica la fuente.

Los hashes de las imágenes están congelados en `data/versions/<conjunto>/image_integrity.csv`, y la evaluación
verifica cada archivo antes de usarlo.

## 2. Cómo se identifica la Orinoquía

**Orinoquía = departamentos Meta, Casanare, Arauca y Vichada.** Es el mismo criterio que la descarga GBIF
`0007586-260221153910048` entregada por el equipo del proyecto (`StateProvince is one of (Meta, Casanare, Arauca,
Vichada)`).

| Fuente | Cómo se obtiene el departamento |
|---|---|
| iNaturalist | Lugar de nivel administrativo 10 (departamento) dentro de Colombia entre los `place_ids` de la observación, que iNaturalist calcula a partir de las coordenadas (instantáneas `data/raw/inaturalist_api/api/ext_co_001_places_p*.json`). |
| Mushroom Observer | Departamento nombrado en la ubicación MO, por ejemplo «Yopal, Casanare, Colombia». |
| MIND.Funga | No aplica: sin ubicación. |

Resultado: **309 observaciones (857 imágenes) en la Orinoquía**, 3.029 en otros departamentos y 10 colombianas sin
departamento (ubicaciones MO como «Páramo de Chingaza, Colombia»).

| Departamento | iNaturalist | Mushroom Observer | Total |
|---|---:|---:|---:|
| Casanare | 78 | 127 | 205 |
| Meta | 90 | 6 | 96 |
| Vichada | 6 | 0 | 6 |
| Arauca | 2 | 0 | 2 |
| **Total** | **176** | **133** | **309** |

Dónde quedan identificadas:

* [`data/versions/ext_co_001/orinoquia_observations.csv`](../data/versions/ext_co_001/orinoquia_observations.csv):
  las 309, con enlace al registro original, departamento, fecha, coordenadas, género y `gbif_id` si también están
  en la descarga GBIF.
* `reports/external/<conjunto>/orinoquia_predictions.csv`: la predicción del modelo para cada una.
* En todos los informes aparecen como el grupo **Orinoquía** (`region = orinoquia` en los CSV), con tablas por
  género y por departamento.

### Relación con la descarga GBIF de la Orinoquía

La descarga GBIF (`data/raw/gbif_orinoquia/`) sirvió como verificación cruzada, no como fuente de datos:

* 63 de sus registros coinciden por `occurrenceID` con observaciones de iNaturalist del conjunto, y las 63 quedaron
  en la Orinoquía con el método de arriba.
* Ninguno de sus otros 15 registros con imagen en las 10 clases es utilizable
  ([`gbif_orinoquia_exclusions.csv`](../data/versions/ext_co_001/gbif_orinoquia_exclusions.csv)):
  * 5 de iNaturalist están hoy en grado `needs_id`.
  * 5 son *Hexagonia papyracea*, que GBIF ubica en *Trametes* pero la taxonomía de iNaturalist que usa `v001` no.
  * 5 del Caño Siete Machos tienen imágenes CC-BY-NC-ND, licencia que el proyecto excluye.
* La descarga no contiene registros de Mushroom Observer, aunque el 28-10-2026 la API de GBIF devolvía 172 con el
  mismo filtro de departamentos. No se determinó la causa. Por eso la Orinoquía se identifica por departamento en
  cada fuente: así salen 309 observaciones en lugar de 78.

Las capas 2–4 de la base de datos del proyecto (`data/raw/orinoquia_capas_bd/`) no intervienen en esta prueba.

## 3. Resultados (MobileNet S5-C)

Nivel de observación (`mean_prob`). IC95 por bootstrap de 2.000 remuestreos de observaciones. Macro-F1 sobre las
clases presentes en cada grupo (en el conjunto completo y en Colombia están las 10).

| Grupo | Obs. | Clases | Macro-F1 | IC95 | Exactitud | IC95 |
|---|---:|---:|---:|---|---:|---|
| FungiTastic `val` (referencia, en dominio) | 3.452 | 10 | 0,972 | | 0,977 | |
| **Conjunto externo completo** | 4.895 | 10 | **0,766** | [0,747; 0,786] | **0,858** | [0,848; 0,868] |
| **Colombia** (iNaturalist + MO) | 3.348 | 10 | **0,772** | [0,746; 0,797] | **0,883** | [0,873; 0,894] |
| iNaturalist CO | 2.364 | 10 | 0,777 | [0,734; 0,812] | 0,929 | [0,918; 0,939] |
| Mushroom Observer CO | 984 | 10 | 0,714 | [0,674; 0,747] | 0,774 | [0,748; 0,800] |
| **MIND.Funga v2** | 1.547 | 8 | 0,763 | [0,737; 0,787] | 0,803 | [0,784; 0,823] |
| **Orinoquía** | 309 | 8 | **0,721** | [0,636; 0,786] | **0,770** | [0,722; 0,816] |
| Colombia fuera de la Orinoquía | 3.029 | 10 | 0,772 | [0,744; 0,796] | 0,895 | [0,883; 0,906] |

Lectura:

1. **El cambio de dominio es grande.** El macro-F1 de observación cae de 0,972 en FungiTastic a 0,766 en el conjunto
   externo completo, y la exactitud de 0,977 a 0,858.
2. **Colombia y MIND.Funga rinden parecido en macro-F1 y distinto en exactitud.** Comparando sobre las clases
   presentes, el macro-F1 es 0,772 en Colombia y 0,763 en MIND.Funga, pero la exactitud es 0,883 y 0,803. Colombia
   tiene mucho *Amanita*, que el modelo reconoce bien (F1 0,977).
3. **La Orinoquía es la parte más difícil.** Su exactitud es 0,770, frente a 0,895 en el resto de Colombia, sin
   solapamiento de los IC95.
4. **El error dominante en todas las fuentes es *Auricularia* predicha como *Trametes*.** En la Orinoquía,
   103 de las 143 *Trametes* son *Trametes sanguinea* (la antigua *Pycnoporus sanguineus*, roja y tropical) y las
   *Auricularia* son especies tropicales (*A. brasiliana*, *A. delicata*, *A. fuscosuccinea*, *A. nigricans*).
   * El modelo confunde con *Trametes* 23 de las 73 *Auricularia* de la Orinoquía y 52 de las 340 de MIND.Funga.
   * En MIND.Funga, *Auricularia* también se confunde con *Pleurotus* (38).
   * *Lycoperdon* es la clase más débil de MIND.Funga (F1 0,390, 19 observaciones).
5. **Las clases ausentes también generan errores.** MIND.Funga no tiene *Hericium* ni *Morchella*, pero el modelo
   predice esos géneros en 16 de sus observaciones. Por eso su macro-F1 sobre las 10 clases (0,610) es menor que
   sobre las 8 presentes.

F1 por género y grupo, por fuente y por departamento:
[`reports/external/ext_full_001/README.md`](../reports/external/ext_full_001/README.md).

**Consistencia:** las predicciones del modelo para las 8.102 imágenes colombianas son idénticas en las dos
evaluaciones (`ext_co_001` y `ext_full_001`), así que las cifras de Colombia y de la Orinoquía coinciden en ambos
informes.

### Informe visual con ejemplos

```powershell
python scripts/reports/make_external_gallery.py     # ~1 min; no vuelve a ejecutar el modelo
```

El script genera `reports/external/ext_full_001/galeria/galeria_validacion_externa.pdf`: 16 páginas en formato
16:9, con cada página también como PNG para diapositivas. Contiene:

* resumen de métricas;
* matrices de confusión del conjunto completo y de la Orinoquía;
* aciertos por género;
* las 6 confusiones más frecuentes;
* los 10 errores con más confianza;
* aciertos y errores de la Orinoquía;
* créditos de las 110 fotos.

Los ejemplos se eligen al azar con semilla fija dentro de cada categoría, repartidos entre fuentes o géneros. Solo
la página de errores con más confianza sigue una regla: mayor probabilidad predicha, un caso por nombre de la fuente.

La carpeta está fuera de Git porque contiene fotos de terceros. Su licencia es CC-BY-NC o CC-BY-SA en la mayoría,
así que el PDF sirve para uso académico interno con los créditos incluidos.

Al revisar la galería apareció un hallazgo:

* MIND.Funga tiene 11 pares de fotos casi idénticas, por ejemplo `.bak.jpg` junto al original.
* En la parte colombiana, 34 fotos aparecen dos veces (22 de ellas en observaciones distintas).

Esto infla un poco el peso de esas imágenes, pero ningún par tiene etiquetas distintas.

## 4. Límites

* **Etiquetas de comunidad o de carpeta.** Ninguna fuente externa tiene la curaduría experta de FungiTastic, así que
  parte de la caída puede ser ruido de etiqueta. En MIND.Funga la etiqueta es el nombre de la carpeta.
* **Cobertura de clases.** *Hericium* tiene 6 observaciones externas, todas colombianas y fuera de la Orinoquía.
  *Hericium* y *Morchella* faltan en la Orinoquía y en MIND.Funga. En la Orinoquía, *Amanita* (3), *Lycoperdon* (6)
  y *Pleurotus* (10) tienen muy pocas observaciones.
* **Dependencia entre observaciones.** En la Orinoquía, las 133 observaciones de Mushroom Observer caen en 71
  sitio-días (90 comparten uno) y las 176 de iNaturalist en 152. El bootstrap remuestrea observaciones, así que los
  IC95 de ese grupo pueden ser optimistas.
* **Unidad distinta en MIND.Funga.** Tiene una imagen por observación, mientras que las observaciones colombianas
  promedian 2,4 imágenes. La agregación `mean_prob` favorece a las observaciones con más imágenes.
* **Taxonomía.** El conjunto sigue la taxonomía de iNaturalist con que se construyó `v001` (*Trametes sanguinea*
  dentro de *Trametes*, *Hexagonia* fuera).
* **Una evaluación por checkpoint y conjunto.** `reports/external/<conjunto>/<run_id>/` se crea de forma exclusiva y
  bloquea repeticiones. En S5 solo se admite el modelo seleccionado. La referencia en dominio es `val`, porque el
  `test` final de S5 aún no se ha ejecutado (`scripts/train/evaluate_s5_final.py`).
* **ResNet-18.** El baseline ResNet-18 también se evaluó en `ext_co_001` antes de decidir usar solo S5-C. Ese
  registro sigue en `reports/external/ext_co_001/` (macro-F1 Colombia 0,726, Orinoquía 0,645), pero no se evaluó en el
  conjunto completo ni forma parte de este análisis.

## 5. Reproducir

```powershell
python scripts/data/build_external_set.py --check     # ext_co_001 coincide con el publicado
python scripts/data/build_external_full.py --check    # ext_full_001 coincide con el publicado
python scripts/data/download_external_images.py --set ext_full_001   # ~1,3 GB; reanudable y verificado
python scripts/train/evaluate_external.py --set ext_full_001 --summary-only
python scripts/train/evaluate_external.py --set ext_co_001 --summary-only
```

Las imágenes se guardan en `data/raw/images/external_v001/` (compartido por los dos conjuntos, ignorado por Git).
La evaluación tardó unos 3 minutos en CPU (este equipo) para las 9.649 imágenes.

Nota de procedencia: las evaluaciones de `ext_co_001` se ejecutaron antes de añadir el parámetro `--set` y la ruta
compartida de imágenes. Por eso el `code_sha256` de `evaluate_external.py` guardado en sus `evaluation.json` no
coincide con el archivo actual. Las predicciones guardadas no cambiaron y coinciden con las de `ext_full_001`.
