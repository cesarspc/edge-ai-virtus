# S9: clima, revisión de errores y disponibilidad en Android

> Actualización del 13-11-2026: se ejecutó un ensayo S9/S10 con la parte visual congelada,
> ramas de contexto entrenadas en CPU y evaluación externa de Colombia/Orinoquía.
> Ver el [reporte de resultados](../reports/week_09_10/main.tex).
> Las referencias a entrenamiento pendiente más abajo corresponden al ajuste conjunto en GPU.

## Alcance y estado

C = imagen + GPS + clima. Sustrato se reserva para S10; hábitat, elevación y
fecha no se incorporan como entradas adicionales. Los tres trabajos de auditoría
solicitados acompañan la comparación incremental descrita en
[`s8_multimodalidad.md`](s8_multimodalidad.md).

La propuesta original especifica ERA5-Land por coordenada y fecha/hora. Esta
iteración usa el clima de FungiTastic solicitado: **CHELSA, no ERA5-Land**.
Es una adaptación explícita del protocolo, no la ejecución del enriquecimiento
ERA5 original. [El artículo, sección 2.1](https://arxiv.org/html/2408.13632v2)
identifica CHELSA y sus productos. La integración Android con ERA5 queda pendiente.

Preparación y auditorías ejecutadas; **A/B/C corregidos pendientes de entrenamiento**
por ausencia de imágenes/tablas v002 y CUDA en este equipo. No hay métricas nuevas
de B o C. La comprobación está en `reports/s9/preflight.json`.

## Clima preparado

El [servidor oficial](https://cmp.felk.cvut.cz/datagrid/FungiTastic/shared/download/)
publica `climaticData.zip`; el nombre `climatic.zip` usado por el descargador da
403. Se verificó CRC del ZIP de 545.926.654 bytes, con SHA-256
`39391ba78f378b88b36356f935811cac32bd4726fe6c0ff2321f7f6c228008d8`.
La adquisición se registra en `data/manifests/raw_files.csv`.

Se usa `FungiTastic-Climatic-Average.csv`: `Bio1`–`Bio19`, normales de 1981–2010.
No describen el tiempo meteorológico del día ni el microclima del espécimen.
Se mantiene la escala almacenada, sin afirmar que los números sean directamente
°C o mm; se estandariza cada columna con observaciones únicas de train. El producto
no contiene humedad relativa: no se inventa ni se sustituye por otra variable.

Las series mensuales del otro CSV se conservan pero no entran en C. Según el
artículo, su cobertura acaba en 2020; no representan clima reciente completo de
val 2022. Una variante futura deberá seleccionar ventanas anteriores al evento
y explicitar años sin cobertura.

| Cohorte | Observaciones | GPS `OK` | Clima completo | Clima ausente |
|---|---:|---:|---:|---:|
| train | 17.799 | 17.795 | 17.782 | 17 |
| val | 3.452 | 3.452 | 3.442 | 10 |

Se mantienen las observaciones sin clima para comparar cohortes idénticas con
máscaras de ausencia. Las normales incluyen información posterior al evento
para 745 observaciones de train anteriores a 2011. Es un estudio **retrospectivo
de contexto climático**, no una reproducción estricta de lo conocido en cada
fecha. Antes de alegar evaluación prospectiva, usar un producto disponible a
tiempo o predefinir una sensibilidad temporal sobre una cohorte común A/B/C.

El join usa `observationID`, nunca taxonomía; rechaza duplicados. El anexo local
no altera tablas ni huella de v002 y registra hashes de fuentes, código y salidas:

```powershell
python scripts/data/prepare_s8_s9_context.py
python scripts/train/run_s8_s9.py --check
python scripts/train/run_s8_s9.py --stage all
```

El último comando entrena A/B/C desde el mismo checkpoint S5-C. Para C individual:

```powershell
python scripts/train/train_s8.py --config configs/train/s9_climate_v002.yaml --mode multimodal
```

C se guarda en `runs/s9_geo_climate_v002/`. El modo `metadata` queda como diagnóstico
opcional, no como sustituto de A/B/C. La fusión y sensibilidades de faltantes,
GPS impreciso e intercambio independiente de modalidades se describen en S8.

## 1. Revisar errores de validación

`scripts/reports/audit_s9_validation.py` analiza predicciones guardadas sin leer
test. Valida IDs, etiquetas, clases, cohorte de imágenes, probabilidades y huella
de dataset. Agrega por observación y genera errores confiados, confusiones con
denominador de clase y recall/F1 con soporte train/val. Define clases pequeñas
por el cuartil inferior del soporte de train, sin ajustar el umbral en val.

Ejecutado sobre las tres corridas exploratorias históricas de S8:

- El visual tiene 69 errores de 3.452 observaciones. Destacan Trametes →
  Ganoderma (10), Lycoperdon → Amanita (7) y Amanita → Lycoperdon (5).
- Hericium tiene recall 0,9091 (60/66) y F1 0,9375. Morchella obtiene 11/11
  aciertos: ese soporte pequeño no acredita robustez general.
- De 1.712 observaciones con varias fotografías, 223 tienen géneros predichos
  discordantes entre vistas en el visual. La divergencia Jensen–Shannon entre
  distribuciones por vista permite priorizarlas.
- La fusión exploratoria mantiene los mismos errores por observación del visual;
  no se atribuye a ello un beneficio de los metadatos.

[`REPORT.md`](../reports/s9/validation/REPORT.md) contiene tablas por corrida.
Cada una tiene `manual_review_queue.csv`, con imágenes, confianza, desacuerdo y
campos de revisión humana. Se descargaron y revisaron visualmente 11 fotografías
de seis casos prioritarios: [`revisión preliminar`](../reports/s9/visual_review/REVIEW.md).
Hay un caso con microscopía, otro con vistas visualmente heterogéneas que exige
revisar la agrupación, y errores con morfología en repisa y detalle limitado.
La adjudicación experta y el resto de la cola siguen `PENDING`. Revisar todas las
vistas antes de asignar causa; la confianza no autoriza cambiar etiquetas.
No se modificó ninguna etiqueta ni partición.

Tras las corridas corregidas, sustituir A/B/C por sus directorios reales:

```powershell
python scripts/reports/audit_s9_validation.py --runs runs/s8_geolocation_v002/A runs/s8_geolocation_v002/B runs/s9_geo_climate_v002/C --output reports/s9/validation_corrected
```

Reportar B−A y C−B por observación, clase, disponibilidad climática y celdas GPS
vistas/nuevas. Solo 72 observaciones de val están en celdas de 0,1° no vistas
en train; 3.380 comparten celda. Las conclusiones espaciales requieren cautela.

## 2. Auditar los metadatos

Se reconstruyó train/val desde el `metadata.zip` con el hash de v002 y los IDs
congelados de `splits.csv`. Se reproduce la agregación del curador: fotografías
ordenadas, primer valor no nulo por campo y observación. No se halló discrepancia
entre fotografías en los campos auditados. Consistencia no equivale a veracidad.

| Campo | Origen y tratamiento local | Calidad train / val |
|---|---|---|
| Hábitat | `habitat` del registro; anotación del observador según la fuente; `HABITAT_MAP` | `OK` 17.799 / 3.452; reconocido por el mapa, sin validación ecológica independiente |
| Sustrato | `substrate`; fallback a `metaSubstrate`; mapas separados y valor original conservado | `OK` 15.988 / 3.451; `LOW_PRECISION` 1.811 / 1 |
| Elevación | `elevation` del dataset; rango −500 a 9.000 m; cero marcado `SUSPECT` | `OK` 10.996 / 2.197; cero `SUSPECT` 6.786 / 1.245; ausente 17 / 10 |
| Fecha | `eventDate` del registro; parser valida rango y precisión día/mes/año | `OK` 17.799 / 3.452; train 1874–2021, val 2022 |
| GPS | `latitude`, `longitude`, `coorUncert`; rango, rechazo de (0,0), >10 km baja precisión | `OK` 17.795 / 3.452 |
| Clima | CHELSA asociado por ID; normal 1981–2010; escala original | Derivado de fuente si completo; faltantes explícitos |

El [extractor oficial de elevación](https://github.com/BohemianVRA/FungiTastic/blob/main/metadata_extraction/tabular/extract_elevation_data.py)
apunta a un raster ASTER y extrae por coordenada. Evidencia el método disponible,
pero no prueba la versión ni la trazabilidad de cada valor del ZIP actual;
se conserva `SOURCE_DERIVED_UNDOCUMENTED` del esquema previo. Un cero puede ser
real en la costa: `SUSPECT` es una regla conservadora, no demostración de error.
`OK` en todos estos campos indica que pasó las reglas locales, no fiabilidad medida.

### Los 1.811 casos frente a uno

En **todos** falta `substrate` y `metaSubstrate=wood`. La regla E6 usa
`wood_unspecified` y marca `LOW_PRECISION`. No son 1.811 errores demostrados:
son anotaciones menos detalladas. Tasas: 10,1747 % en train y 0,0290 % en val.
El split es temporal; no hay muestreo aleatorio entre train y val.

Los desgloses están en `reports/s9/audit/substrate_by_year.csv`,
`substrate_by_genus.csv` y `substrate_by_metaSubstrate.csv`. En train predominan
Trametes (521), Ganoderma (433), Pleurotus (269), Auricularia (209), Hericium
(183) y Xylaria (178). Hay casos en varios años, incluso 2020 (231) y 2021 (39).
No basta atribuirlos a registros antiguos o a un cambio de formulario no documentado.
Se confirma la diferencia de detalle en los CSV originales; su causa histórica
no está establecida.

Esto puede influir en la comparación exploratoria que usó sustrato. S10 deberá
probar ausencia, colapso común a categorías amplias y calidad por clase. Con un
solo caso de baja precisión en val no se estima de forma fiable ese estrato.
S8/S9 ignoran completamente el sustrato y no heredan ese factor como entrada.

## 3. Disponibilidad en Android

[`android_metadatos.md`](android_metadatos.md) define proveedor, conexión,
calidad, uso y fallback por campo. Es un contrato para implementar, no una prueba
en teléfono. Captura e inferencia preliminar offline; enriquecimiento remoto
al sincronizar. No se presupone conocimiento de hábitat/sustrato ni sensores
ambientales. Cada ausencia conserva máscara y procedencia.

## Cierre de S9

Completar A/B/C y sus sensibilidades; comparar cohortes idénticas; inspeccionar
las fotografías priorizadas y registrar causas; decidir si la ganancia justifica
el clima y documentar faltantes. Hasta entonces no afirmar mejora, captura
climática automática ni validación móvil.
