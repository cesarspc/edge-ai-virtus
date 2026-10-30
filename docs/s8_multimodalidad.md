# S8: inicio de la línea multimodal

## Objetivo

Estudiar si los metadatos ambientales disponibles en el momento de uso mejoran la
clasificación de los diez géneros frente al modelo visual, manteniendo las
particiones oficiales de v002. Este documento describe el trabajo inicial; no
presenta una corrida ni resultados de un modelo multimodal entrenado.

## Datos disponibles

La inspección de `data/processed/v002/observations.csv` se limitó a train y val:

| Campo | Train (17 799 observaciones) | Val (3 452 observaciones) |
|---|---:|---:|
| Hábitat presente | 17 799 | 3 452 |
| Sustrato presente | 17 799 | 3 452 |
| Elevación con calidad OK | 10 996 | 2 197 |
| Clima ausente | 17 799 | 3 452 |

El sustrato tiene calidad `LOW_PRECISION` en 1 811 observaciones de train y en
una de val; la diferencia entre particiones debe investigarse antes de interpretar
resultados. Los metadatos pueden ser datos derivados por la fuente con método
no documentado o anotaciones humanas relacionadas con la identificación. Su
disponibilidad y origen en el uso real deben verificarse; las correlaciones
geográficas y ecológicas pueden generar atajos y limitar la generalización.

## Implementación

`scripts/train/pipeline/multimodal.py` implementa un codificador que acepta
únicamente observaciones de train al ajustar categorías y escala de elevación.
Incluye indicadores para categorías desconocidas, fecha ausente o de calidad no
OK y elevación ausente o de calidad no OK. El estado del codificador debe
guardarse con cada checkpoint para reproducir la inferencia. El hábitat y el
sustrato se incluyen por categoría; sus marcas de calidad todavía no son
entradas separadas. La fecha se codifica como posición cíclica en el año.
No se emplean taxonomía, identificadores, coordenadas, año ni clima.

El modelo inicial suma a los logits visuales una contribución tabular lineal
inicializada en cero. Antes del entrenamiento, sus predicciones coinciden con
las del modelo visual cargado. Una contribución aditiva permite probar una
primera fusión sencilla, pero no modela interacciones imagen-metadatos.

`train_s8.py` enlaza imágenes y observaciones con validación de identificador,
partición y etiqueta, y ofrece tres modos con la misma configuración y cohorte.
El modo visual y el multimodal parten del checkpoint C; el modo de metadatos
entrena una capa tabular. El entrenamiento usa imágenes como unidad, por lo que
una observación con varias imágenes contribuye varias veces. Las métricas de
validación también se calculan por observación promediando probabilidades.

Además de la validación normal, los modos con metadatos se evalúan con todos
los campos ausentes y con los metadatos intercambiados entre observaciones de
forma reproducible. Son pruebas de sensibilidad: no equivalen a mediciones de
datos ausentes o incorrectos en un entorno real. En la fusión, la rama tabular
incluye un sesgo; por ello, la condición ausente no equivale necesariamente a
la salida visual original.

Las tres primeras corridas S8 están completas y sus resultados se encuentran en
`runs/s8_multimodal_v002/`. Se evaluó únicamente val; no se consultó test.

## Ejecución manual en PowerShell

Desde la raíz del repositorio y con el entorno y las imágenes v002 ya
preparados, ejecutar cada comando por separado. Los procesos comprueban los
hashes de las imágenes, por lo que el inicio puede tardar varios minutos. Por
defecto exigen CUDA; si se desea CPU, añadir `--device cpu`.

```powershell
cd D:\Dev\Virtus\edge-ai-virtus
.\.venv\Scripts\python.exe scripts\train\train_s8.py --config configs\train\s8_multimodal_v002.yaml --mode visual
.\.venv\Scripts\python.exe scripts\train\train_s8.py --config configs\train\s8_multimodal_v002.yaml --mode metadata
.\.venv\Scripts\python.exe scripts\train\train_s8.py --config configs\train\s8_multimodal_v002.yaml --mode multimodal
```

Cada corrida muestra la carpeta creada y guarda `last.pt` al terminar cada
época. Si se interrumpe con Ctrl+C, se pierde el trabajo de la época en curso;
para reanudar, usar el mismo modo y configuración, y sustituir la ruta por la
que se mostró en pantalla:

```powershell
.\.venv\Scripts\python.exe scripts\train\train_s8.py --config configs\train\s8_multimodal_v002.yaml --mode multimodal --resume "runs\s8_multimodal_v002\NOMBRE_DE_LA_RUN\last.pt"
```

El mismo comando sirve para `visual` o `metadata` cambiando `--mode` y la ruta.
La reanudación exige el mismo dispositivo, código, configuración, dataset y
checkpoint inicial. Si se cambia el código tras una interrupción, conservar el
estado anterior para reanudar esa corrida. Se puede añadir
`--pause-after-epoch` para detenerse ordenadamente al final de la próxima época.
Las corridas finalizadas no se reanudan.

Cada carpeta contiene `run.json`, `progress.json`, `history.csv`, `last.pt`,
`best.pt`, métricas y predicciones de val, `RESULTS.md` y `completed.json` al
terminar. La selección y parada temprana usan F1 macro por observación en val.

## Siguientes pasos

1. Comprobar la procedencia, calidad y disponibilidad en Android de cada
   variable; investigar el cambio de calidad del sustrato entre train y val.
2. Comparar las métricas por género y las pruebas de sensibilidad en val;
   estudiar la variación entre semillas antes de atribuir una mejora a la fusión.
3. Seleccionar el protocolo antes de cualquier evaluación final en test y medir
   disponibilidad de metadatos y latencia en Android antes de concluir que es
   viable para despliegue.
