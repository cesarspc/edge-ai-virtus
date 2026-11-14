# S8 corregida: imagen y geolocalización

## Alcance y estado

La propuesta `Propuesta_Virtus (2).pdf`, páginas 3–5, define A = imagen,
B = imagen + geolocalización (S8), C = imagen + geolocalización + clima
(S9) y D = las anteriores + sustrato (S10). Hábitat, elevación y fecha no
son entradas de A/B/C en este protocolo. La fecha se conserva para auditoría.

El commit `4a41e06` entrenó otra comparación: imagen frente a imagen con
hábitat, sustrato, elevación y estacionalidad. Se conservan esas corridas en
`runs/s8_multimodal_v002/` como exploratorias; **no prueban el aporte del GPS**.
Su configuración histórica está en
`configs/train/archive/s8_environmental_exploratory_v002.yaml` y el código
original permanece en ese commit. No reanudar checkpoints antiguos con el
código corregido: la comprobación de hashes lo impide.

La implementación corregida y sus pruebas están preparadas. **Las nuevas
corridas A y B no se han entrenado**: en este checkout faltan las fotografías,
el manifiesto de descarga y las tablas procesadas v002. PyTorch es CPU.
`reports/s9/preflight.json` registra esa comprobación; no contiene métricas nuevas.

## Comparación controlada

| Modelo | Entrada | Semana | Configuración | Modo |
|---|---|---|---|---|
| A | Imagen | S8 | `configs/train/s8_multimodal_v002.yaml` | `visual` |
| B | Imagen + GPS | S8 | misma configuración | `multimodal` |
| C | Imagen + GPS + clima CHELSA | S9 | `configs/train/s9_climate_v002.yaml` | `multimodal` |

Los tres parten del mismo checkpoint visual S5-C, con la misma cohorte,
particiones oficiales, augmentations, resolución, semilla, optimizador, pesos
de clase y criterio de selección. S5-C nombra la candidata visual previa;
no es el Modelo C de modalidades. B y C no se inicializan del mejor A o B:
así evitan heredar distintas cantidades de entrenamiento. Se comparan
regímenes de entrenamiento equivalentes con parada temprana, no un número
forzosamente igual de épocas. Una semilla no permite afirmar significancia;
repetir semillas antes de una selección definitiva.

La unidad de entrenamiento sigue siendo la imagen: una observación con más
vistas tiene más peso. Selección por macro-F1 **por observación**, promediando
las probabilidades de sus fotografías. Se guardan también métricas por imagen,
género, top-1 y top-3. Solo se usan train y val; no se modifica ni evalúa test.

## GPS y ausencia de contexto

`pipeline/context.py` limita las entradas a una lista explícita. GPS se redondea
a 0,1 grados y se codifica como tres coordenadas en la esfera y un indicador
de ausencia. La resolución angular no equivale a una distancia fija ni a una
garantía de privacidad. La codificación lineal es una primera aproximación
de baja capacidad. Coordenadas inválidas o con calidad distinta de `OK` se
enmascaran; la incertidumbre se conserva para auditoría y no entra al clasificador.
`OK` significa que pasó las reglas locales, no precisión de GPS certificada.

No entran identificadores, etiquetas, país, región, hábitat, sustrato,
elevación, fecha ni clima a B. C añade únicamente `Bio1`–`Bio19`, con medias y
desviaciones ajustadas en observaciones únicas de train y máscaras por variable.
El codificador y la huella de los datos se guardan con cada checkpoint.

La fusión suma un residual tabular inicializado en cero a los logits visuales.
Se aplica dropout de modalidades de 0,2 en entrenamiento. Si falta todo el
contexto, el residual se anula exactamente y se usa la rama visual entrenada
de esa corrida; esto no garantiza igualdad con un A entrenado por separado.

Se evalúan contexto normal, todo ausente, GPS ausente, GPS intercambiado por
observación y GPS redondeado a un grado. C añade clima ausente e intercambiado
independientemente del GPS. Las pruebas son sensibilidades reproducibles,
no simulaciones certificadas del error de Android. El intercambio conserva
juntas todas las vistas de cada observación.

## Datos, ejecución y trazabilidad

El contexto vive en un anexo local `data/processed/s8_s9_context_v002/`.
No se reescriben las tablas ni la huella de v002. El preparador lee únicamente
los CSV originales Train y ClosedSet-Val, limita sus IDs a la cohorte congelada
y registra hashes de fuentes, código y salidas. Antes de entrenar se verifican
ese anexo y la integridad publicada de tablas, clases e imágenes.

```powershell
python scripts/data/prepare_s8_s9_context.py
python scripts/train/run_s8_s9.py --check
python scripts/train/run_s8_s9.py --stage s8
```

El último comando ejecuta A y después B. Requiere las imágenes originales en
la ubicación configurada, las tablas v002 reproducidas contra sus hashes y un
entorno CUDA. CPU se solicita explícitamente con `--device cpu`.

Para controlar cada corrida o reanudar una interrupción:

```powershell
python scripts/train/train_s8.py --config configs/train/s8_multimodal_v002.yaml --mode visual
python scripts/train/train_s8.py --config configs/train/s8_multimodal_v002.yaml --mode multimodal
python scripts/train/train_s8.py --config configs/train/s8_multimodal_v002.yaml --mode multimodal --resume "runs/s8_geolocation_v002/NOMBRE/last.pt"
```

`--pause-after-epoch` permite una pausa al terminar la época. Los checkpoints
guardan optimizador, scheduler, RNG, codificador y procedencia; las corridas
completas no se reanudan. Las nuevas salidas se guardan en
`runs/s8_geolocation_v002/`, separadas de las exploratorias.

## Lectura de resultados disponibles

La auditoría de las corridas exploratorias está en
[`reports/s9/validation/REPORT.md`](../reports/s9/validation/REPORT.md).
Visual y fusión tienen 69 errores sobre 3.452 observaciones y macro-F1
0,977108. Las probabilidades difieren, pero no cambian los aciertos o errores
por observación. Esto no demuestra mejora multimodal ni evalúa geolocalización.
El modelo solo de metadatos tiene macro-F1 0,188961.

La validación comparte celda GPS redondeada con train en 3.380/3.452 casos;
72 quedan en celdas no vistas. Una posible mejora de B debe leerse junto a
este solapamiento y al desglose por celdas vistas/nuevas. No se puede extrapolar
automáticamente a Colombia ni atribuir causalidad ecológica a la ubicación.

S9, la auditoría de metadatos y el contrato Android se describen en
[`s9_clima_auditoria.md`](s9_clima_auditoria.md).
