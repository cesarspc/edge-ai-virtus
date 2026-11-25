# S9 y S10: ejecución realizada y alcance

## Resultado entregado

Se entrenaron en CPU tres ramas de contexto con las 17.799 observaciones de train:
GPS (B), GPS+CHELSA (C) y GPS+CHELSA+sustrato amplio (D). La parte visual es
MobileNetV3-Large S5-C congelada; se reutilizan sus probabilidades por imagen
y se agregan por observación. Es el mismo checkpoint en val y externo.

Se compara A = foto con B/C/D = 90 % foto + 10 % contexto, usando las mismas
3.452 observaciones de val. El peso y los hiperparámetros se fijaron antes de
calcular resultados; no hubo búsqueda ni ajuste posterior en val/externo.
No se entrenó conjuntamente el backbone visual. Los modelos de contexto
quedan en JSON con coeficientes, estadísticas y categorías, sin depender de
un pickle. El estado completo del protocolo y sus hashes se conserva.

Informe LaTeX: [`main.tex`](../reports/week_09_10/main.tex).
PDF compilado: [`main.pdf`](../reports/week_09_10/main.pdf).
Resumen: [`REPORTE.md`](../reports/s9_s10/REPORTE.md).

## Hallazgos

- A/B: 3.372 aciertos y 80 errores. C/D: 3.373 aciertos y 79 errores.
- C frente a B corrige dos e introduce uno. D frente a C corrige dos e introduce dos.
- El IC95 de ambas diferencias de macro-F1 incluye cero. No hay evidencia
  suficiente para afirmar una mejora estable con este diseño y una semilla.
- Se verificaron nuevamente las 9.649 imágenes del conjunto externo completo.
  En Colombia, A obtuvo 88,32 %, C directo 87,54 % y D directo 88,23 % de acierto.
  En Orinoquía: A 77,02 %, C directo 76,05 %, D directo 77,02 %.
- El contexto externo no tiene clima/sustrato compatibles. Los resultados de
  C/D externos prueban rutas de datos ausentes y transferencia geográfica,
  no el beneficio de clima/sustrato externos reales.
- El filtro de dominio, fijado con percentiles de coordenadas de train, deja
  todo el externo en fallback visual. No se vende esa igualdad con A como una
  mejora ni como validación de un detector de dominio.

Los conjuntos externos ya se habían evaluado antes. Son una auditoría de
robustez sobre fuentes conocidas, no un test nuevo e intacto. Colombia está
contenida en el conjunto completo; Orinoquía está contenida en Colombia.
No sumar esos grupos ni tratarlos como réplicas independientes.

## Reproducir el ensayo disponible en este equipo

```powershell
python scripts/data/prepare_s8_s9_context.py
python scripts/train/run_s9_s10_frozen.py --external
python scripts/reports/build_s9_s10_report.py
```

El generador del informe utiliza la biblioteca estándar de Python y pdfLaTeX
(MiKTeX o TeX Live). Genera una fuente autónoma desde la plantilla y las tablas
guardadas, y compila el documento en dos pasadas. `--source-only` genera solo
la fuente. La tipografía y estructura siguen el informe de las semanas 7 y 8.
Los resultados, sensibilidades y bootstrap se guardan en
`reports/s9_s10/frozen_context/`. La configuración está en
`configs/experiments/s9_s10_frozen_context.yaml`. El test interno no se evalúa.

Se usaron las normales CHELSA 1981-2010, no ERA5-Land. Las 745 observaciones
de train anteriores a 2011 impiden presentar esto como una reproducción
histórica estricta de información disponible en cada fecha. Los valores se
estandarizan en su escala original, sin conversiones físicas no verificadas.

## Preparación para ajuste conjunto en el equipo con GPU

El conflicto entre el pull y los cambios locales se resolvió conservando
el modo GPS-only del compañero como diagnóstico y el protocolo incremental
con entradas explícitas como principal. GPS-only no es el Modelo B incremental.

Cuando estén las fotografías y tablas v002 completas:

```powershell
python scripts/train/run_s8_s9.py --check
python scripts/train/run_s8_s9.py --stage all
```

Ese comando entrena A/B/C/D; S10 usa `configs/train/s10_substrate_v002.yaml`.
La fusión neural tiene máscaras, dropout por modalidad y fallback exacto a
su rama visual cuando falta todo el contexto. Sus resultados deben publicarse
separados del ensayo de combinación final hecho en CPU.

## Comprobaciones realizadas

```powershell
python -m pytest tests/test_multimodal.py tests/test_context.py tests/test_context_training.py tests/test_s9_audit.py tests/test_s9_s10_frozen.py -q
```

21 pruebas aprobadas: codificación train-only, modalidades permitidas, sustrato
amplio, ausencias, mezcla de probabilidades, cohortes guardadas, bootstrap y
ciclo de entrenamiento/pausa/reanudación de S9 y S10 con datos sintéticos.
El PDF en LaTeX tiene siete páginas, con fuentes incrustadas y revisión visual.

Continúan disponibles las auditorías y revisión visual anteriores en `reports/s9/`;
no se borraron ni se reetiquetaron observaciones para mejorar resultados.
