# Resultados de las semanas 9 y 10: integración de clima y sustrato

El informe principal se entrega en [LaTeX](../week_09_10/main.tex) y [PDF](../week_09_10/main.pdf), siguiendo la estructura y presentación del informe de las semanas 7 y 8.

## Alcance experimental

Se ajustaron tres clasificadores de contexto en CPU con 17.799 observaciones de train y se conservaron las predicciones visuales de MobileNetV3-Large S5-C. La fusión usa un peso fijo de 0,1 para el contexto. No se realizó ajuste conjunto de la red visual ni una nueva evaluación en test interno.

## Validación

| Modelo | Modalidades | F1 macro | Exactitud (%) | Aciertos | Errores |
|---|---|---:|---:|---:|---:|
| A | Imagen | 0,971674 | 97,68 | 3372 | 80 |
| B | Imagen + GPS | 0,971674 | 97,68 | 3372 | 80 |
| C | Imagen + GPS + clima | 0,971950 | 97,71 | 3373 | 79 |
| D | Imagen + GPS + clima + sustrato | 0,971880 | 97,71 | 3373 | 79 |

La incorporación de clima corrigió dos errores e introdujo uno. El sustrato corrigió dos e introdujo dos respecto a C. Los intervalos de las diferencias incrementales de F1 incluyen cero; no se establece una mejora concluyente con este diseño.

## Evaluación externa

Se verificaron nuevamente 9.649 fotografías y se evaluaron 4.895 observaciones. Colombia incluye 3.348 observaciones y contiene las 309 de Orinoquía. Son fuentes ya evaluadas, no un conjunto nuevo e intacto. Sin clima ni sustrato compatibles, la evaluación mide transferencia geográfica y modalidades faltantes, no el aporte externo de esas modalidades completas.

En Colombia, A obtuvo 88,32 % de exactitud; B/C directos, 87,54 %; D directo, 88,23 %. El filtro geográfico previo desactiva el contexto en todo el externo y coincide con A por construcción.

## Reproducción

```powershell
python scripts/reports/build_s9_s10_report.py
```

Las tablas provienen de `frozen_context/validation_summary.csv`, `paired_validation.json` y `external_summary.csv`. La fuente `main.tex` es autónoma; también puede compilarse directamente con dos pasadas de pdfLaTeX.
