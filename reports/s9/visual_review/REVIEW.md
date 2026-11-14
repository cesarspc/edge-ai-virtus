# Revisión visual preliminar de seis observaciones de val

Se inspeccionaron las **11 fotografías** originales de seis observaciones,
obtenidas por HTTP Range del ZIP oficial de val. Cada foto pasó CRC, decodificación
y comparación SHA-256 con `data/versions/v002/image_integrity.csv`.
`download_manifest.csv` y `provenance.json` registran la adquisición y selección.
Las fotografías viven en `data/raw/images/s9_review_v002/` y las hojas de contacto
en `tmp/s9_review/`; no se publican copias de las imágenes en Git.

La selección es intencional: dos errores de mayor confianza, dos observaciones
con mayor divergencia entre vistas, un error de Hericium y un caso de Morchella.
No estima la prevalencia de problemas en el dataset. Se revisó la morfología y
calidad visual, sin validar taxonomía experta. Las etiquetas y particiones no
cambian. `VISUAL_TRIAGE` significa inspección preliminar, no caso adjudicado.

| Observación | Etiqueta / predicción agregada | Evidencia visible | Interpretación provisional y siguiente acción |
|---|---|---|---|
| `3758574325` | Trametes / Ganoderma, 0,914 | Una foto: cuerpo en repisa sobre madera, superficie con bandas concéntricas; entorno ocupa bastante imagen y no se muestra el reverso | Ambigüedad visual entre formas en repisa. No hay evidencia suficiente de etiqueta incorrecta. Pedir vista inferior y revisión experta |
| `3451831353` | Pleurotus / Ganoderma, 0,897 | Una foto de cuerpos en repisa en tronco; contraste bajo, aspecto velado y zonas claras; detalle inferior limitado | Calidad/contraste y morfología compartida son explicaciones plausibles. Revisar detalle de la cara inferior; no atribuirlo automáticamente al clasificador o etiqueta |
| `3463110327` | Auricularia / Auricularia, 0,324 | Tres fotos: forma oscura compacta, forma marrón en copa y vista más abierta del tronco con cuerpo pequeño; variación marcada de aspecto y escala | Revisar agrupación de fotos y si representan el mismo espécimen. No basta la imagen para confirmar mezcla de taxones. Las predicciones por vista son Xylaria (0,932), Auricularia (0,915), Trametes (0,758): el acierto agregado es poco confiado y oculta desacuerdo |
| `3469661329` | Xylaria / Xylaria, 0,306 | Dos fotografías macroscópicas de estructuras blancas ramificadas/elongadas; una tercera es una imagen microscópica | Discordancia por tipo de vista: imagen macro completa, detalle macro y microscopía. Predicciones Xylaria (0,885), Amanita (0,787), Ganoderma (0,834). Revisar un análisis separado de microscopía; no descartar la etiqueta por ese desacuerdo |
| `3819292316` | Hericium / Trametes, 0,833 | Dos fotos del mismo conjunto claro sobre tocón; una cercana y otra amplia. La cara inferior tiene textura fina pero la resolución no permite verificar rasgos diagnósticos | Ambas vistas predicen Trametes (0,799 y 0,867): es un error consistente, no desacuerdo de vistas. Solicitar detalle inferior/revisión experta para distinguir ambigüedad, etiqueta o fallo del modelo |
| `3756036316` | Morchella / Morchella, 0,968 | Una foto del cuerpo completo, con superficie alargada con cavidades y algo de desenfoque | Acierto compatible con forma visible; no confirma robustez en clases pequeñas. Val solo tiene 11 observaciones de este género |

Las fotografías discordantes no son todas errores: ambas observaciones elegidas
por divergencia terminan acertadas al promediar probabilidades. Deben reportarse
el desacuerdo y la confianza además del acierto agregado. El caso de microscopía
motiva una auditoría de tipos de fotografía; no se aplicó una exclusión a posteriori.

## Qué queda pendiente

Adjudicación por especialista y revisión del resto de la cola. Usar causas
separadas (modelo, calidad, ambigüedad, etiqueta dudosa, agrupación dudosa,
dominio de vista o evidencia insuficiente), permitir varias por caso y conservar
las pruebas. No recalcular una métrica «limpia» eliminando errores después de
ver las predicciones. Cualquier futura corrección requiere una versión nueva
y evaluación de todas las variantes sobre la misma cohorte.

Para reconstruir la muestra y hojas de contacto:

```powershell
python scripts/data/fetch_s9_review_images.py
```
