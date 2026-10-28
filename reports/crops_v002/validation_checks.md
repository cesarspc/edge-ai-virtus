# Verificación de la campaña de recortes

- Pasaron 12 pruebas dirigidas: geometría y áreas; identidad del control; promedio de vistas por foto; SHA-256 paralelo y detección de archivos modificados; rechazo de bloques incompatibles; bootstrap pareado contra sklearn y reanudación exacta; inicialización nueva y reanudación con la referencia de pesos; separación de parámetros y buffers; evaluación 2 × 2 y reutilización de bloques; ciclo de entrenamiento/evaluación sin imágenes de test; cobertura de hashes de código e interfaz pública.
- La sintaxis de ambos lanzadores PowerShell y los módulos Python nuevos fue verificada.
- Se probó una pausa real tras 64 fotos de la evaluación congelada y se reanudó hasta las 6 290. Los bloques terminados se conservaron.
- El control congelado reprodujo todas las decisiones originales de C y sus dos matrices de confusión. Sus parámetros y buffers mantuvieron el mismo SHA-256 antes y después de inferencia.
- Las dos ramas de ajuste usaron exactamente los mismos hashes de fuentes de entrenamiento, inicialización y pesos de pérdida por género; cada una completó 471 actualizaciones de AdamW en una época.
- Los pilotos iniciales quedaron pausados con `last.pt`, referencia inicial y registros de cambios de pesos. Sus continuaciones están consolidadas en el apartado siguiente.
- Las evaluaciones de validación del piloto conservaron los estados de ambos modelos. Las métricas en el encuadre propio reproducen las del historial de entrenamiento.
- El checkpoint de referencia C conserva el SHA-256 `e7b2dee4c742adce5b9c601c36a1b9844e9e183b4a8389e08d1274ee38959cd8`.
- No existe el marcador de test final de S5. La campaña de recortes registra cero evaluaciones de test y mantiene la selección histórica de S5.
- Las fotografías y recortes de la galería siguen excluidos de Git. Solo las figuras estadísticas están habilitadas para versionado.

Resultados congelados: `20261024T161742286751Z_frozen_C/RESULTS.md`.

Piloto pareado: `pilot_seed42/RESULTS.md`. Comandos y límites de interpretación: `docs/recortes_v002.md`.

## Continuaciones completadas

- Control y central 75 % finalizaron por parada temprana en la época 4; la mejor época fue 1 en ambas. Sus marcadores `completed.json` registran únicamente val y cero evaluaciones de test.
- Se verificaron exactamente 312 tensores por modelo, incluidos los buffers. El estado de cada mejor checkpoint coincide con el piloto; el cambio de hash de los archivos se debe a su reserialización.
- Las predicciones finales propias reproducen las matrices de confusión del piloto y coinciden con sus probabilidades dentro de la tolerancia de redondeo. Se reutilizó la evaluación 2 × 2 sin inferencia nueva.
- Pasaron tres pruebas nuevas que comprueban equivalencia tras reserialización y rechazo de parámetros, buffers, clases o receta diferentes.
- La consolidación se ejecutó nuevamente con los mismos argumentos: reutilizó el bootstrap guardado y completó los mismos resultados.
- Se incorporaron las curvas de entrenamiento y distancia de pesos de las cuatro épocas. El candidato por validación es control con encuadre original; la selección histórica de S5 y el test permanecen intactos.

Informe: `completed_seed42/RESULTS.md`. Pruebas de identidad y hashes: `completed_seed42/reuse_audit.json`.

## Estabilidad del control con tres semillas

- Las semillas 43 y 44 se ejecutaron secuencialmente en CUDA, partiendo del mismo checkpoint C y cambiando únicamente la semilla y sus metadatos. Terminaron por parada temprana en las épocas 5 y 9; sus mejores épocas fueron 2 y 6.
- Se preservó la semilla 42 como candidato fijado antes de las repeticiones. El análisis no elige la mejor semilla ni combina probabilidades en un ensemble.
- Pasaron siete pruebas nuevas: receta idéntica salvo semilla; reutilización de la primera corrida completa sin elegir por puntuación; descubrimiento del último checkpoint incompleto; exclusión y liberación del bloqueo; desviación estándar muestral; preservación del candidato 42; agregación y validación de identidades por observación.
- La sintaxis del lanzador PowerShell y los módulos nuevos se verificó. El dispositivo CUDA explícito se comprobó en las dos ejecuciones reales.
- Repetir el lanzador tras terminar reutilizó las tres corridas completas, sin iniciar entrenamiento nuevo. Repetir el análisis reutilizó los 10 000 remuestreos guardados.
- Se verificaron la cohorte completa de val, las matrices de confusión reproducidas desde los CSV, las versiones de bibliotecas, GPU, precisión, inicialización y pesos de pérdida comunes.
- La comparación a épocas/lotes comunes contiene 48 filas; las actualizaciones reales del optimizador coinciden en ellas.
- La media del F1 macro por observación fue 0,9732 y la desviación estándar muestral 0,0037. Todos los intervalos pareados frente a C incluyen cero; no se afirma una mejora concluyente.
- Los archivos de referencia C y del candidato 42 conservan sus hashes. Las tres corridas guardan `best.pt` y `last.pt`; ninguna evaluó test. La selección histórica de S5 permanece intacta.
- Se revisó visualmente la legibilidad de las figuras de validación, entrenamiento y trayectoria de pesos.

Informe y trazabilidad: `control_stability/results/RESULTS.md` y `analysis_manifest.json`.
