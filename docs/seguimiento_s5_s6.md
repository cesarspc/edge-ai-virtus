# Seguimiento de S5 y S6

Actualizado: 12 de octubre de 2026.

Este documento registra el estado práctico de las fases S5 y S6 y los controles que deben mantenerse para que las comparaciones sean reproducibles.

## Estado de S5: línea base visual eficiente

S5 compara MobileNetV3-Large con cuatro configuraciones controladas sobre el dataset `v002` de FungiTastic. La métrica para escoger modelo es macro-F1 por observación en `val`. `test` queda reservado para una evaluación final del único modelo seleccionado.

| Ensayo | Tasa de aprendizaje | Ponderación de clases | Estado |
|---|---:|---|---|
| A | 0,0003 | Ninguna | Completado; early stop en época 24, mejor época 19 |
| B | 0,0001 | Ninguna | Pendiente |
| C | 0,0003 | Inversa de raíz del conteo | Pendiente |
| D | 0,0001 | Inversa de raíz del conteo | Pendiente |

El ensayo A alcanzó macro-F1 de observación **0,9706** y macro-F1 de imagen **0,9254** en `val`. No evaluó `test`. Los detalles y hashes se actualizan en `reports/s5/development_results.md` y `.json`; el historial completo queda en la carpeta local de la corrida, que Git ignora.

### Pendientes y controles S5

1. Ejecutar B, C y D con sus configuraciones fijadas, la misma huella de dataset, seed, particiones, resolución y procedimiento de entrenamiento que A.
2. Conservar `last.pt` y `best.pt` locales de cada corrida, y revisar que se haya completado normalmente. No reanudar una corrida ya marcada como completada.
3. Comparar primero A contra B para el efecto del LR y A contra C para la ponderación. D permite observar la combinación; cuatro ensayos y una sola seed no permiten afirmar estabilidad entre semillas.
4. Revisar macro-F1, precisión, recall, matriz de confusión y soporte en `val`, prestando atención a Morchella, Hericium y Geastrum, cuyos soportes son pequeños. No ajustar usando `test`.
5. Después de decidir cerrar la campaña y registrar los ensayos pendientes, fijar el checkpoint ganador por macro-F1 de observación en `val` con `scripts/train/summarize_s5.py --select-run "runs/<run_id>"`.
6. Ejecutar `scripts/train/evaluate_s5_final.py` una sola vez para la selección congelada. La evaluación final de `test` cierra los ajustes de esta campaña.
7. Completar la medición móvil en Android cuando se disponga del teléfono concreto. Registrar modelo del dispositivo, versión de Android, runtime y versión, precisión numérica, tamaño del artefacto, latencias (incluidos percentiles), memoria y condiciones de medición. El perfil sintético de CPU del PC no equivale a rendimiento en Android.
8. Actualizar el informe formal con el resultado MobileNet y su comparación con ResNet-18 después de completar la selección y evaluación final; indicar claramente que la comparación usa el mismo `v002` y reportar métricas por imagen y observación.

## Alcance siguiente: S6

Según la planificación del proyecto, S6 consolida las métricas base por nivel taxonómico y modulariza el pipeline para preparar la Fase 5. S6 debe iniciar una vez cerrado y documentado S5.

### Trabajo previsto para S6

- Definir qué niveles taxonómicos están disponibles y confiables en los metadatos de `v002`; documentar etiquetas faltantes o ambiguas antes de calcular métricas.
- Publicar métricas base por nivel taxonómico, además del nivel de género ya evaluado: macro-F1, precisión, recall, exactitud, soporte y matrices de confusión donde la definición aplique.
- Mantener el mismo protocolo oficial de particiones y agregación por observación. No mezclar métricas por imagen y por observación ni cambiar la partición entre niveles sin registrarlo.
- Diseñar interfaces pequeñas para carga/versionado del dataset, transformaciones, modelo, entrenamiento, evaluación y exportación de resultados. Separar responsabilidades sin alterar silenciosamente la receta S5 ni romper la reanudación por checkpoint.
- Añadir configuración explícita y registro de hashes/versiones para que una corrida identifique dataset, clases, arquitectura, pesos iniciales, transformaciones, hiperparámetros y entorno.
- Definir criterios de aceptación: métricas reproducibles a partir de predicciones guardadas, cobertura de clases por nivel, checkpoints reanudables y ejecución visible con barras de progreso y guardados periódicos.
- Dejar la integración de la aplicación móvil y tareas posteriores de Fase 5 fuera del alcance de S6 salvo que la planificación se actualice expresamente.

### Riesgos y decisiones abiertas

- **Soporte por clase:** los resultados de clases con pocas observaciones tienen incertidumbre alta; mostrar siempre soporte y evitar presentar un único F1 como concluyente.
- **Unidad estadística:** varias imágenes pueden pertenecer a una observación. Las comparaciones principales deben continuar a nivel de observación para no contar cada fotografía como una muestra independiente.
- **Una sola seed:** registrar esta limitación en las conclusiones; no describir diferencias pequeñas entre ensayos como evidencia definitiva.
- **Trazabilidad:** mantener `v001` y `v002` inmutables. Cualquier cambio en fuentes, limpieza, imágenes o particiones exige una versión nueva del dataset.
- **Privacidad/licencia de datos:** subir código, metadatos derivados permitidos y documentación; no subir fotografías originales, checkpoints ni carpetas de ejecución.

## Comandos de referencia

Lanzar B, C o D desde la raíz del repositorio (cambiar el sufijo de la configuración):

```powershell
powershell -ExecutionPolicy Bypass -File scripts/train_visible.ps1 -Config configs/train/s5_mobilenet_v3_large_v002_b.yaml -SkipInstall
```

Actualizar el resumen de corridas terminadas:

```powershell
.\.venv\Scripts\python.exe scripts/train/summarize_s5.py
```

La selección final y la evaluación de test se hacen solo cuando se acuerde cerrar la comparación, siguiendo el orden y las salvaguardas de `docs/s5_mobilenet_v002.md`.
