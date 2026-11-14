# Disponibilidad de metadatos para Android

Contrato de adquisición e inferencia para S8/S9. No hay implementación Android
ni mediciones en teléfono en este cambio. La fuente, método, calidad, unidad,
instante de adquisición y precisión deben acompañar cada valor almacenado.
Datos derivados y mediciones directas conservan procedencias distintas.

| Campo | Quién lo aporta | ¿Conexión? | Calidad que conservar | Si falta / uso del modelo |
|---|---|---|---|---|
| Fotografías y vistas | Persona + cámara | No para captura e inferencia | Fecha, enfoque/exposición y agrupación manual de vistas; no asumir un objeto único | Pedir al menos una foto; las vistas adicionales son opcionales; A/B/C usan imagen |
| Latitud/longitud | Proveedor de localización del sistema, GNSS cuando esté disponible | GNSS puede funcionar offline; obtención y tiempo de fijado no están garantizados | Permiso, proveedor, antigüedad, precisión horizontal y si es ubicación aproximada | Sin permiso/fix válido: máscara GPS; C puede usar clima previamente asociado válido; sin contexto usa su rama visual |
| Incertidumbre GPS | `Location.getAccuracy()` y datos del sistema | No si la ubicación está disponible | Radio horizontal, edad y presencia de precisión | No inventar cero; marcar desconocida; no es entrada predictiva en B/C |
| Fecha/hora | Reloj del sistema al capturar; EXIF o persona si es una importación | No; sincronización de reloj puede requerir conexión | UTC, zona y procedencia; no confundir fecha de importación con evento | Importada sin fecha: desconocida; no bloquea A/B/C de normales CHELSA; impide enriquecimiento meteorológico por evento fiable |
| Elevación | Opcional: `Location` si tiene altitud; raster/DEM externo si se enriquece | GNSS no exige conexión; DEM remoto sí, salvo caché local | Referencia vertical y precisión; altitud elipsoidal no equivale automáticamente a elevación sobre el nivel del mar | Ausente, sin imputar cero; no entra en S8/S9 |
| Hábitat | Persona opcional con opción «no sé»; o fuente cartográfica derivada explícita | No para anotación; mapas remotos sí, salvo paquete previo | Categoría, autor o producto, método y calidad | No se presupone que la persona lo conozca; no inferirlo a partir del género predicho; no entra en S8/S9 |
| Sustrato | Persona opcional: amplio, detallado o «no sé» | No para anotación | Detalle/amplitud, valor original y autor; no deducirlo de la etiqueta taxonómica | Desconocido por defecto; reservado para S10 con entrenamiento de ausencia |
| Bio1–Bio19 CHELSA | Fuente externa de normales 1981–2010; no son sensores del teléfono | Sí para primer enriquecimiento remoto; offline solo con datos precargados/caché correcta | Producto/versión, periodo, escala, coordenada/celda de consulta y fecha de descarga | C aplica máscaras por variable; sin clima sigue con imagen+GPS de su propia rama, sin contexto con visual |
| Clima por evento ERA5-Land | Servicio externo a partir de coordenada y fecha/hora | Al sincronizar; offline solo con caché válida | Producto, instante, ventana, resolución, unidades y disponibilidad temporal | Aplazar enriquecimiento; no bloquear inferencia preliminar; no alimentar un modelo CHELSA con variables ERA5 sin reentrenar/validar |
| Temperatura/humedad ambientales directas | Sensor ambiental físico solo si está presente | No | Tipo de sensor, calibración/calidad y fecha | Verificar `getDefaultSensor`; ausente si no existe. No usar temperatura de batería/dispositivo como ambiente; no sustituye las 19 Bio |

## Flujo acordado

1. Capturar foto, fecha y ubicación si la persona concede permiso y hay un fix
   utilizable. El sistema puede conceder solo ubicación aproximada. Un timeout
   o permiso denegado produce ausencia explícita, sin bloquear la identificación.
2. Ejecutar inferencia local con los campos disponibles. El encoder reproduce
   redondeo, categorías, normalización y máscaras del checkpoint; no ajusta
   estadísticas con nuevas observaciones. No hay valores por defecto inferidos
   del género ni del resultado del modelo.
3. Guardar pendientes de enriquecimiento. Con conexión, consultar el producto
   correspondiente; vincular resultado al ID de observación y a sus coordenadas,
   producto y periodo. No reutilizar el clima de otra ubicación por proximidad
   no definida. Un cambio de GPS invalida el enriquecimiento previo incompatible.
4. Actualizar la inferencia si llega contexto válido, conservando qué campos
   tenía cada predicción. Indicar claramente si el contexto climático fue derivado.

Las rutas con datos ausentes y el dropout están implementados en el modelo
Python; la paridad en Android, latencia, memoria y precisión se comprobarán al
convertir el modelo. El fallback visual es el de cada modelo fusionado, no una
promesa de igualdad con A entrenado separadamente.

## Verificaciones pendientes en dispositivo

Probar permiso denegado/aproximado, GPS sin fix y ubicación antigua; modo avión
sin caché climática, caché correcta/incorrecta, reloj o EXIF ausente, teléfono
sin sensores ambientales y anotaciones «no sé». Comprobar que ninguna ruta
requiera hábitat o sustrato, que no se envíen ubicaciones antes de sincronizar y
que las máscaras y resultados coincidan con Python. Medir tiempos reales.

## Fuentes verificadas

- [Permisos y ubicación aproximada](https://developer.android.com/develop/sensors-and-location/location/permissions/runtime).
- [`Location`: precisión, antigüedad y altitud](https://developer.android.com/reference/android/location/Location).
- [Sensores ambientales y comprobación de disponibilidad](https://developer.android.com/develop/sensors-and-location/sensors/sensors_environment).
- [FungiTastic, datos climáticos CHELSA](https://arxiv.org/html/2408.13632v2).
