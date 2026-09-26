# Traspaso — Fase 4: entorno de entrenamiento y primer entrenamiento de referencia

> Estado al 2026-09-26. Todo lo de esta guía está **preparado pero no ejecutado**: no se descargaron las imágenes,
> no se entrenó ningún modelo y el dataset sigue siendo `v001` (sin cambios).

## Qué hay listo

| Pieza | Archivo | Estado |
|---|---|---|
| Dataset procesado | `data/versions/v001/` + `data/processed/v001/` | Publicado (huella `6704c19c…`). No modificar. |
| Descarga de imágenes (solo FungiTastic) | `scripts/data/download_images.py` | Probado con 9 imágenes y con `--dry-run` completo (43.869 de 43.869 encontradas, 1,08 GB). |
| Dependencias de entrenamiento | `requirements-train.txt` | Sin instalar en el equipo de preparación (faltaba `torchvision`). |
| Configuración del primer entrenamiento | `configs/train/ma_resnet18_v001.yaml` | ResNet-18 preentrenada (baseline académica del alcance). |
| Verificación del entorno | `scripts/train/check_env.py` | Probado. |
| Entrenamiento y evaluación | `scripts/train/train_ma.py` (+ `data.py`, `metrics.py`) | `data.py` y `metrics.py` probados; `train_ma.py` **compilado pero no ejecutado** (sin `torchvision`). Validarlo primero con `--smoke`. |

**Alcance de esta fase:** solo FungiTastic (train/val/test de `v001`), modelo solo imagen ($M_A$), **sin segmentación**.
Las fuentes de prueba externa (iNaturalist, Mushroom Observer, MIND.Funga) quedan para después.

## Pasos (en orden)

### 1. Descargar las imágenes (una orden)

```bash
python scripts/data/download_images.py
```

* Descarga solo las 43.869 imágenes de `v001` (≈ 1,1 GB) desde los ZIP oficiales de 300 px, leyendo por rangos HTTP
  (no baja los 14,5 GB completos). ≈ 34.000 peticiones con 4 hilos.
* Es **reanudable**: si se corta, se vuelve a ejecutar la misma orden y solo baja lo que falta o falló.
* Cada imagen se verifica (CRC-32 del ZIP, decodificación con Pillow) y queda en
  `data/raw/images/fungitastic/download_manifest.csv` con su SHA-256, tamaño y hash perceptual.
* **OneDrive:** el repositorio está en una carpeta sincronizada. Recomendado:
  `python scripts/data/download_images.py --out D:/virtus_images/fungitastic` y cambiar `data.image_root` y
  `data.download_manifest` en la configuración de entrenamiento a esa ruta.
* Para probar antes: `--dry-run` (solo planifica) o `--limit 50` (50 imágenes por partición).

### 2. Instalar el entorno de entrenamiento

```bash
# elegir la variante de PyTorch según el hardware (https://pytorch.org/get-started/locally/), p. ej. GPU NVIDIA:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt -r requirements-train.txt
python scripts/train/check_env.py --config configs/train/ma_resnet18_v001.yaml   # debe terminar en "LISTO"
```

`check_env.py` comprueba bibliotecas, dispositivo (GPU/CPU), que `data/processed/v001` coincida con el manifiesto
publicado y que todas las imágenes de train/val/test estén descargadas.

### 3. Prueba rápida del pipeline

```bash
python scripts/train/train_ma.py --config configs/train/ma_resnet18_v001.yaml --smoke
```

Entrena 3 lotes con 4 imágenes por clase y evalúa: sirve para detectar errores de entorno en minutos. Las
métricas de esta prueba no significan nada.

### 4. Primer entrenamiento de referencia

```bash
python scripts/train/train_ma.py --config configs/train/ma_resnet18_v001.yaml
```

* Se recomienda GPU; en CPU cada época sobre 30.663 imágenes tardará horas.
* Salida en `runs/<fecha>_ma_resnet18_v001/` (ignorado por Git): `run.json` (configuración y su SHA-256, versión y
  huella del dataset, commit, versiones, dispositivo), `history.csv`, `best.pt`, `metrics_val.json`,
  `metrics_test.json` y `predictions_<split>.csv`.
* La selección del mejor modelo y la parada temprana usan **solo `val`** (macro-F1 por observación). `test` se
  evalúa una sola vez al final. No ajustar hiperparámetros mirando `test`.

## Qué reportar del primer entrenamiento

Por partición (val y test), a nivel **imagen** y **observación** (media de probabilidades de las fotos de cada
observación): macro-F1 (métrica principal), exactitud, exactitud balanceada, top-1/top-3, F1 por clase y matriz de
confusión. Todo está en `metrics_*.json`. Citar siempre la versión del dataset (`v001`) y su huella, y el `run_id`.

## Decisiones ya tomadas (no re-discutir sin motivo)

* Partición principal `grouped_hash` por sitio-día (`split`). Existe `split_temporal` (partición oficial por año)
  como experimento de robustez: `data.split_column: split_temporal`.
* Entrada 224 px desde las imágenes de 300 px (lado corto ≈ 225 px); aumentos suaves de color porque el color es
  diagnóstico.
* Sin segmentación: las máscaras de FungiTastic solo cubren FungiTastic-Mini (de nuestras clases, solo *Amanita*).
* Licencia de FungiTastic tratada como CC BY-NC-SA (la más restrictiva declarada): uso académico no comercial.

## Pendientes y advertencias

1. **`train_ma.py` no se ha ejecutado nunca.** Si falla el `--smoke`, corregir antes del entrenamiento completo.
2. **Desbalance:** Amanita tiene ~20 veces más observaciones de entrenamiento que Morchella (4.209 vs. 214). La
   configuración de referencia no pondera clases (`class_weighting: none`); comparar después con `inverse_sqrt`.
3. **Otras arquitecturas del alcance:** MobileNetV3-Large y EfficientNet funcionan cambiando `model.architecture`
   (`mobilenet_v3_large`, `efficientnet_b0`, …); EfficientNet-Lite0 y MobileViT-XXS no están en torchvision
   (requieren `timm`, no incluido).
4. **Si la descarga deja imágenes fallidas o se decide excluir casi-duplicados** (el manifiesto ya trae `phash64`),
   eso cambia el contenido del dataset y debe publicarse como `v002` siguiendo `data/versions/README.md`; no editar `v001`.
5. **Espacio en disco:** el equipo de preparación tenía 11,2 GB libres en C:.
