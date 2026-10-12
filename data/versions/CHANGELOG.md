# Historial de versiones del dataset

## v002 — 2026-10-01

Primera baseline con imágenes descargadas; configuración `configs/dataset_v002.yaml`.
Huella: `144323e165ecfbe86d29a3484e56fd91b8e8cb1dce1eef5f1d1355fea7bedb90`.

* **Fuente y clases:** solo FungiTastic, los mismos 10 géneros de v001.
* **Particiones oficiales:** Train (17.799 observaciones / 30.172 imágenes), ClosedSet-Val
  (3.452 / 6.290), ClosedSet-Test (4.039 / 7.159); sin `test_external`.
* **Integridad:** SHA-256 verificado para las 43.621 imágenes. Cero duplicados exactos entre
  particiones; prioridad de exclusión implementada: test > val > train.
* **Auditorías:** cero cruces sitio-día; un par perceptualmente similar entre train y test
  (Hamming 4), reportado sin exclusión. Se conservan y reportan 33 imágenes pequeñas.
* **Entrenamiento:** ResNet-18 preentrenada, selección por macro-F1 de observación en val;
  smoke y eval-only consultan solo val, y la corrida completa evalúa test una vez al final.
* **Limitaciones:** desbalance oficial, poco soporte de Morchella en val/test y similitud
  perceptual reportada; la partición oficial no elimina todos los posibles sesgos.


## v001 — 2026-09-26

Primera versión procesada (Semana 3). Configuración `configs/dataset_v001.yaml`, esquema `metadata_schema_v1`.
Huella de contenido: `6704c19c0fb8b7f92932c7a586b10d8e34449f0f13c211d1e1a08eb154c3dec7`.

* **Clases:** 10 géneros del alcance (Amanita, Auricularia, Ganoderma, Geastrum, Hericium, Lycoperdon, Morchella,
  Pleurotus, Trametes, Xylaria).
* **Desarrollo (train/val/test):** FungiTastic — 25.374 observaciones, 43.869 imágenes (metadatos; imágenes no descargadas).
* **Prueba externa (`test_external`):** iNaturalist Colombia (2.364 obs.), Mushroom Observer Colombia (984 obs.),
  MIND.Funga v2 (1.547 imágenes, solo imagen).
* **Excluidas:** DF20, GBIF, SiB Colombia, iNaturalist Open Data (motivos en la configuración).
* **Partición:** `grouped_hash` por sitio-día (fecha + celda de 0,01°), 70/15/15, *salt* `edge-ai-virtus/v001`;
  se conserva además la partición temporal oficial de FungiTastic (`split_temporal`).
* **Limitaciones:** sin clima (climatic.zip → HTTP 403); reglas de integridad de imagen definidas pero no ejecutadas.
