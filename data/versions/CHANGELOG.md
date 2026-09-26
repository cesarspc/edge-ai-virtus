# Historial de versiones del dataset

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
