"""Builds ``data/manifests/dataset_inventory.csv`` and the metadata matrix.

Numbers come from the audit results (``reports/audit``) so the inventory cannot drift from the computed values.
The qualitative text (status, reasons, licence notes, relationships) is documentation and is written in Spanish;
column names and enumerated values (status, evidence tags) stay in English.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from common import RAW_FILES_CSV, _read_rows

ACCESS_DATE_FALLBACK = "2026-09-18"

COLUMNS = [
    "dataset_id", "dataset_name", "source_type", "official_url", "version", "access_date", "status", "status_reason",
    "recommended_next_step", "download_method", "local_data_available", "metadata_available",
    "images_reported", "images_reported_basis", "images_verified", "images_verified_basis",
    "observations_reported", "observations_verified", "taxa_reported", "species_verified", "genera_verified",
    "taxonomy_available", "observation_id_available", "image_id_available", "latitude_available", "longitude_available",
    "date_available", "substrate_available", "habitat_available", "climate_available", "license_available",
    "license_notes", "geographic_scope", "duplicate_risk", "relationship_to_other_datasets", "visual_suitability", "notes",
]
NA = "NA"

# table whose column roles define the availability flags of each dataset
PRIMARY_TABLE = {
    "fungitastic": "full_union", "df20": "train+public_test", "mind_funga_v2": "files_v2",
    "inaturalist_api": "random_sample200_research_photos", "gbif": "colombia_stillimage_sample900",
    "sib_macrohongosmitu": "occurrence", "sib_macrohongos_jbb_2023_1": "occurrence", "sib_hongos_col2023": "occurrence",
    "mushroom_observer": "fungi_all_observations",
}
FLAG_ROLES = {
    "taxonomy_available": ("species", "genus"), "observation_id_available": ("observation_id",),
    "image_id_available": ("image_id", "image_file"), "latitude_available": ("latitude",), "longitude_available": ("longitude",),
    "date_available": ("date",), "substrate_available": ("substrate",), "habitat_available": ("habitat",),
    "license_available": ("license",),
}


def _num(x: Any) -> Any:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return NA
    if isinstance(x, (np.integer, float, np.floating)) and float(x).is_integer():
        return int(x)
    return x


def _size(size: pd.DataFrame, ds: str, table: str, col: str) -> Any:
    r = size[(size.dataset_id == ds) & (size.table == table)]
    return _num(r.iloc[0][col]) if len(r) else NA


def _flag(missing: pd.DataFrame, ds: str, roles: tuple[str, ...]) -> str:
    tb = PRIMARY_TABLE.get(ds)
    m = missing[(missing.dataset_id == ds) & (missing.table == tb) & (missing.role.isin(roles)) & (missing.column_present == True)]  # noqa: E712
    if m.empty:
        return "no"
    best = m["pct_missing"].min()
    return "yes" if best <= 5 else ("partial" if best < 100 else "no")


def _dl_dates() -> dict[str, str]:
    d: dict[str, str] = {}
    for r in _read_rows(RAW_FILES_CSV):
        d[r["dataset_id"]] = min(d.get(r["dataset_id"], "9999"), r["download_date"])   # date of the first acquisition
    return d


def build_inventory(facts: dict[str, dict[str, Any]], size: pd.DataFrame, missing: pd.DataFrame, dups: pd.DataFrame) -> pd.DataFrame:
    f = lambda ds: facts.get(ds, {})  # noqa: E731
    dl = _dl_dates()
    ft, df20, mind, gb, ina, inod, mo = (f(k) for k in ("fungitastic", "df20", "mind_funga_v2", "gbif", "inaturalist_api", "inaturalist_open_data", "mushroom_observer"))
    dk_share = 100 * ft.get("countryCode_counts", {}).get("DK", 0) / max(1, sum(ft.get("countryCode_counts", {}).values()) or 1)
    df_dk_share = 100 * df20.get("countryCode_counts", {}).get("DK", 0) / max(1, sum(df20.get("countryCode_counts", {}).values()) or 1)

    gb_inat_pct = round(100 * (gb.get("fungi_colombia_stillimage_from_inaturalist_dataset") or 0) / max(1, gb.get("fungi_colombia_stillimage") or 1), 1)
    gb_mo_n = next((d["records"] for d in gb.get("top_datasets_colombia_stillimage", []) if d.get("title") == "Mushroom Observer"), NA)

    def sz(ds, table, col):
        return _size(size, ds, table, col)

    rows: list[dict[str, Any]] = []

    def add(**kw):
        row = {c: NA for c in COLUMNS}
        row.update(kw)
        ds = row["dataset_id"]
        row["access_date"] = dl.get(ds, ACCESS_DATE_FALLBACK) if ds in dl else ACCESS_DATE_FALLBACK
        if ds in PRIMARY_TABLE:
            for flag, roles in FLAG_ROLES.items():
                if row.get(flag, NA) == NA:
                    row[flag] = _flag(missing, ds, roles)
        rows.append(row)

    # ------------------------------------------------------------------ FungiTastic
    add(dataset_id="fungitastic", dataset_name="FungiTastic", source_type="benchmark_dataset",
        official_url="https://bohemianvra.github.io/FungiTastic/ | https://github.com/BohemianVRA/FungiTastic | https://www.kaggle.com/datasets/picekl/fungitastic",
        version="Kaggle v15 (2025-12-03); metadata.zip oficial Last-Modified 2025-12-05; repositorio commit fb65b7e (2025-10-15)",
        status="METADATA_AVAILABLE",
        status_reason="Se descargó y auditó localmente el metadata.zip oficial completo (13 CSV: completo, Mini y FewShot). Las imágenes (≈50 GB a 500 px en Kaggle) NO se descargaron.",
        recommended_next_step="Semana 3: decidir subconjunto/resolución mínimos y descargar con dataset/download.py (requiere wget) solo lo necesario; resolver la licencia antes de usar las imágenes.",
        download_method="Metadatos: URL oficial del script dataset/download.py (metadata.zip) descargada con requests (el script oficial requiere wget). Imágenes: script oficial (--subset m|fs|full --size 300|500|720|fullsize) o Kaggle (~50 GB).",
        local_data_available="metadatos completos (CSV, sin imágenes)", metadata_available="yes",
        images_reported=630000, images_reported_basis="SOURCE_REPORTED (artículo arXiv 2408.13632: '630k photographs'; sitio: '>600.000 imágenes')",
        images_verified=sz("fungitastic", "full_union", "n_images"), images_verified_basis="LOCAL_VERIFIED: filas únicas por 'filename' en los CSV (archivos de imagen NO comprobados)",
        observations_reported=350000, observations_verified=sz("fungitastic", "full_union", "n_observations"),
        taxa_reported=5000, species_verified=sz("fungitastic", "full_union", "n_species"), genera_verified=sz("fungitastic", "full_union", "n_genera"),
        climate_available="separate_file_not_verified", license_available="dataset_level_inconsistent",
        license_notes="Tres afirmaciones distintas: página del artículo arXiv 'CC BY 4.0' (no aclara si aplica al dataset); ficha de Kaggle 'CC BY-NC-SA 4.0'; repositorio (código) BSD-3-Clause. Los CSV no traen licencia ni autor por imagen. Los registros de origen en GBIF (Danish Mycological Society) declaran CC BY-NC 4.0. REQUIERE REVISIÓN antes de usar imágenes.",
        geographic_scope=f"Dinamarca ({dk_share:.1f}% de las imágenes con countryCode=DK) y otros países europeos; 20 años de registros (año {ft.get('year_min', NA)}–{ft.get('year_max', NA)})",
        duplicate_risk="ALTO con DF20 (misma colección); interno: ClosedSet ⊂ OpenSet repite archivos entre CSV (por diseño)",
        relationship_to_other_datasets="Amplía la colección de la que procede DF20 (Atlas de hongos daneses / GBIF). Ambos usan claves de ocurrencia de GBIF del dataset 'Danish Mycological Society, fungal records database' (verificado con 1 ID de cada CSV). Ver overlaps.csv.",
        visual_suitability="SUITABLE_CANDIDATE (fuera del contexto geográfico colombiano; licencia por resolver)",
        notes="Columna 'captions' (texto generado) presente, no auditada. elevation/landcover/biogeographicalRegion vienen en los CSV; clima y satélite son archivos aparte. category_id=-1 marca especies desconocidas en OpenSet.")

    add(dataset_id="fungitastic_aux_modalities", dataset_name="FungiTastic — clima, satélite y máscaras", source_type="benchmark_dataset_component",
        official_url="https://cmp.felk.cvut.cz/datagrid/FungiTastic/shared/download/", version="archivos oficiales (Last-Modified 2024-09-02 satélite; 2025-10-15 máscaras)",
        status="ACCESS_PROBLEM",
        status_reason="climatic.zip responde HTTP 403 en el servidor oficial (satellite_RGB/NIR y masks.zip responden 200 pero no se descargaron: 4,1 GB, 3,6 GB y 0,7 GB). Según el repositorio, el clima también está en la versión de Kaggle (requiere cuenta).",
        recommended_next_step="No se intenta eludir el 403. Semana 3: obtener climaticData vía Kaggle con cuenta propia o solicitarlo a los autores; decidir si el clima de FungiTastic o ERA5-Land (Semana posterior).",
        download_method="Script oficial: --climatic / --satellite / --masks", local_data_available="ninguno",
        metadata_available="no", climate_available="no_accesible", notes="Estado de los enlaces registrado en data/raw/fungitastic/api/official_download_head_probes.json.",
        visual_suitability="NOT_VERIFIED")

    # ------------------------------------------------------------------ DF20
    add(dataset_id="df20", dataset_name="Danish Fungi 2020 (DF20)", source_type="benchmark_dataset",
        official_url="https://sites.google.com/view/danish-fungi-dataset | https://github.com/BohemianVRA/DanishFungiDataset | https://arxiv.org/abs/2103.10107",
        version="DF20-metadata.zip (Last-Modified 2022-03-11), CSV 'PROD-2'; sitio: DF20 y DF20-Mini",
        status="METADATA_AVAILABLE",
        status_reason="Se descargó y auditó localmente DF20-metadata.zip (train + public test). Imágenes (6,4 GB a 300 px; 110 GB originales) NO descargadas. El enlace al artículo WACV suministrado tiene un guion mal escrito (devuelve 404); la URL corregida responde 200.",
        recommended_next_step="Semana 3: no combinar con FungiTastic; si se usa, excluir por observationID las observaciones ya presentes en FungiTastic o preferir uno de los dos.",
        download_method="Metadatos: http://ptak.felk.cvut.cz/plants/DanishFungiDataset/DF20-metadata.zip; imágenes: DF20-300px.tar.gz (6,4 GB) o DF20-train_val.tar.gz (110 GB)",
        local_data_available="metadatos completos (CSV, sin imágenes)", metadata_available="yes",
        images_reported=295938, images_reported_basis="SOURCE_REPORTED (artículo, Tabla 2: 295.938 imágenes, 1.604 especies, 566 géneros)",
        images_verified=sz("df20", "train+public_test", "n_images"), images_verified_basis="LOCAL_VERIFIED: filas únicas por ImageUniqueID (archivos NO comprobados)",
        observations_reported=NA, observations_verified=sz("df20", "train+public_test", "n_observations"),
        taxa_reported=1604, species_verified=sz("df20", "train+public_test", "n_species"), genera_verified=sz("df20", "train+public_test", "n_genera"),
        climate_available="no", license_available="dataset_level_unclear+author_per_record",
        license_notes="El artículo (arXiv) indica CC BY 4.0; el sitio no declara licencia del dataset; el CSV incluye rightsHolder (autor) por imagen. Los registros de origen en GBIF declaran CC BY-NC 4.0.",
        geographic_scope=f"Dinamarca ({df_dk_share:.1f}% countryCode=DK) y otros países europeos; años {df20.get('year_min', NA)}–{df20.get('year_max', NA)}",
        duplicate_risk="ALTO con FungiTastic (ver overlaps.csv); interno: mismas observaciones en train y test (ver overlaps.csv)",
        relationship_to_other_datasets="Predecesor histórico de FungiTastic (misma fuente: Atlas de hongos daneses vía GBIF).",
        visual_suitability="SUITABLE_CANDIDATE (contexto europeo; solapado con FungiTastic)",
        notes="El artículo declara 1.604 especies; los CSV locales contienen menos valores distintos de 'species' (ver size_summary.csv).")

    # ------------------------------------------------------------------ MIND.Funga
    add(dataset_id="mind_funga_v2", dataset_name="MIND.Funga App — dataset de imágenes de macrohongos neotropicales (v2)", source_type="image_dataset",
        official_url="https://data.mendeley.com/datasets/sfrbdjvxcc/2 | https://mindfunga.ufsc.br/",
        version=f"Mendeley Data v{mind.get('version', 2)} ({str(mind.get('publish_date', ''))[:10]}), DOI {mind.get('doi', '10.17632/sfrbdjvxcc.2')}",
        status="REQUIRES_REVIEW",
        status_reason=(f"Índice completo de archivos ({mind.get('n_files_api', NA)} JPG, con SHA-256 publicado por la fuente) y nombres de carpeta (taxón) obtenidos sin descargar imágenes. "
                       f"Discrepancia: {mind.get('n_files_api', NA)} archivos frente a 17.467 imágenes declaradas; sin observación, GPS, fecha ni sustrato; licencia CC BY-NC 3.0; "
                       "posibles variantes de fondo ('branco'/'verde'). La página de Mendeley y su API responden con un desafío de Cloudflare (HTTP 403) a las solicitudes de Python (requests); curl sí es atendido (ver informe, limitaciones)."),
        recommended_next_step="Contactar al proyecto (mind.funga@gmail.com, según su sitio) para aclarar agrupación por observación, variantes de fondo y la discrepancia entre archivos e imágenes declaradas; descargar el ZIP (3,2 GB) manualmente desde el navegador solo si se decide usarlo.",
        download_method="API pública de Mendeley Data (JSON de la ficha, con curl) + lectura del directorio central del ZIP oficial por HTTP Range (curl). ZIP completo: 3,2 GB (no descargado).",
        local_data_available="índice de archivos + nombres de carpeta (sin imágenes)", metadata_available="partial",
        images_reported=17467, images_reported_basis="SOURCE_REPORTED (descripción de Mendeley y sitio del proyecto)",
        images_verified=sz("mind_funga_v2", "files_v2", "n_images"), images_verified_basis="LOCAL_VERIFIED: entradas del índice de archivos de la API y del directorio central del ZIP (imágenes NO descargadas)",
        observations_reported=NA, observations_verified=NA, taxa_reported=">500 taxones (Mendeley)",
        species_verified=sz("mind_funga_v2", "files_v2", "n_species"), genera_verified=sz("mind_funga_v2", "files_v2", "n_genera"),
        taxonomy_available="partial", observation_id_available="no", image_id_available="partial", latitude_available="no", longitude_available="no",
        date_available="no", substrate_available="no", habitat_available="no", climate_available="no", license_available="dataset_level",
        license_notes="CC BY-NC 3.0 (Mendeley Data): uso no comercial con atribución. Imágenes de fuentes mixtas (grupos de Facebook, ciencia ciudadana, artículos, socios); la procedencia por imagen no está en los metadatos.",
        geographic_scope="Neotrópico (imágenes desde Brasil y otros países de América, p. ej. nombres de archivo con Ecuador/Perú); sin coordenadas",
        duplicate_risk="MEDIO-ALTO: hashes SHA-256 idénticos entre archivos (reportados por la fuente) y variantes 'branco/verde'; ver duplicates.csv",
        relationship_to_other_datasets="Independiente de las demás fuentes; puede solapar con iNaturalist/GBIF (fotografías de ciencia ciudadana) — no verificable sin observación/URL.",
        visual_suitability="REQUIRES_REVIEW (contexto neotropical relevante; sin agrupación por observación)",
        notes="species_verified/genera_verified se DERIVAN del nombre de carpeta (solo binomios sin 'sp./aff./cf.'); classes = carpetas. Ver adapters.audit_mind_funga.")

    # ------------------------------------------------------------------ iNaturalist
    add(dataset_id="inaturalist_api", dataset_name="iNaturalist — plataforma y API v1", source_type="platform_api",
        official_url="https://www.inaturalist.org/ | https://api.inaturalist.org/v1/docs/", version="API v1 (consultas del 2026-09-18)",
        status="REMOTE_QUERYABLE",
        status_reason="Se ejecutaron decenas de consultas pequeñas (conteos, licencias, conteos por especie y una muestra aleatoria de 200 observaciones) guardadas como instantáneas. No se descargaron fotos.",
        recommended_next_step="Semana 3: para construir un subconjunto colombiano usar Open Data/AWS (no la API) o exportaciones; filtrar por licencia de la foto y grado de calidad.",
        download_method="GET /observations, /observations/species_counts (place_id=7196 Colombia, taxon_id=47170 Fungi); límite recomendado ~60 solicitudes/min",
        local_data_available="instantáneas JSON de consultas + muestra de 200 observaciones", metadata_available="partial",
        images_reported=NA, images_reported_basis="NOT_VERIFIED (la API cuenta observaciones, no fotos; en la muestra: 2,34 fotos/observación)",
        images_verified=NA, observations_reported=ina.get("fungi_colombia_observations", NA), observations_verified=NA,
        taxa_reported=ina.get("species_colombia_all", NA), species_verified=NA, genera_verified=NA,
        climate_available="no", license_available="per_photo", image_id_available="yes",
        license_notes=f"Licencia por foto (CC0, CC BY, CC BY-NC, CC BY-SA, CC BY-ND, CC BY-NC-SA, CC BY-NC-ND o todos los derechos reservados). Fungi Colombia grado de investigación: {ina.get('fungi_colombia_research_grade', NA)} observaciones, de las cuales {ina.get('fungi_colombia_research_photos_licensed', NA)} con fotos con licencia abierta (photo_licensed=true).",
        geographic_scope="Mundial; filtro Colombia (place_id 7196): observaciones de hongos = ver observations_reported",
        duplicate_risk="ALTO respecto a GBIF (GBIF integra el dataset 'iNaturalist Research-grade Observations') y respecto a iNaturalist Open Data (misma fuente)",
        relationship_to_other_datasets=f"Misma plataforma que inaturalist_open_data; {gb_inat_pct}% de los registros de GBIF con imagen de hongos en Colombia provienen de este dataset.",
        visual_suitability="SUITABLE_CANDIDATE (para prueba colombiana; requiere filtrar licencias)",
        notes="observations_reported = TODAS las calidades (hongos, Colombia). Muestra: 200 observaciones aleatorias grado de investigación con fotos, NO representativa del total.")

    add(dataset_id="inaturalist_open_data", dataset_name="iNaturalist Open Data (GitHub + AWS Open Data)", source_type="open_data_bucket",
        official_url="https://github.com/inaturalist/inaturalist-open-data | https://registry.opendata.aws/inaturalist-open-data/",
        version=f"instantánea mensual del bucket (objetos con Last-Modified 2026-08-27); s3://inaturalist-open-data",
        status="REMOTE_QUERYABLE",
        status_reason=("Se descargó solo taxa.csv.gz (40 MB) y fragmentos iniciales (2 y 3 MB) de observations/photos para verificar el esquema. "
                       "observations.csv.gz (13,1 GB) y photos.csv.gz (20,1 GB) NO se descargaron (magnitud extraordinaria); no contienen país, solo lat/lon."),
        recommended_next_step="Semana 3: filtrar por Colombia requiere unir observations+photos+taxa y un polígono/caja de Colombia (sin columna de país) — decidir si se justifica frente a la API o GBIF; no bajar imágenes masivamente.",
        download_method="HTTPS público o `aws s3 ... --no-sign-request` (sin cuenta AWS); imágenes en photos/<id>/{original,large,medium,small,thumb,square}.<ext>",
        local_data_available="taxonomía completa (taxa.csv.gz) + fragmentos de esquema", metadata_available="partial",
        images_reported=70000000, images_reported_basis="SOURCE_REPORTED (README: 'over 70 million photos', todos los taxones, no solo hongos)",
        images_verified=NA, observations_reported=NA, observations_verified=NA, taxa_reported=NA, species_verified=NA, genera_verified=NA,
        taxonomy_available="yes", observation_id_available="yes", image_id_available="yes", latitude_available="yes", longitude_available="yes",
        date_available="yes", substrate_available="no", habitat_available="no", climate_available="no", license_available="per_photo",
        license_notes="Solo fotos con licencia abierta según el registro AWS ('Creative Commons o Dominio Público, varía por imagen'); columna license por foto (CC0, CC-BY, CC-BY-NC, …). Mantener autoría (observer) y licencia.",
        geographic_scope="Mundial (coordenadas, sin campo de país)", duplicate_risk="ALTO con iNaturalist API y con GBIF (misma fuente)",
        relationship_to_other_datasets="Subconjunto de iNaturalist con licencia abierta; publicado también en GBIF como 'iNaturalist Research-grade Observations'.",
        visual_suitability="SUITABLE_CANDIDATE (fuente principal de fotos abiertas; requiere unir tablas)",
        notes=f"Taxones fúngicos en taxa.csv: {inod.get('fungi_taxa_total_rows', NA)} filas ({inod.get('fungi_taxa_active', NA)} activos; {inod.get('fungi_active_by_rank', {}).get('species', NA)} de rango especie). Es taxonomía, NO conteo de observaciones.")

    # ------------------------------------------------------------------ SiB Colombia
    add(dataset_id="sib_colombia_portal", dataset_name="SiB Colombia — portal de datos", source_type="portal",
        official_url="https://biodiversidad.co/ | https://biodiversidad.co/data/", version="NA", status="DOCUMENTED_ONLY",
        status_reason="Portal informativo (registros biológicos, listas, fichas, IPT). No se usó una API propia: los datos se auditaron por los tres recursos IPT indicados y, para una visión nacional, por GBIF (publishingCountry=CO).",
        recommended_next_step="Semana 3: si hace falta más cobertura, enumerar recursos IPT/GBIF publicados desde Colombia con multimedia (ver gbif) y revisar sus licencias.",
        download_method="Vía IPT (DwC-A) o vía GBIF", local_data_available="ninguno (ver recursos sib_*)", metadata_available="no",
        license_available="per_resource", geographic_scope="Colombia", duplicate_risk="ALTO con GBIF (el SiB publica en GBIF)",
        relationship_to_other_datasets="Los recursos publicados por el SiB aparecen en GBIF cuando están registrados allí.",
        visual_suitability="NOT_VERIFIED", notes=f"GBIF: registros de hongos con imagen publicados desde Colombia: {gb.get('fungi_published_by_colombia_stillimage', NA)} (SOURCE_REPORTED).")

    sib_specs = {
        "sib_macrohongosmitu": ("Exploración y Valoración de la Diversidad Fúngica en cercanías de la comunidad Mitú-Cachivera", "https://ipt.biodiversidad.co/permisos/resource?r=macrohongosmitu",
                                "1.1 (publicada 2026-01-01)", "CC BY 4.0 (recurso); sin licencia por registro"),
        "sib_macrohongos_jbb_2023_1": ("Organismos asociados a las plantas: Fungario del Jardín Botánico José Celestino Mutis 2023-1", "https://ipt.biodiversidad.co/permisos/resource?r=macrohongos_2023-1",
                                       "1.0 (publicada 2023-12-22)", "CC BY-NC 4.0 (recurso); sin licencia por registro"),
        "sib_hongos_col2023": ("Hongos de Colombia (proyecto)", "https://ipt.biodiversidad.co/permisos/resource?r=hongos_col2023",
                               "1.4 (publicada 2024-02-06)", "CC0 1.0 (recurso); sin licencia por registro"),
    }
    for ds, (title, url, ver, lic) in sib_specs.items():
        s = f(ds)
        n_rec = s.get("n_records", NA)
        add(dataset_id=ds, dataset_name=f"SiB Colombia IPT — {title}", source_type="ipt_resource_dwca", official_url=url, version=ver,
            status="AVAILABLE_LOCAL",
            status_reason=f"Se descargó y auditó el DwC-A completo ({n_rec} registros). El archivo NO incluye extensión multimedia y la columna associatedMedia está vacía en {n_rec - s.get('n_associatedMedia_nonempty', 0) if isinstance(n_rec, int) else NA} de {n_rec} registros: no hay fotografías enlazadas.",
            recommended_next_step="Usar solo como referencia taxonómica/distribución. Si el recurso tiene fotografías, deben solicitarse a la institución (fuera del DwC-A).",
            download_method="IPT: archive.do?r=<recurso> (Darwin Core Archive)", local_data_available="registros completos (DwC-A); sin imágenes", metadata_available="yes",
            images_reported=NA, images_reported_basis="NOT_VERIFIED (la descripción del recurso menciona fotografías, sin enlaces en los datos)" if ds == "sib_macrohongosmitu" else "NOT_VERIFIED",
            images_verified=0, images_verified_basis="LOCAL_VERIFIED: 0 registros con multimedia/associatedMedia",
            observations_reported=n_rec, observations_verified=n_rec, taxa_reported=NA,
            species_verified=s.get("n_species_derived", NA), genera_verified=s.get("n_genera", NA),
            image_id_available="no", climate_available="no", license_available="resource_level",
            license_notes=lic, geographic_scope="Colombia",
            duplicate_risk="BAJO internamente (ver duplicates.csv); posible solapamiento con GBIF si el recurso está registrado allí",
            relationship_to_other_datasets="Recurso IPT del SiB Colombia; GBIF ya integra parte del SiB (ver notes).",
            visual_suitability="NOT_SUITABLE (sin imágenes en los datos)",
            notes=({"sib_macrohongos_jbb_2023_1": "Coordenadas con coma decimal (ver duplicates.csv) y fechas no ISO.",
                    "sib_macrohongosmitu": f"Sin coordenadas; taxón solo en texto libre (organismName) y fecha en year/month/day; verbatimEventDate contradice year/month/day en {s.get('verbatimEventDate_disagrees_with_year_month_day', NA)} de {s.get('verbatimEventDate_comparable', NA)} registros. Especie/género/familia/fecha DERIVADOS.",
                    "sib_hongos_col2023": "Coordenadas corruptas por separadores decimales/de miles (ver missing_values.csv) y fechas d/m/aaaa."}.get(ds, ""))
                  + " Registro en GBIF: " + ("ver gbif.registry_search_results." if ds != "sib_hongos_col2023" else "la página IPT indica 'no registrado en GBIF'."))

    # ------------------------------------------------------------------ GBIF
    add(dataset_id="gbif", dataset_name="GBIF — Global Biodiversity Information Facility", source_type="aggregator_api",
        official_url="https://www.gbif.org/ | https://api.gbif.org/v1/occurrence/search", version="API v1 (consultas del 2026-09-18)",
        status="REMOTE_QUERYABLE",
        status_reason="La API pública responde; la web gbif.org devuelve 403 a clientes automatizados (no necesaria). Se guardaron decenas de instantáneas (conteos, facetas por dataset/licencia/base del registro, 900 registros de muestra). No se descargaron imágenes ni descargas masivas (requieren cuenta).",
        recommended_next_step="Semana 3: si se usa GBIF, generar una descarga oficial con DOI (requiere cuenta) filtrada por país=CO, Fungi y mediaType=StillImage, y conservar URL/licencia/autor por imagen; excluir los datasets que ya se audite aparte (iNaturalist, Mushroom Observer).",
        download_method="GET /v1/occurrence/search (kingdomKey=5, country=CO, mediaType=StillImage; facetas); descarga oficial por API requiere autenticación",
        local_data_available="instantáneas JSON (conteos y 900 registros de muestra)", metadata_available="partial",
        images_reported=NA, images_reported_basis="NOT_VERIFIED (GBIF cuenta registros, no imágenes; en la muestra: ver audit_facts.json)",
        images_verified=NA, observations_reported=gb.get("fungi_colombia_stillimage", NA), observations_verified=NA, taxa_reported=NA,
        species_verified=NA, genera_verified=NA, climate_available="no", license_available="per_record_and_per_media",
        image_id_available="yes (URL del medio)",
        license_notes=f"Licencia por registro/medio. Hongos de Colombia con imagen: {gb.get('license_counts_colombia_stillimage', {})}. Conservar URL original, autoría (creator/rightsHolder) y licencia.",
        geographic_scope="Mundial; filtro country=CO",
        duplicate_risk="ALTO: agrega iNaturalist, Mushroom Observer y colecciones (NYBG, USNM, MNHN…); ver overlaps.csv",
        relationship_to_other_datasets=f"Agregador: incluye iNaturalist Research-grade ({gb_inat_pct}% de hongos con imagen en Colombia), Mushroom Observer y herbarios. También aloja la fuente de DF20/FungiTastic (Danish Mycological Society).",
        visual_suitability="SUITABLE_CANDIDATE (cobertura colombiana; muchas fotos son de especímenes de herbario)",
        notes=f"observations_reported = registros GBIF de hongos con StillImage en Colombia ({gb.get('fungi_colombia_stillimage', NA)} según la consulta del 2026-09-18); mundo: {gb.get('fungi_world_with_stillimage', NA)} con imagen de {gb.get('fungi_world_records', NA)} registros de hongos. 'Registro' ≠ observación única ≠ imagen.")

    # ------------------------------------------------------------------ Mushroom Observer
    add(dataset_id="mushroom_observer", dataset_name="Mushroom Observer", source_type="platform_csv_dumps",
        official_url="https://mushroomobserver.org/ | https://github.com/MushroomObserver/mushroom-observer/blob/main/README_API.md", version="volcados CSV nocturnos (Last-Modified 2026-09-18)",
        status="METADATA_AVAILABLE",
        status_reason="Se descargaron los 6 CSV oficiales (observations, images_observations, images, names, name_classifications, locations; ~250 MB) que MO recomienda usar en lugar de la API para grandes volúmenes. Imágenes NO descargadas.",
        recommended_next_step="Semana 3: reconciliar nombres (sinónimos/deprecated) y, si se usa, descargar solo las imágenes necesarias respetando los límites de MO.",
        download_method="https://mushroomobserver.org/<tabla>.csv (dump nocturno oficial); la API no está pensada para extracción masiva y limita a ~20 solicitudes/min",
        local_data_available="metadatos completos (CSV, sin imágenes)", metadata_available="yes",
        images_reported=NA, images_reported_basis="NOT_VERIFIED", images_verified=sz("mushroom_observer", "fungi_all_images", "n_images"),
        images_verified_basis="LOCAL_VERIFIED: pares imagen–observación de hongos (Kingdom=Fungi) en los CSV (archivos NO comprobados)",
        observations_reported=NA, observations_verified=sz("mushroom_observer", "fungi_all_observations", "n_observations"),
        taxa_reported=NA, species_verified=sz("mushroom_observer", "fungi_all_observations", "n_species"),
        genera_verified=sz("mushroom_observer", "fungi_all_observations", "n_genera"),
        substrate_available="no", habitat_available="no", climate_available="no", license_available="per_image", image_id_available="yes",
        license_notes="Licencia y titular por imagen en images.csv (Creative Commons: Wikipedia-compatible, NC, BY, BY-SA, etc.); imágenes con ok_for_export=0 no deben usarse. Los registros de MO en GBIF figuran como CC BY-NC 4.0.",
        geographic_scope="Mundial (mayoría EE. UU.); Colombia: ver reports/audit/size_summary.csv (tabla fungi_colombia_*)",
        duplicate_risk="ALTO con GBIF (MO es un dataset de GBIF)",
        relationship_to_other_datasets=f"Publicado también en GBIF ('Mushroom Observer', {gb_mo_n} registros con imagen en Colombia al 2026-09-18).",
        visual_suitability="SUITABLE_CANDIDATE (complementario; identificación comunitaria con votos)",
        notes="Especie = nombres con rank 400; género/país DERIVADOS. Calidad de identificación: vote_cache (consenso), no verificable como verdad terreno. Sustrato/hábitat no están en los CSV.")

    out = pd.DataFrame(rows, columns=COLUMNS)
    return out.fillna(NA)


def build_metadata_matrix(inv: pd.DataFrame, missing: pd.DataFrame) -> pd.DataFrame:
    """Dataset x metadata-role matrix: % missing on the primary table (NA if the role is not present)."""
    roles = ["species", "genus", "family", "observation_id", "image_id", "latitude", "longitude", "date", "country", "substrate",
             "habitat", "license", "author"]
    rows = []
    for ds in inv["dataset_id"]:
        tb = PRIMARY_TABLE.get(ds)
        r: dict[str, Any] = {"dataset_id": ds, "primary_table": tb or NA}
        m = missing[(missing.dataset_id == ds) & (missing.table == tb)] if tb else missing.iloc[0:0]
        for role in roles:
            x = m[m.role == role]
            r[f"{role}_pct_missing"] = float(x.iloc[0]["pct_missing"]) if len(x) and bool(x.iloc[0]["column_present"]) else NA
        rows.append(r)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- per-source YAML manifests
ACQUISITION_PARAMS: dict[str, dict[str, Any]] = {
    "fungitastic": {"modalities_downloaded": ["metadata"], "images_downloaded": False, "resolution_requested": "ninguna",
                    "subset_requested": "todos los CSV de metadata.zip (completo, Mini, FewShot)", "command": "python scripts/data/acquire_datasets.py fungitastic"},
    "fungitastic_aux_modalities": {"modalities_downloaded": [], "images_downloaded": False, "resolution_requested": "ninguna",
                                   "probes": "HEAD a climatic.zip, masks.zip, satellite_RGB.zip, satellite_NIR.zip"},
    "df20": {"modalities_downloaded": ["metadata"], "images_downloaded": False, "resolution_requested": "ninguna",
             "subset_requested": "DF20-metadata.zip (train + public test)", "command": "python scripts/data/acquire_datasets.py df20"},
    "mind_funga_v2": {"modalities_downloaded": ["índice de archivos (JSON) y directorio central del ZIP (CSV)"], "images_downloaded": False,
                      "resolution_requested": "ninguna", "command": "python scripts/data/acquire_datasets.py mind_funga_v2 (usa curl; requests recibe un desafío Cloudflare)"},
    "inaturalist_api": {"modalities_downloaded": ["respuestas JSON de la API"], "images_downloaded": False, "taxon_id": 47170, "place_id": 7196,
                        "command": "python scripts/data/acquire_datasets.py inaturalist_api"},
    "inaturalist_open_data": {"modalities_downloaded": ["taxa.csv.gz completo", "fragmentos iniciales de observations/photos (solo esquema)"],
                              "images_downloaded": False, "command": "python scripts/data/acquire_datasets.py inaturalist_open_data"},
    "sib_colombia_portal": {"modalities_downloaded": []},
    "sib_macrohongosmitu": {"modalities_downloaded": ["DwC-A"], "images_downloaded": False, "ipt_resource": "macrohongosmitu"},
    "sib_macrohongos_jbb_2023_1": {"modalities_downloaded": ["DwC-A"], "images_downloaded": False, "ipt_resource": "macrohongos_2023-1"},
    "sib_hongos_col2023": {"modalities_downloaded": ["DwC-A"], "images_downloaded": False, "ipt_resource": "hongos_col2023"},
    "gbif": {"modalities_downloaded": ["respuestas JSON de la API de ocurrencias y del registro"], "images_downloaded": False,
             "filters": {"kingdomKey": 5, "country": "CO", "mediaType": "StillImage"}, "command": "python scripts/data/acquire_datasets.py gbif"},
    "mushroom_observer": {"modalities_downloaded": ["volcados CSV nocturnos"], "images_downloaded": False,
                          "command": "python scripts/data/acquire_datasets.py mushroom_observer"},
}


def write_source_manifests(inv: pd.DataFrame, out_dir) -> None:
    """One small YAML per source, generated from the inventory + raw_files.csv + remote_queries.csv."""
    import yaml
    from common import REMOTE_QUERIES_CSV

    raws = _read_rows(RAW_FILES_CSV)
    queries = _read_rows(REMOTE_QUERIES_CSV)
    out_dir.mkdir(parents=True, exist_ok=True)
    for _, r in inv.iterrows():
        ds = r["dataset_id"]
        own = [x for x in raws if x["dataset_id"] == ds]
        files = [{"path": x["relative_path"], "size_bytes": int(x["size_bytes"]), "sha256": x["sha256"], "source_url": x["source_url"],
                  "download_date": x["download_date"]} for x in own if "/api/" not in x["relative_path"]]
        qs = [q for q in queries if q["dataset_id"] == ds]
        doc = {
            "dataset": {"id": ds, "name": r["dataset_name"], "type": r["source_type"], "status": r["status"], "status_reason": r["status_reason"]},
            "source": {"official_urls": [u.strip() for u in str(r["official_url"]).split("|")], "version": r["version"]},
            "accessed": r["access_date"], "download_method": r["download_method"],
            "parameters": ACQUISITION_PARAMS.get(ds, {}),
            "local_files": files or "ninguno (ver remote_queries)",
            "api_snapshots": {"count": len(qs), "manifest": "data/manifests/remote_queries.csv",
                              "folder": f"data/raw/{ds}/api/"} if qs else "ninguno",
            "counts": {"images_reported": r["images_reported"], "images_verified": r["images_verified"],
                       "observations_reported": r["observations_reported"], "observations_verified": r["observations_verified"]},
            "license": r["license_notes"], "known_relationships": r["relationship_to_other_datasets"],
            "next_step": r["recommended_next_step"], "notes": r["notes"],
        }
        (out_dir / f"{ds}.yaml").write_text(yaml.safe_dump(doc, allow_unicode=True, sort_keys=False, width=110), encoding="utf-8")
