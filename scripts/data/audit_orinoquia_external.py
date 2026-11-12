"""Audit of the Orinoquía GBIF download as a candidate `test_external` set for the v002 class list.

Reads the filtered Darwin Core Archive in ``data/raw/gbif_orinoquia/dwca/`` (158 macromycete genera of the
Orinoquía species list), intersects it with the 10 genera of ``configs/dataset_v002.yaml`` and reports the support
that is actually usable for a visual evaluation: occurrences with at least one ``StillImage`` whose ``identifier``
is an HTTP URL. It also measures the overlap with the ``test_external`` split already published in ``v001``
(iNaturalist CO / Mushroom Observer CO), because GBIF is an aggregator and re-publishes both.

Nothing is modified in the dataset versions: this only writes ``reports/audit/orinoquia_external_candidate.{csv,md}``
and registers the raw files in ``data/manifests/raw_files.csv``.

    python scripts/data/audit_orinoquia_external.py
    python scripts/data/audit_orinoquia_external.py --no-register
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import AUDIT_DIR, RAW_DIR, REPO_ROOT, log, register_raw_file, rel, setup_logging  # noqa: E402

DATASET_ID = "gbif_orinoquia"
SRC = RAW_DIR / DATASET_ID
DWCA = SRC / "dwca"
GBIF_DOWNLOAD = "GBIF0007586-260221153910048"
SOURCE_URL = "https://www.gbif.org/occurrence/download/GBIF0007586-260221153910048"
CONFIG = REPO_ROOT / "configs" / "dataset_v002.yaml"
OUT_CSV = AUDIT_DIR / "orinoquia_external_candidate.csv"
OUT_MD = AUDIT_DIR / "orinoquia_external_candidate.md"

# Files registered in raw_files.csv. The per-dataset EML in dwca/dataset/ (60 files) and the unfiltered copy are
# left out on purpose: the first is reproducible metadata of the archive, the second is not an input of any version.
_DWCA_NOTE = f"Darwin Core Archive de la descarga GBIF {GBIF_DOWNLOAD}, filtrado a los 158 géneros de macromicetos de la Orinoquía"
CORE_FILES = {
    "dwca/occurrence.txt": f"{_DWCA_NOTE}: ocurrencias (230 columnas DwC)",
    "dwca/multimedia.txt": f"{_DWCA_NOTE}: extensión multimedia (URL, licencia y autoría por imagen)",
    "dwca/meta.xml": f"{_DWCA_NOTE}: descriptor del archivo",
    "dwca/metadata.xml": f"{_DWCA_NOTE}: metadatos EML de la descarga",
    "dwca/citations.txt": f"{_DWCA_NOTE}: citas y DOI de cada dataset de origen",
    "dwca/rights.txt": f"{_DWCA_NOTE}: licencias declaradas por dataset de origen",
    "Lista_especies_macrohongos_orinoquia.xlsx": "Lista maestra de 330 especies / 158 géneros usada como criterio del filtrado",
    "INFORME_FILTRADO_MACROMICETOS_GBIF0007586.txt": "Informe del filtrado de la descarga original a macromicetos",
}

# Datasets of origin that v001 already audits on their own; their records are double counting inside GBIF.
AGGREGATED = {
    "50c9509d-22c7-4a22-a47d-8c48425ef4a7": "iNaturalist Research-grade Observations",
    "d714382d-5890-4234-ae81-696eeb53658a": "Mushroom Observer",
}


def mil(n: int) -> str:
    """Thousands separated with a dot, as in the rest of the Spanish reports."""
    return f"{n:,}".replace(",", ".")


def dataset_title(key: str) -> str:
    """Title of a GBIF dataset, from AGGREGATED or from the snapshots already stored under data/raw/gbif/api/."""
    if key in AGGREGATED:
        return AGGREGATED[key]
    snap = RAW_DIR / "gbif" / "api" / f"dataset_{key}.json"
    if snap.is_file():
        return json.loads(snap.read_text(encoding="utf-8")).get("title", key)
    return "título no disponible localmente"


def read_dwc(name: str, **kw) -> pd.DataFrame:
    """Darwin Core text files from GBIF are tab separated and unquoted."""
    return pd.read_csv(DWCA / name, sep="\t", dtype=str, low_memory=False, quoting=3, **kw)


def classes_and_synonyms() -> tuple[list[str], dict[str, str]]:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    return cfg["classes"], cfg.get("genus_synonyms", {}) or {}


def usable_images(occ: pd.DataFrame, mm: pd.DataFrame) -> pd.DataFrame:
    """One row per (occurrence, image) with an HTTP StillImage URL."""
    img = mm[mm["type"].fillna("").str.casefold().eq("stillimage")
             & mm["identifier"].fillna("").str.startswith("http")]
    return img.merge(occ, on="gbifID", how="inner", suffixes=("_media", "_occ"))


def v001_external_ids() -> set[str] | None:
    """Observation ids of the iNaturalist/Mushroom Observer records already in v001's test_external."""
    path = REPO_ROOT / "data" / "processed" / "v001" / "observations.csv"
    if not path.is_file():
        return None
    obs = pd.read_csv(path, low_memory=False, usecols=["source_id", "source_observation_id", "split"])
    ext = obs[(obs["split"] == "test_external") & obs["source_id"].isin(["inaturalist_co", "mushroom_observer_co"])]
    return set(ext["source_observation_id"].astype(str))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-register", action="store_true", help="do not touch data/manifests/raw_files.csv")
    args = ap.parse_args()
    setup_logging()
    if not DWCA.is_dir():
        raise SystemExit(f"{rel(DWCA)} not found: place the filtered Darwin Core Archive there")

    classes, syn = classes_and_synonyms()
    occ = read_dwc("occurrence.txt")
    mm = read_dwc("multimedia.txt")
    occ["genus_norm"] = occ["genus"].replace(syn)
    log.info("download %s: %d occurrences, %d multimedia rows", GBIF_DOWNLOAD, len(occ), len(mm))

    sel = occ[occ["genus_norm"].isin(classes)].copy()
    j = usable_images(sel, mm)
    with_image = set(j["gbifID"])

    per_class = (pd.DataFrame({"genus": classes})
                 .merge(sel.groupby("genus_norm").size().rename("occurrences").reset_index()
                        .rename(columns={"genus_norm": "genus"}), on="genus", how="left")
                 .merge(j.groupby("genus_norm").agg(observations_with_image=("gbifID", "nunique"),
                                                    images=("identifier", "size")).reset_index()
                        .rename(columns={"genus_norm": "genus"}), on="genus", how="left")
                 .fillna(0))
    for c in ("occurrences", "observations_with_image", "images"):
        per_class[c] = per_class[c].astype(int)
    per_class["label_id"] = range(len(classes))
    per_class = per_class[["label_id", "genus", "occurrences", "observations_with_image", "images"]]
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    per_class.to_csv(OUT_CSV, index=False, lineterminator="\n")

    s = sel[sel["gbifID"].isin(with_image)]
    lic = s["license"].value_counts()
    prov = s["datasetKey"].value_counts()
    states = s["stateProvince"].value_counts(dropna=False)
    years = pd.to_numeric(s["year"], errors="coerce").dropna().astype(int)
    known = v001_external_ids()
    overlap = None
    if known is not None:
        ids = s["occurrenceID"].fillna("").astype(str)
        bare = ids.str.rsplit("/", n=1).str[-1]
        overlap = int((ids.isin(known) | bare.isin(known)).sum())

    n_obs, n_img = len(with_image), len(j)
    present = int((per_class["observations_with_image"] > 0).sum())
    weak = per_class[(per_class["observations_with_image"] > 0) & (per_class["observations_with_image"] < 10)]

    md = [f"# Orinoquía (GBIF {GBIF_DOWNLOAD}) como candidata a `test_external`", "",
          f"> Generado por `scripts/data/audit_orinoquia_external.py`; fuente `{rel(DWCA)}`.", "",
          f"Descarga filtrada: **{mil(len(occ))} ocurrencias** y **{mil(len(mm))} registros multimedia** "
          "(158 géneros de macromicetos de la Orinoquía).",
          f"Intersección con las {len(classes)} clases de `{rel(CONFIG)}`: **{mil(len(sel))} ocurrencias**, "
          f"de las cuales **{mil(n_obs)} tienen al menos una imagen con URL** y suman **{mil(n_img)} imágenes**.", "",
          f"Clases con soporte > 0: **{present}/{len(classes)}**. "
          f"Clases con menos de 10 observaciones: {', '.join(weak['genus']) or 'ninguna'}. "
          f"Clases vacías: {', '.join(per_class.loc[per_class['observations_with_image'] == 0, 'genus']) or 'ninguna'}.",
          "", "## Soporte por clase", "",
          "| `label_id` | Género | Ocurrencias | Observaciones con imagen | Imágenes |", "|---:|---|---:|---:|---:|"]
    md += [f"| {r.label_id} | {r.genus} | {r.occurrences} | {r.observations_with_image} | {r.images} |"
           for r in per_class.itertuples()]
    md += ["", "## Procedencia, licencia y cobertura", "",
           f"* Rango de años: {years.min()}–{years.max()}." if len(years) else "* Sin año registrado.",
           f"* Departamentos: " + "; ".join(f"{k} ({v})" for k, v in states.items()) + ".",
           "* Licencias declaradas en la ocurrencia: " + "; ".join(f"{k} ({v})" for k, v in lic.items()) + ".",
           "* Licencias de las imágenes (extensión multimedia, la que rige para usarlas): "
           + "; ".join(f"{k.rstrip('/').split('/licenses/')[-1].split('/publicdomain/')[-1]} ({v})"
                       for k, v in j["license_media"].value_counts().items()) + ".",
           "* Datasets de origen en GBIF:"]
    md += [f"  * `{k}` ({v} ocurrencias) — {dataset_title(k)}"
           + (", **fuente ya auditada aparte en `v001`**" if k in AGGREGATED else "") for k, v in prov.items()]
    if overlap is not None:
        md += ["", f"* Solapamiento con el `test_external` ya publicado en `v001` (iNaturalist CO + Mushroom "
                   f"Observer CO), por `occurrenceID`: **{overlap} de {n_obs}** observaciones. "
                   "Si se usan las dos fuentes hay que deduplicar."]
    md += ["", "Uso real en la prueba externa: [`data/versions/ext_co_001/`](../../data/versions/ext_co_001/DATASET_CARD.md) "
           "toma la Orinoquía de las fuentes de `v001` por departamento (más amplia que esta descarga) y lista en "
           "`gbif_orinoquia_exclusions.csv` por qué ninguno de los registros de esta descarga ausentes de `v001` es "
           "utilizable."]
    md += ["", "## Lectura", "",
           f"El conjunto es utilizable (coordenadas, fecha y licencia completas), pero su tamaño para estas "
           f"{len(classes)} clases es de {n_obs} observaciones, con {len(weak)} clases por debajo de 10 "
           "observaciones. No sostiene un macro-F1 de 10 clases; sirve como sonda regional con soporte por clase "
           "e intervalos de confianza, o como subgrupo dentro de una prueba externa colombiana más amplia.", ""]
    OUT_MD.write_text("\n".join(md), encoding="utf-8")

    if not args.no_register:
        registered = 0
        for p, note in CORE_FILES.items():
            path = SRC / p
            if not path.is_file():
                log.warning("not registered (missing): %s", rel(path))
                continue
            register_raw_file(DATASET_ID, path, source_url=SOURCE_URL, notes=note, download_date="2026-02-24")
            registered += 1
        log.info("registered %d files in data/manifests/raw_files.csv", registered)

    print(per_class.to_string(index=False))
    print(f"\nobservaciones con imagen: {n_obs} | imágenes: {n_img} | clases con soporte: {present}/{len(classes)}")
    if overlap is not None:
        print(f"solapamiento con test_external de v001: {overlap}/{n_obs} observaciones")
    print(f"\n{rel(OUT_CSV)}\n{rel(OUT_MD)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
