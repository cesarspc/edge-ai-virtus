"""Build the Colombian external evaluation set ``ext_co_001`` and label every observation by region.

The set is a *derived view* of the published dataset ``v001``: it takes the ``test_external`` observations of the
two Colombian sources (``inaturalist_co`` and ``mushroom_observer_co``) without changing any label, image or
filter, and adds the department and the region of each observation. ``v001`` itself is not modified.

* ``mind_funga_v2`` is left out: its images have no coordinates, no date and no country (neotropical images of
  mixed origin, folder labels), so they are not a *Colombian* test and cannot be placed in a region.
* Region ``orinoquia`` = departments Meta, Casanare, Arauca and Vichada, the same predicate used by the GBIF
  download ``0007586-260221153910048`` (``StateProvince is one of (Meta, Casanare, Arauca, Vichada)``).
* Department of an iNaturalist observation: the place of administrative level 10 (department) inside Colombia
  (place 7196) among its ``place_ids``, which iNaturalist computes from the coordinates. The place names are
  resolved once with the public places API and stored as snapshots.
* Department of a Mushroom Observer observation: the department named in its MO location
  (``"Yopal, Casanare, Colombia"``). Locations that do not name a department stay ``unknown``.
* The GBIF Orinoquía download is matched by ``occurrenceID``. Its records with images in the 10 classes that are
  not in ``v001`` are listed with the reason why they are not usable (``gbif_orinoquia_exclusions.csv``). If any of
  them were usable, the build stops: adding records needs a loader and a new set version, not a silent append.

Outputs:
    data/processed/ext_co_001/observations.csv, images.csv          (ignored by Git, reproducible)
    data/versions/ext_co_001/manifest.json, classes.csv, DATASET_CARD.md,
        orinoquia_observations.csv, gbif_orinoquia_exclusions.csv   (versioned)

    python scripts/data/build_external_set.py           # build (refuses to change a published set)
    python scripts/data/build_external_set.py --check   # rebuild in memory and compare with the published hashes
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import io
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adapters as A  # noqa: E402
from common import RAW_DIR, REPO_ROOT, log, make_session, rel, setup_logging, sha256_file, snapshot_json, utc_now  # noqa: E402

SET_ID = "ext_co_001"
BASE_VERSION = "v001"
SOURCES = ("inaturalist_co", "mushroom_observer_co")
EXCLUDED_SOURCES = {"mind_funga_v2": "sin coordenadas, fecha ni país: imágenes neotropicales de origen mixto con "
                                     "etiqueta de carpeta; no es una prueba colombiana ni se puede ubicar por región"}
ORINOQUIA = ("Arauca", "Casanare", "Meta", "Vichada")
GBIF_DOWNLOAD = "0007586-260221153910048"
GBIF_DWCA = RAW_DIR / "gbif_orinoquia" / "dwca"
INAT_API = "https://api.inaturalist.org/v1"
INAT_COLOMBIA = 7196
INAT_DATASET_KEY = "50c9509d-22c7-4a22-a47d-8c48425ef4a7"
ALLOWED_LICENSES = ("CC0", "PDM", "CC-BY", "CC-BY-SA", "CC-BY-NC", "CC-BY-NC-SA")   # configs/dataset_v001.yaml
CLASSES_FILE = REPO_ROOT / "data" / "versions" / BASE_VERSION / "classes.csv"
PROC_IN = REPO_ROOT / "data" / "processed" / BASE_VERSION
PROC_OUT = REPO_ROOT / "data" / "processed" / SET_ID
VER_OUT = REPO_ROOT / "data" / "versions" / SET_ID

# Colombian departments (ISO 3166-2:CO) plus the capital district; keys are accent-free and case-folded.
DEPARTMENTS = ["Amazonas", "Antioquia", "Arauca", "Atlántico", "Bogotá", "Bolívar", "Boyacá", "Caldas", "Caquetá",
               "Casanare", "Cauca", "Cesar", "Chocó", "Córdoba", "Cundinamarca", "Guainía", "Guaviare", "Huila",
               "La Guajira", "Magdalena", "Meta", "Nariño", "Norte de Santander", "Putumayo", "Quindío", "Risaralda",
               "San Andrés y Providencia", "Santander", "Sucre", "Tolima", "Valle del Cauca", "Vaupés", "Vichada"]
ALIASES = {"valle de cauca": "Valle del Cauca", "cudinamarca": "Cundinamarca", "bogota d.c.": "Bogotá",
           "bogota dc": "Bogotá", "distrito capital de bogota": "Bogotá", "distrito capital": "Bogotá",
           "san andres": "San Andrés y Providencia", "archipielago de san andres, providencia y santa catalina":
           "San Andrés y Providencia", "guajira": "La Guajira"}


def key(text: str) -> str:
    s = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", s).strip().casefold()


CANON = {key(d): d for d in DEPARTMENTS} | ALIASES


def department(name: str | None) -> str | None:
    return CANON.get(key(name)) if isinstance(name, str) and name.strip() else None


def region(dep: pd.Series) -> pd.Series:
    return pd.Series(["orinoquia" if d in ORINOQUIA else "other" if isinstance(d, str) else "unknown" for d in dep],
                     index=dep.index)


# --------------------------------------------------------------------------- inputs
def verified_v001() -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    manifest = json.loads((REPO_ROOT / "data" / "versions" / BASE_VERSION / "manifest.json").read_text(encoding="utf-8"))
    for name in ("observations.csv", "images.csv"):
        expected = next(o["sha256"] for o in manifest["outputs"] if o["name"] == name)
        if sha256_file(PROC_IN / name) != expected:
            raise SystemExit(f"{rel(PROC_IN / name)} is not the published {BASE_VERSION}; rebuild it with build_dataset.py")
    obs = pd.read_csv(PROC_IN / "observations.csv", low_memory=False, dtype={"source_observation_id": str})
    img = pd.read_csv(PROC_IN / "images.csv", low_memory=False, dtype={"source_image_id": str})
    keep = (obs["split"] == "test_external") & obs["source_id"].isin(SOURCES)
    obs = obs[keep].copy()
    img = img[img["observation_uid"].isin(obs["observation_uid"])].copy()
    return manifest, obs, img


def inat_departments(obs: pd.DataFrame, refresh: bool) -> tuple[pd.DataFrame, list[Path]]:
    wanted = set(obs.loc[obs["source_id"] == "inaturalist_co", "source_observation_id"].astype(int))
    place_ids: dict[int, list[int]] = {}
    files = sorted(Path(f) for f in glob.glob(str(RAW_DIR / "inaturalist_api" / "api" / "v001_co_*.json")))
    for f in files:
        for r in json.loads(f.read_text(encoding="utf-8"))["results"]:
            if r["id"] in wanted:
                place_ids[r["id"]] = r.get("place_ids") or []
    missing = wanted - set(place_ids)
    if missing:
        raise SystemExit(f"{len(missing)} iNaturalist observations of v001 are not in the stored API snapshots")
    distinct = sorted({p for v in place_ids.values() for p in v})
    s = make_session()
    places: dict[int, dict[str, Any]] = {}
    snaps = []
    for i in range(0, len(distinct), 100):
        chunk = distinct[i:i + 100]
        qid = f"{SET_ID}_places_p{i // 100 + 1}"
        data = snapshot_json(s, "inaturalist_api", qid, f"{INAT_API}/places/{','.join(map(str, chunk))}",
                             refresh=refresh, summary_key="total_results")
        snaps.append(RAW_DIR / "inaturalist_api" / "api" / f"{qid}.json")
        for p in data["results"]:
            places[p["id"]] = p
        absent = set(chunk) - {p["id"] for p in data["results"]}
        if absent:
            log.warning("%s: %d place ids not returned by the API: %s", qid, len(absent), sorted(absent)[:10])
    dept_places = {pid: p["name"] for pid, p in places.items()
                   if p.get("admin_level") == 10 and INAT_COLOMBIA in (p.get("ancestor_place_ids") or [])}
    rows = []
    for oid, pids in place_ids.items():
        names = sorted({dept_places[p] for p in pids if p in dept_places})
        canon = sorted({department(n) or n for n in names})
        rows.append({"source_id": "inaturalist_co", "source_observation_id": str(oid),
                     "department": canon[0] if len(canon) == 1 else None,
                     "department_raw": "|".join(names), "region_basis": "inaturalist_place_ids"})
    out = pd.DataFrame(rows)
    ambiguous = out["department_raw"].str.contains("|", regex=False) & out["department"].isna()
    if ambiguous.any():
        log.warning("%d iNaturalist observations fall in more than one department: left as unknown", int(ambiguous.sum()))
    return out, snaps


def mo_departments(obs: pd.DataFrame) -> pd.DataFrame:
    ids = set(obs.loc[obs["source_id"] == "mushroom_observer_co", "source_observation_id"].astype(int))
    mo = A._mo("observations")[["id", "location_id"]]
    loc = A._mo("locations")[["id", "name"]].rename(columns={"id": "location_id", "name": "location_name"})
    m = mo[mo["id"].isin(ids)].merge(loc, on="location_id", how="left")

    def pick(name: str) -> str | None:
        parts = [p.strip() for p in str(name).split(",")][:-1]          # last part is the country
        found = [department(p) for p in reversed(parts) if department(p)]
        return found[0] if found else None

    return pd.DataFrame({"source_id": "mushroom_observer_co", "source_observation_id": m["id"].astype(str),
                         "department": m["location_name"].map(pick),
                         "department_raw": m["location_name"], "region_basis": "mushroom_observer_location_name"})


def gbif_license(url: Any) -> str:
    m = re.search(r"creativecommons\.org/(licenses|publicdomain)/([a-z-]+)/", str(url))
    if not m:
        return f"UNKNOWN:{url}"
    code = m.group(2)
    return {"zero": "CC0", "mark": "PDM"}.get(code, "CC-" + code.upper())


def gbif_matches(obs: pd.DataFrame, classes: list[str], refresh: bool) -> tuple[pd.DataFrame, pd.DataFrame, list[Path]]:
    """gbifID for v001 observations present in the download, and the download records that are not in v001."""
    occ = pd.read_csv(GBIF_DWCA / "occurrence.txt", sep="\t", dtype=str, quoting=3, low_memory=False)
    mm = pd.read_csv(GBIF_DWCA / "multimedia.txt", sep="\t", dtype=str, quoting=3, low_memory=False)
    mm = mm[mm["type"].eq("StillImage") & mm["identifier"].fillna("").str.startswith("http")]
    occ = occ[occ["genus"].isin(classes) & occ["gbifID"].isin(mm["gbifID"])].copy()
    occ["bare_id"] = occ["occurrenceID"].fillna("").str.rstrip("/").str.rsplit("/", n=1).str[-1]
    occ["source_hint"] = occ["datasetKey"].map({INAT_DATASET_KEY: "inaturalist_co"})
    known = set(zip(obs["source_id"], obs["source_observation_id"]))
    hit = occ[[k in known for k in zip(occ["source_hint"], occ["bare_id"])]]
    match = hit[["source_hint", "bare_id", "gbifID", "stateProvince"]].rename(
        columns={"source_hint": "source_id", "bare_id": "source_observation_id", "gbifID": "gbif_id",
                 "stateProvince": "gbif_state_province"})

    rest = occ[~occ["gbifID"].isin(hit["gbifID"])].copy()
    lic = mm.assign(lic=mm["license"].map(gbif_license)).groupby("gbifID")["lic"].agg(lambda s: "|".join(sorted(set(s))))
    rest["image_licenses"] = rest["gbifID"].map(lic)
    snaps: list[Path] = []
    inat_ids = rest.loc[rest["source_hint"] == "inaturalist_co", "bare_id"].tolist()
    status: dict[str, dict[str, Any]] = {}
    if inat_ids:
        qid = f"{SET_ID}_gbif_orinoquia_inat_status"
        data = snapshot_json(make_session(), "inaturalist_api", qid, f"{INAT_API}/observations",
                             params={"id": ",".join(sorted(inat_ids)), "per_page": 200}, refresh=refresh,
                             summary_key="total_results")
        snaps.append(RAW_DIR / "inaturalist_api" / "api" / f"{qid}.json")
        status = {str(r["id"]): r for r in data["results"]}

    def reason(r: pd.Series) -> str:
        if r["source_hint"] == "inaturalist_co":
            o = status.get(r["bare_id"])
            if o is None:
                return "iNaturalist: la observación ya no existe en la API"
            taxon = (o.get("taxon") or {}).get("name", "")
            if o.get("quality_grade") != "research":
                return f"iNaturalist: grado '{o.get('quality_grade')}' (v001 exige research grade); taxón actual {taxon}"
            if taxon.split(" ")[0] != r["genus"]:
                return (f"iNaturalist la identifica como {taxon}; el árbol de GBIF la ubica en {r['genus']}. "
                        "v001 usa la taxonomía de iNaturalist")
            return "iNaturalist: research grade en el género; ausente de la consulta API de v001 (revisar)"
        allowed = [x for x in str(r["image_licenses"]).split("|") if x in ALLOWED_LICENSES]
        if not allowed:
            return f"licencia de imagen {r['image_licenses']} no admitida por allowed_licenses (se excluye ND)"
        return "UTILIZABLE"

    rest["reason"] = rest.apply(reason, axis=1)
    excl = rest[["gbifID", "datasetKey", "occurrenceID", "genus", "scientificName", "taxonRank",
                 "identificationVerificationStatus", "stateProvince", "eventDate", "image_licenses", "reason"]]
    excl = excl.rename(columns={"gbifID": "gbif_id"}).sort_values("gbif_id").reset_index(drop=True)
    usable = excl[excl["reason"].str.startswith("UTILIZABLE") | excl["reason"].str.endswith("(revisar)")]
    if len(usable):
        raise SystemExit(f"{len(usable)} GBIF Orinoquía records would be usable: they need a loader and a new set "
                         f"version, not a silent append:\n{usable.to_string()}")
    return match, excl, snaps


# --------------------------------------------------------------------------- outputs
OBS_COLUMNS = ["observation_uid", "source_id", "source_observation_id", "record_url", "genus", "label_id",
               "taxon_name_raw", "identification_basis", "event_date", "latitude", "longitude",
               "coordinate_uncertainty_m", "geo__quality", "department", "department_raw", "region", "region_basis",
               "gbif_id", "n_images"]
IMG_COLUMNS = ["image_uid", "observation_uid", "source_id", "source_image_id", "locator", "license", "license_scope",
               "rights_holder", "genus", "label_id", "region"]


def record_url(r: pd.Series) -> str:
    if r["source_id"] == "inaturalist_co":
        return f"https://www.inaturalist.org/observations/{r['source_observation_id']}"
    return f"https://mushroomobserver.org/observations/{r['source_observation_id']}"


def csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False, lineterminator="\n")
    return buf.getvalue().encode("utf-8")


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def build(refresh: bool) -> dict[str, Any]:
    v001, obs, img = verified_v001()
    classes = pd.read_csv(CLASSES_FILE).sort_values("label_id")["genus"].tolist()
    inat, inat_snaps = inat_departments(obs, refresh)
    mo = mo_departments(obs)
    dep = pd.concat([inat, mo], ignore_index=True)
    obs = obs.merge(dep, on=["source_id", "source_observation_id"], how="left", validate="one_to_one")
    if obs["region_basis"].isna().any():
        raise SystemExit(f"{int(obs['region_basis'].isna().sum())} observations without a region lookup")
    obs["region"] = region(obs["department"])
    match, excl, gbif_snaps = gbif_matches(obs, classes, refresh)
    obs = obs.merge(match, on=["source_id", "source_observation_id"], how="left", validate="one_to_one")
    disagree = obs[obs["gbif_id"].notna() & (obs["region"] != "orinoquia")]
    if len(disagree):
        raise SystemExit(f"{len(disagree)} observations in the GBIF Orinoquía download were not placed in the "
                         f"Orinoquía:\n{disagree[['observation_uid', 'department_raw', 'gbif_state_province']]}")
    obs["record_url"] = obs.apply(record_url, axis=1)
    obs = obs[OBS_COLUMNS].sort_values("observation_uid", kind="stable").reset_index(drop=True)
    img["region"] = img["observation_uid"].map(obs.set_index("observation_uid")["region"])
    img = img[IMG_COLUMNS].sort_values("image_uid", kind="stable").reset_index(drop=True)
    if obs["observation_uid"].duplicated().any() or img["image_uid"].duplicated().any():
        raise SystemExit("duplicated identifiers in the external set")

    ori = obs[obs["region"] == "orinoquia"].copy()
    ori["in_gbif_orinoquia_download"] = ori["gbif_id"].notna()
    ori = ori[["observation_uid", "source_id", "source_observation_id", "record_url", "genus", "label_id",
               "taxon_name_raw", "department", "department_raw", "event_date", "latitude", "longitude",
               "coordinate_uncertainty_m", "geo__quality", "n_images", "in_gbif_orinoquia_download", "gbif_id"]]
    tables = {"observations.csv": csv_bytes(obs), "images.csv": csv_bytes(img),
              "classes.csv": csv_bytes(pd.DataFrame({"label_id": range(len(classes)), "genus": classes})),
              "orinoquia_observations.csv": csv_bytes(ori), "gbif_orinoquia_exclusions.csv": csv_bytes(excl)}
    outputs = [{"name": n, "sha256": sha(b), "rows": int(b.count(b"\n") - 1)} for n, b in tables.items()]
    fingerprint = sha("".join(f"{o['name']}:{o['sha256']}\n" for o in sorted(outputs, key=lambda o: o["name"])).encode())

    in_files = [PROC_IN / "observations.csv", PROC_IN / "images.csv", CLASSES_FILE,
                REPO_ROOT / "data" / "versions" / BASE_VERSION / "manifest.json",
                RAW_DIR / "mushroom_observer" / "observations.csv", RAW_DIR / "mushroom_observer" / "locations.csv",
                GBIF_DWCA / "occurrence.txt", GBIF_DWCA / "multimedia.txt", *inat_snaps, *gbif_snaps]
    inat_raw = sorted(glob.glob(str(RAW_DIR / "inaturalist_api" / "api" / "v001_co_*.json")))
    manifest = {
        "set_id": SET_ID, "kind": "external_evaluation_set", "created_utc": utc_now(),
        "content_fingerprint": fingerprint,
        "derived_from": {"version": BASE_VERSION, "content_fingerprint": v001["content_fingerprint"],
                         "split": "test_external", "sources": list(SOURCES)},
        "excluded_sources": EXCLUDED_SOURCES,
        "region_definition": {"orinoquia": list(ORINOQUIA),
                              "same_predicate_as": f"GBIF download {GBIF_DOWNLOAD} (StateProvince)",
                              "inaturalist": "place of admin_level 10 inside Colombia (place 7196) among place_ids",
                              "mushroom_observer": "department named in the MO location"},
        "classes": classes,
        "code": {rel(Path(__file__)): sha256_file(Path(__file__))},
        "inputs": [{"path": rel(p), "sha256": sha256_file(p)} for p in in_files]
                  + [{"path": "data/raw/inaturalist_api/api/v001_co_*.json", "files": len(inat_raw),
                      "sha256_of_hashes": sha("".join(sha256_file(Path(f)) for f in inat_raw).encode())}],
        "outputs": outputs,
        "counts": {
            "observations": len(obs), "images": len(img),
            "observations_by_source": obs["source_id"].value_counts().sort_index().to_dict(),
            "observations_by_region": obs["region"].value_counts().sort_index().to_dict(),
            "images_by_region": img["region"].value_counts().sort_index().to_dict(),
            "orinoquia_by_department": ori["department"].value_counts().sort_index().to_dict(),
            "orinoquia_by_source": ori["source_id"].value_counts().sort_index().to_dict(),
            "orinoquia_in_gbif_download": int(ori["in_gbif_orinoquia_download"].sum()),
            "gbif_orinoquia_records_not_usable": len(excl),
        },
    }
    return {"obs": obs, "img": img, "ori": ori, "excl": excl, "tables": tables, "manifest": manifest, "classes": classes}


def mil(n: int) -> str:
    return f"{int(n):,}".replace(",", ".")


def card(res: dict[str, Any]) -> str:
    obs, img, ori, excl, man, classes = res["obs"], res["img"], res["ori"], res["excl"], res["manifest"], res["classes"]
    regions = ["orinoquia", "other", "unknown"]
    piv = obs.pivot_table(index="genus", columns="region", values="observation_uid", aggfunc="count", fill_value=0)
    piv = piv.reindex(index=classes, columns=regions, fill_value=0)
    src = obs.pivot_table(index="source_id", columns="region", values="observation_uid", aggfunc="count", fill_value=0)
    src = src.reindex(columns=regions, fill_value=0)
    lines = [f"# Conjunto de evaluación externa `{SET_ID}` (Colombia)", "",
             "> Generado por `scripts/data/build_external_set.py`; no editar a mano.", "",
             f"* **Huella de contenido:** `{man['content_fingerprint']}`",
             f"* **Derivado de:** `{BASE_VERSION}` (`test_external`, huella `{man['derived_from']['content_fingerprint'][:16]}…`), "
             "sin cambiar etiquetas, imágenes ni filtros.",
             f"* **Fuentes:** {', '.join(f'`{s}`' for s in SOURCES)}. Excluida: `mind_funga_v2` ({EXCLUDED_SOURCES['mind_funga_v2']}).",
             f"* **Clases ({len(classes)}):** " + ", ".join(f"{i}={g}" for i, g in enumerate(classes)),
             f"* **Observaciones:** {mil(len(obs))} · **Imágenes:** {mil(len(img))}", "",
             "## Región", "",
             f"`orinoquia` = departamentos {', '.join(ORINOQUIA)}, el mismo criterio de la descarga GBIF "
             f"`{GBIF_DOWNLOAD}`. `other` = otro departamento de Colombia. `unknown` = la ubicación no nombra un departamento.",
             "", "* iNaturalist: lugar de nivel administrativo 10 dentro de Colombia entre los `place_ids` de la observación.",
             "* Mushroom Observer: departamento nombrado en la ubicación MO (por ejemplo «Yopal, Casanare, Colombia»).", "",
             "| Fuente | Orinoquía | Otra región | Sin departamento |", "|---|---:|---:|---:|"]
    lines += [f"| `{s}` | {mil(r['orinoquia'])} | {mil(r['other'])} | {mil(r['unknown'])} |" for s, r in src.iterrows()]
    lines += ["", "### Observaciones por género y región", "",
              "| Género | Orinoquía | Otra región | Sin departamento |", "|---|---:|---:|---:|"]
    lines += [f"| {g} | {mil(r['orinoquia'])} | {mil(r['other'])} | {mil(r['unknown'])} |" for g, r in piv.iterrows()]
    lines += ["", "## Observaciones de la Orinoquía", "",
              f"**{mil(len(ori))} observaciones** ({mil(int(ori['n_images'].sum()))} imágenes). Lista completa, con enlace "
              "a cada registro, en [`orinoquia_observations.csv`](orinoquia_observations.csv).", "",
              "| Departamento | iNaturalist | Mushroom Observer | Total |", "|---|---:|---:|---:|"]
    dsrc = ori.pivot_table(index="department", columns="source_id", values="observation_uid", aggfunc="count",
                           fill_value=0).reindex(columns=list(SOURCES), fill_value=0)
    lines += [f"| {d} | {mil(r['inaturalist_co'])} | {mil(r['mushroom_observer_co'])} | {mil(r.sum())} |"
              for d, r in dsrc.iterrows()]
    lines += ["", f"De ellas, {mil(int(ori['in_gbif_orinoquia_download'].sum()))} aparecen también en la descarga GBIF "
                  f"`{GBIF_DOWNLOAD}` (columna `gbif_id`). Esa descarga (24-02-2026) no contiene ningún registro de "
                  "Mushroom Observer, aunque el 28-10-2026 la API de GBIF devolvía 172 registros de MO en Colombia con "
                  "el mismo filtro de departamentos; la causa no se determinó. Por eso la Orinoquía se identifica aquí "
                  "por departamento en cada fuente y la descarga GBIF solo sirve de verificación cruzada.", "",
              "## Registros de la descarga GBIF de la Orinoquía que no entran", "",
              f"La descarga tiene {mil(len(excl))} registros con imagen en las 10 clases que no están en `{BASE_VERSION}`. "
              "Ninguno es utilizable; motivo por registro en "
              "[`gbif_orinoquia_exclusions.csv`](gbif_orinoquia_exclusions.csv):", ""]
    reasons = excl["reason"].str.replace(r"; taxón actual .*$", "", regex=True).str.replace(
        r"la identifica como [^;]+?;", "la identifica en otro género;", regex=True).value_counts()
    lines += [f"* {r} ({n})" for r, n in reasons.items()]
    lines += ["", "## Uso", "",
              "Solo evaluación de cambio de dominio: no se usa para seleccionar modelos ni hiperparámetros. "
              f"Las imágenes se descargan con `python scripts/data/download_external_images.py --set {SET_ID}` y los "
              f"modelos se evalúan con `python scripts/train/evaluate_external.py --set {SET_ID} --run runs/<run_id>`. "
              "El conjunto completo (Colombia + MIND.Funga) es [`ext_full_001`](../ext_full_001/DATASET_CARD.md).", ""]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="rebuild in memory and compare with the published outputs")
    ap.add_argument("--refresh", action="store_true", help="re-run the stored API queries")
    args = ap.parse_args()
    setup_logging()
    res = build(args.refresh)
    man = res["manifest"]
    published = VER_OUT / "manifest.json"
    if published.exists():
        old = json.loads(published.read_text(encoding="utf-8"))
        same = old["content_fingerprint"] == man["content_fingerprint"]
        if args.check:
            print("OK: identical to the published set" if same else "DIFFERENT from the published set")
            return 0 if same else 1
        if not same:
            raise SystemExit(f"{SET_ID} is published with another content; create {SET_ID[:-3]}002 instead of changing it")
        man["created_utc"] = old["created_utc"]
    elif args.check:
        raise SystemExit(f"{SET_ID} is not published yet")
    PROC_OUT.mkdir(parents=True, exist_ok=True)
    VER_OUT.mkdir(parents=True, exist_ok=True)
    for name in ("observations.csv", "images.csv"):
        (PROC_OUT / name).write_bytes(res["tables"][name])
    for name in ("classes.csv", "orinoquia_observations.csv", "gbif_orinoquia_exclusions.csv"):
        (VER_OUT / name).write_bytes(res["tables"][name])
    published.write_text(json.dumps(man, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (VER_OUT / "DATASET_CARD.md").write_text(card(res), encoding="utf-8")
    c = man["counts"]
    print(json.dumps(c, indent=2, ensure_ascii=False))
    print(f"fingerprint {man['content_fingerprint']}\n{rel(VER_OUT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
