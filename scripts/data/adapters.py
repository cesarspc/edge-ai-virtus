"""One adapter per source: loads the raw files (read-only) and maps them onto the common audit roles.

Every ``audit_<dataset>()`` returns an :class:`AdapterResult`. Adapters never write to ``data/raw`` and never
alter records; derived columns (e.g. genus parsed from a scientific name) are created in memory only and are
flagged as DERIVED in the report.
"""
from __future__ import annotations

import gzip
import io
import json
import re
import zipfile
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from audit_core import AuditTable
from common import RAW_DIR, log, rel


@dataclass
class AdapterResult:
    dataset_id: str
    tables: list[AuditTable] = field(default_factory=list)
    class_counts: list[pd.DataFrame] = field(default_factory=list)   # pre-aggregated counts (API sources)
    facts: dict[str, Any] = field(default_factory=dict)              # numbers for the inventory / report
    overlaps: list[dict[str, Any]] = field(default_factory=list)
    id_sets: dict[str, set] = field(default_factory=dict)            # observation ids kept for cross-dataset overlap
    problems: list[str] = field(default_factory=list)                # access problems / notes


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _api(dataset_id: str, query_id: str) -> Any | None:
    p = RAW_DIR / dataset_id / "api" / f"{query_id}.json"
    return _read_json(p) if p.exists() else None


def _count(dataset_id: str, query_id: str) -> int | None:
    d = _api(dataset_id, query_id)
    if d is None:
        return None
    return d.get("count", d.get("total_results"))


def _counts_frame(dataset_id: str, table: str, level: str, taxa: list[str], n_obs: list[float] | None = None,
                  n_img: list[float] | None = None) -> pd.DataFrame:
    df = pd.DataFrame({"dataset_id": dataset_id, "table": table, "level": level, "taxon": taxa,
                       "n_images": n_img if n_img is not None else np.nan,
                       "n_observations": n_obs if n_obs is not None else np.nan})
    return df.sort_values(["n_images", "n_observations"], ascending=False, na_position="last").reset_index(drop=True)


def _first_token(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip().str.split(r"\s+", n=1, regex=True).str[0]


# =========================================================================== FungiTastic
FT_FULL_FILES = {
    "Train": "FungiTastic/FungiTastic-Train.csv",
    "ClosedSet-Val": "FungiTastic/FungiTastic-ClosedSet-Val.csv",
    "ClosedSet-Test": "FungiTastic/FungiTastic-ClosedSet-Test.csv",
    "DNA-Test": "FungiTastic/FungiTastic-DNA-Test.csv",
    "OpenSet-Val": "FungiTastic/FungiTastic-OpenSet-Val.csv",
    "OpenSet-Test": "FungiTastic/FungiTastic-OpenSet-Test.csv",
}
FT_MINI_FILES = {k: f"FungiTastic-Mini/FungiTastic-Mini-{k}.csv" for k in ("Train", "Val", "Test", "DNA-Test")}
FT_FS_FILES = {k: f"FungiTastic-FewShot/FungiTastic-FewShot-{k}.csv" for k in ("Train", "Val", "Test")}
FT_ROLES = {
    "observation_id": "observationID", "image_file": "filename", "species": "species", "genus": "genus",
    "family": "family", "order": "order", "date": "eventDate", "latitude": "latitude", "longitude": "longitude",
    "country": "countryCode", "region": "region", "substrate": "substrate", "habitat": "habitat",
}


def _ft_read(z: zipfile.ZipFile, member: str) -> pd.DataFrame:
    with z.open(member) as fh:   # 'captions' (long generated text) is not needed for the audit
        return pd.read_csv(fh, usecols=lambda c: c != "captions", low_memory=False)


def audit_fungitastic() -> AdapterResult:
    res = AdapterResult("fungitastic")
    zpath = RAW_DIR / "fungitastic" / "metadata.zip"
    if not zpath.exists():
        res.problems.append("metadata.zip not found; run scripts/data/acquire_datasets.py fungitastic")
        return res
    frames: dict[str, pd.DataFrame] = {}
    with zipfile.ZipFile(zpath) as z:
        for group, files in (("full", FT_FULL_FILES), ("mini", FT_MINI_FILES), ("fewshot", FT_FS_FILES)):
            for name, member in files.items():
                log.info("FungiTastic: reading %s", member)
                frames[f"{group}/{name}"] = _ft_read(z, member)
        res.facts["csv_columns_full_train"] = list(frames["full/Train"].columns)
        with z.open(FT_FULL_FILES["Train"]) as fh:
            res.facts["caption_column_present"] = "captions" in pd.read_csv(fh, nrows=1).columns

    # per-file detail tables (size only)
    for key, df in frames.items():
        res.tables.append(AuditTable("fungitastic", key, df, FT_ROLES, "image", main=(key == "full/Train"),
                                     note="CSV oficial individual" + ("; su distribución es la relevante para entrenamiento" if key == "full/Train" else "")))
    # unions (in memory, for counting only): closed-set files + DNA + open-set files, unique image filename
    def union(group: str, names: list[str]) -> tuple[pd.DataFrame, int]:
        cat = pd.concat([frames[f"{group}/{n}"].assign(_file=n) for n in names], ignore_index=True)
        return cat.drop_duplicates(subset="filename", keep="first").reset_index(drop=True), len(cat)

    full, n_full_rows = union("full", list(FT_FULL_FILES))
    mini, n_mini_rows = union("mini", list(FT_MINI_FILES))
    fs, n_fs_rows = union("fewshot", list(FT_FS_FILES))
    note = "unión en memoria (solo para contar) de todos los CSV del subconjunto, deduplicando por 'filename'"
    for name, df, nrows in (("full_union", full, n_full_rows), ("mini_union", mini, n_mini_rows),
                            ("fewshot_union", fs, n_fs_rows)):
        t = AuditTable("fungitastic", name, df, FT_ROLES, "image", note=note + f"; filas totales antes de deduplicar: {nrows}")
        res.tables.append(t)
    full_t = next(t for t in res.tables if t.table == "full_union")

    # ---- dataset-specific checks
    train, cval, ctest, dna = (frames[f"full/{n}"] for n in ("Train", "ClosedSet-Val", "ClosedSet-Test", "DNA-Test"))
    sets = {n: set(frames[f"full/{n}"]["observationID"].dropna().astype("int64")) for n in FT_FULL_FILES}
    res.id_sets = {f"fungitastic:{n}": s for n, s in sets.items()}
    res.id_sets["fungitastic:full_union"] = set(full["observationID"].dropna().astype("int64"))
    pairs = [("Train", "ClosedSet-Val"), ("Train", "ClosedSet-Test"), ("ClosedSet-Val", "ClosedSet-Test"),
             ("Train", "DNA-Test"), ("ClosedSet-Test", "DNA-Test"), ("ClosedSet-Val", "OpenSet-Val"),
             ("ClosedSet-Test", "OpenSet-Test")]
    for a, b in pairs:
        nested = a.startswith("Closed") and b.startswith("Open")
        res.overlaps.append({"scope": "within FungiTastic (full)", "set_a": f"fungitastic:{a}", "set_b": f"fungitastic:{b}",
                             "key": "observationID", "n_a": len(sets[a]), "n_b": len(sets[b]),
                             "n_intersection": len(sets[a] & sets[b]),
                             "note": ("estructura oficial: el conjunto cerrado está contenido en el abierto" if nested
                                      else "sin observaciones compartidas entre particiones distintas (0 esperado)")})
    full_obs = res.id_sets["fungitastic:full_union"]
    for group, files in (("mini", FT_MINI_FILES), ("fewshot", FT_FS_FILES)):
        s = set(pd.concat([frames[f"{group}/{n}"] for n in files])["observationID"].dropna().astype("int64"))
        res.id_sets[f"fungitastic:{group}_union"] = s
        res.overlaps.append({"scope": "within FungiTastic (subsets)", "set_a": f"fungitastic:{group}_union",
                             "set_b": "fungitastic:full_union", "key": "observationID", "n_a": len(s), "n_b": len(full_obs),
                             "n_intersection": len(s & full_obs), "note": "subconjunto oficial contenido en el completo"})
    mini_obs, fs_obs = res.id_sets["fungitastic:mini_union"], res.id_sets["fungitastic:fewshot_union"]
    res.overlaps.append({"scope": "within FungiTastic (subsets)", "set_a": "fungitastic:mini_union", "set_b": "fungitastic:fewshot_union",
                         "key": "observationID", "n_a": len(mini_obs), "n_b": len(fs_obs),
                         "n_intersection": len(mini_obs & fs_obs), "note": "Mini y FewShot son subconjuntos independientes"})

    tr_sp = set(train["species"].dropna())
    for name, df in (("ClosedSet-Val", cval), ("ClosedSet-Test", ctest), ("DNA-Test", dna),
                     ("OpenSet-Val", frames["full/OpenSet-Val"]), ("OpenSet-Test", frames["full/OpenSet-Test"])):
        sp = df["species"]
        res.facts[f"{name}_rows_species_not_in_train"] = int((sp.notna() & ~sp.isin(tr_sp)).sum())
        res.facts[f"{name}_rows"] = len(df)
    known = full[full["category_id"] != -1].dropna(subset=["species"])
    res.facts["n_category_id_known_full_union"] = int(known["category_id"].nunique())
    res.facts["n_species_known_full_union"] = int(known["species"].nunique())
    res.facts["n_species_with_multiple_category_ids"] = int((known.groupby("species")["category_id"].nunique() > 1).sum())
    res.facts["n_species_train"] = int(train["species"].nunique())
    res.facts["year_range_by_file"] = {n: [int(frames[f"full/{n}"]["year"].min()), int(frames[f"full/{n}"]["year"].max())] for n in FT_FULL_FILES}
    res.facts["n_category_id_by_file"] = {n: int(frames[f"full/{n}"]["category_id"].nunique()) for n in FT_FULL_FILES}
    res.facts["n_scientificName_by_file"] = {n: int(frames[f"full/{n}"]["scientificName"].nunique()) for n in FT_FULL_FILES}
    res.facts["n_species_by_file"] = {n: int(frames[f"full/{n}"]["species"].nunique()) for n in FT_FULL_FILES}
    res.facts["category_id_minus1_rows_full_union"] = int((full["category_id"] == -1).sum())
    res.facts["category_id_minus1_by_file"] = {n: int((frames[f"full/{n}"]["category_id"] == -1).sum()) for n in FT_FULL_FILES}
    res.facts["countryCode_counts"] = {str(k): int(v) for k, v in full["countryCode"].value_counts(dropna=False).items()}
    res.facts["year_min"], res.facts["year_max"] = int(full["year"].min()), int(full["year"].max())
    res.facts["rows_per_file"] = {k: len(v) for k, v in frames.items()}
    res.facts["filename_repeated_across_files"] = int(n_full_rows - len(full))
    full_t.extra_checks["duplicate_note"] = ("las filas repetidas entre CSV corresponden a la relación ClosedSet/OpenSet "
                                             "(mismos archivos en varios CSV), no a duplicados dentro de un CSV")
    within = {n: int(frames[f"full/{n}"]["filename"].duplicated().sum()) for n in FT_FULL_FILES}
    res.facts["filename_duplicates_within_each_file"] = within
    return res


# =========================================================================== DF20
DF20_ROLES = {
    "observation_id": "gbifID", "image_id": "ImageUniqueID", "image_file": "image_path", "species": "species",
    "genus": "genus", "family": "family", "order": "order", "date": "eventDate", "latitude": "Latitude",
    "longitude": "Longitude", "country": "level0Name", "region": "level1Name", "substrate": "Substrate",
    "habitat": "Habitat", "author": "rightsHolder",
}


def audit_df20() -> AdapterResult:
    res = AdapterResult("df20")
    zpath = RAW_DIR / "df20" / "DF20-metadata.zip"
    if not zpath.exists():
        res.problems.append("DF20-metadata.zip not found; run scripts/data/acquire_datasets.py df20")
        return res
    members = {"train": "DF20-train_metadata_PROD-2.csv", "public_test": "DF20-public_test_metadata_PROD-2.csv"}
    frames = {}
    with zipfile.ZipFile(zpath) as z:
        for k, m in members.items():
            with z.open(m) as fh:
                frames[k] = pd.read_csv(fh, low_memory=False)
    for k, df in frames.items():
        res.tables.append(AuditTable("df20", k, df, DF20_ROLES, "image", main=False, note=f"CSV oficial {members[k]}"))
    union = pd.concat([frames["train"].assign(_file="train"), frames["public_test"].assign(_file="public_test")], ignore_index=True)
    t = AuditTable("df20", "train+public_test", union, DF20_ROLES, "image",
                   note="unión en memoria de train y public_test (sin deduplicar)")
    tr, te = set(frames["train"]["gbifID"].dropna().astype("int64")), set(frames["public_test"]["gbifID"].dropna().astype("int64"))
    res.id_sets = {"df20:train": tr, "df20:public_test": te, "df20:union": tr | te}
    res.overlaps.append({"scope": "within DF20", "set_a": "df20:train", "set_b": "df20:public_test", "key": "gbifID",
                         "n_a": len(tr), "n_b": len(te), "n_intersection": len(tr & te),
                         "note": "observaciones presentes en train y test (fuga de información entre particiones)"})
    imgs_tr, imgs_te = set(frames["train"]["ImageUniqueID"]), set(frames["public_test"]["ImageUniqueID"])
    res.overlaps.append({"scope": "within DF20", "set_a": "df20:train", "set_b": "df20:public_test", "key": "ImageUniqueID",
                         "n_a": len(imgs_tr), "n_b": len(imgs_te), "n_intersection": len(imgs_tr & imgs_te),
                         "note": "0 esperado (cada imagen en una sola partición)"})
    t.extra_checks["conflicting_species_note"] = "observaciones con más de una especie: ver n_observations_with_conflicting_species"
    res.tables.append(t)
    res.facts["rows"] = {k: len(v) for k, v in frames.items()}
    res.facts["columns"] = list(union.columns)
    res.facts["countryCode_counts"] = {str(k): int(v) for k, v in union["countryCode"].value_counts(dropna=False).items()}
    res.facts["year_min"], res.facts["year_max"] = int(union["year"].min()), int(union["year"].max())
    res.facts["n_rightsHolder"] = int(union["rightsHolder"].nunique())
    per_species = union.dropna(subset=["species"]).groupby("species")["class_id"].nunique()
    res.facts.update({
        "n_class_id": int(union["class_id"].nunique()), "n_scientificName": int(union["scientificName"].nunique()),
        "n_species_binomial": int(union["species"].nunique()), "n_rows_species_missing": int(union["species"].isna().sum()),
        "n_species_with_multiple_class_ids": int((per_species > 1).sum()),
    })
    return res


# =========================================================================== MIND.Funga
def audit_mind_funga() -> AdapterResult:
    res = AdapterResult("mind_funga_v2")
    api = RAW_DIR / "mind_funga_v2" / "api" / "mendeley_dataset.json"
    cdp = RAW_DIR / "mind_funga_v2" / "zip_central_directory.csv"
    if not api.exists():
        res.problems.append("ACCESS_PROBLEM: Mendeley dataset JSON not available")
        return res
    d = _read_json(api)
    files = pd.DataFrame([{"file_id": f["id"], "filename": f["filename"], "folder_id": f["folder_id"],
                           "size": f["size"], "sha256_reported": f["content_details"].get("sha256_hash"),
                           "content_type": f["content_details"].get("content_type"),
                           "created": f["content_details"].get("created_date")} for f in d["files"]])
    res.facts.update({"version": d["version"], "publish_date": d["publish_date"], "doi": d["doi"]["id"],
                      "license": d["data_licence"]["short_name"], "total_bytes": d["size"], "n_files_api": len(files),
                      "n_folders_api": int(files["folder_id"].nunique()),
                      "versions": [(v["version"], v["publish_date"][:10]) for v in d["versions"]]})
    if cdp.exists():
        cd = pd.read_csv(cdp)
        cd = cd[~cd["zip_path"].str.endswith("/")].copy()
        parts = cd["zip_path"].str.split("/", expand=True)
        cd["taxon_dir"], cd["filename"] = parts[1], parts[2]
        res.facts["n_files_zip_central_directory"] = len(cd)
        # map Mendeley folder_id -> directory name via unambiguous (filename, size) pairs
        key_a = files.assign(k=files["filename"] + "|" + files["size"].astype(str))
        key_c = cd.assign(k=cd["filename"] + "|" + cd["file_size"].astype(str))
        ua = key_a[~key_a["k"].duplicated(keep=False)]
        uc = key_c[~key_c["k"].duplicated(keep=False)]
        m = ua.merge(uc[["k", "taxon_dir"]], on="k", how="inner")
        votes = m.groupby("folder_id")["taxon_dir"].agg(lambda s: s.value_counts().index[0])
        conflicts = m.groupby("folder_id")["taxon_dir"].nunique()
        res.facts["folder_id_to_dir_conflicts"] = int((conflicts > 1).sum())
        files["taxon_dir"] = files["folder_id"].map(votes)
        res.facts["n_files_without_taxon_dir"] = int(files["taxon_dir"].isna().sum())
        res.facts["files_api_vs_cd_same_multiset"] = bool(
            sorted(zip(files["taxon_dir"], files["filename"], files["size"])) ==
            sorted(zip(cd["taxon_dir"], cd["filename"], cd["file_size"]))) if files["taxon_dir"].notna().all() else False
    else:
        res.problems.append("zip_central_directory.csv missing: taxon (folder) names unavailable; classes identified only by folder_id")
        files["taxon_dir"] = np.nan
    files["class_id"] = files["taxon_dir"].fillna(files["folder_id"])
    tok = files["taxon_dir"].astype("string").str.strip().str.split(r"\s+", regex=True)
    files["genus_derived"] = tok.str[0]
    uncertain = files["taxon_dir"].astype("string").str.contains(r"\b(?:sp|sp\.|spp|spp\.|aff|aff\.|cf|cf\.)(?:\s|$)", case=False, regex=True).fillna(False)
    files["species_derived"] = files["taxon_dir"].where((tok.str.len() >= 2) & ~uncertain)   # certain binomials only
    t = AuditTable("mind_funga_v2", "files_v2", files, {
        "image_id": "file_id", "image_file": "filename", "taxon_label": "class_id", "species": "species_derived",
        "genus": "genus_derived"}, "image",
        note="índice de archivos de la API de Mendeley (v2) + nombres de carpeta del directorio central del ZIP oficial; "
             "especie/género DERIVADOS del nombre de carpeta (una sola palabra = solo género)")
    # exact-duplicate files according to the SHA-256 reported by the source (NOT computed locally)
    g = files.groupby("sha256_reported")
    sizes = g.size()
    dup_hashes = sizes[sizes > 1].index
    dup = files[files["sha256_reported"].isin(dup_hashes)]
    cross = dup.groupby("sha256_reported")["class_id"].nunique()
    t.extra_checks.update({
        "sha256_source_reported_groups_with_duplicates": int(len(dup_hashes)),
        "sha256_source_reported_files_in_duplicate_groups": int(len(dup)),
        "sha256_source_reported_redundant_copies": int((sizes[sizes > 1] - 1).sum()),
        "sha256_source_reported_groups_spanning_multiple_taxa": int((cross > 1).sum()),
        "sha256_duplicates": "SOURCE_REPORTED (hash publicado por Mendeley; no recalculado localmente)",
    })
    bv = files["filename"].str.contains(r"\b(?:branco|verde)\b", case=False, regex=True)
    t.extra_checks["filenames_with_branco_or_verde"] = int(bv.sum())
    res.facts["filenames_with_branco_or_verde"] = int(bv.sum())
    if files["taxon_dir"].notna().any():
        n_tok = files.drop_duplicates("class_id")["taxon_dir"].astype("string").str.split(r"\s+", regex=True).str.len()
        res.facts["n_folders_single_token_genus_only"] = int((n_tok == 1).sum())
        res.facts["n_folders_two_or_more_tokens"] = int((n_tok >= 2).sum())
        res.facts["n_folders_uncertain_sp_aff_cf"] = int(files.assign(u=uncertain).drop_duplicates("class_id")["u"].sum())
        res.facts["n_files_in_uncertain_folders"] = int(uncertain.sum())
    # same filename (and size) stored under more than one taxon folder
    k = files.groupby(["filename", "size"])["class_id"].nunique()
    t.extra_checks["same_filename_and_size_in_multiple_taxa"] = int((k > 1).sum())
    res.tables.append(t)
    return res


# =========================================================================== iNaturalist Open Data
def _read_gz_prefix(path: Path) -> pd.DataFrame:
    """Decompress a truncated gzip fragment; the last (cut) line is dropped."""
    raw = zlib.decompressobj(31).decompress(path.read_bytes())
    lines = raw.decode("utf-8", "replace").split("\n")[:-1]
    return pd.read_csv(io.StringIO("\n".join(lines)), sep="\t", dtype=str, keep_default_na=False)


def audit_inaturalist_open_data() -> AdapterResult:
    res = AdapterResult("inaturalist_open_data")
    d = RAW_DIR / "inaturalist_open_data"
    head = _api("inaturalist_open_data", "s3_head_probes")
    if head:
        res.facts["s3_object_sizes_bytes"] = {k: int(v["content_length"]) for k, v in head.items() if v.get("content_length")}
        res.facts["s3_last_modified"] = {k: v.get("last_modified") for k, v in head.items()}
    taxa_path = d / "taxa.csv.gz"
    if taxa_path.exists():
        taxa = pd.read_csv(taxa_path, sep="\t", dtype={"ancestry": "string", "rank": "string", "name": "string"})
        anc = taxa["ancestry"].fillna("")
        is_fungi = taxa["taxon_id"].eq(47170) | anc.str.contains(r"(?:^|/)47170(?:/|$)", regex=True)
        f = taxa[is_fungi]
        res.facts["taxa_total_rows"] = len(taxa)
        res.facts["taxa_active_total"] = int(taxa["active"].astype(str).str.lower().eq("true").sum())
        res.facts["fungi_taxa_total_rows"] = len(f)
        res.facts["fungi_taxa_active"] = int(f["active"].astype(str).str.lower().eq("true").sum())
        act = f[f["active"].astype(str).str.lower().eq("true")]
        res.facts["fungi_active_by_rank"] = {str(k): int(v) for k, v in act["rank"].value_counts().head(12).items()}
    else:
        res.problems.append("taxa.csv.gz not found")
    frag = {"observations": d / "observations.csv.gz.head2MB", "photos": d / "photos.csv.gz.head3MB"}
    if frag["observations"].exists():
        o = _read_gz_prefix(frag["observations"])
        res.facts["observations_columns"] = list(o.columns)
        res.tables.append(AuditTable("inaturalist_open_data", "observations_head_fragment", o, {
            "observation_id": "observation_uuid", "latitude": "latitude", "longitude": "longitude",
            "date": "observed_on", "taxon_label": "taxon_id", "author": "observer_id"}, "observation",
            evidence="LOCAL_VERIFIED", is_sample=True,
            note="FRAGMENTO (primeros 2 MB del .gz; filas más antiguas). Solo verifica el esquema; NO es representativo"))
    if frag["photos"].exists():
        p = _read_gz_prefix(frag["photos"])
        res.facts["photos_columns"] = list(p.columns)
        res.facts["photos_fragment_license_counts"] = {str(k) if k != "" else "<vacío>": int(v) for k, v in p["license"].value_counts().items()}
        res.tables.append(AuditTable("inaturalist_open_data", "photos_head_fragment", p, {
            "observation_id": "observation_uuid", "image_id": "photo_id", "license": "license", "author": "observer_id"},
            "image", is_sample=True,
            note="FRAGMENTO (primeros 3 MB del .gz; filas más antiguas). Solo verifica el esquema; NO es representativo"))
    return res


# =========================================================================== iNaturalist API
def audit_inaturalist_api() -> AdapterResult:
    res = AdapterResult("inaturalist_api")
    ds = "inaturalist_api"
    q = lambda k: _count(ds, k)  # noqa: E731
    res.facts.update({
        "fungi_world_observations": q("obs_fungi_world"), "fungi_colombia_observations": q("obs_fungi_co_all"),
        "fungi_colombia_with_photos": q("obs_fungi_co_photos"), "fungi_colombia_research_grade": q("obs_fungi_co_research"),
        "fungi_colombia_research_photos_licensed": q("obs_fungi_co_research_photos_licensed"),
        "species_colombia_all": q("species_counts_co_all"), "species_colombia_research": q("species_counts_co_research"),
        "photo_license_counts_research": {lic: q(f"obs_fungi_co_research_photolic_{lic.replace('-', '_')}")
                                            for lic in ["cc0", "cc-by", "cc-by-nc", "cc-by-sa", "cc-by-nd", "cc-by-nc-sa", "cc-by-nc-nd"]},
    })
    place = _api(ds, "place_colombia")
    if place:
        res.facts["colombia_place_id"] = next((p["id"] for p in place["results"] if p.get("admin_level") == 0), None)
    # class counts (observations per species) from the stored species_counts pages
    for qual, pages, table in (("research", 2, "species_counts_research_grade"), ("any", 5, "species_counts_all_grades")):
        taxa, n = [], []
        for pg in range(1, pages + 1):
            page = _api(ds, f"species_counts_co_{qual}_p{pg}")
            if not page:
                continue
            for r in page["results"]:
                taxa.append(r["taxon"]["name"]); n.append(r["count"])
        if taxa:
            sp = _counts_frame(ds, table, "species", taxa, n_obs=n)
            gen = sp.assign(g=sp["taxon"].str.split().str[0]).groupby("g", as_index=False)["n_observations"].sum()
            res.class_counts += [sp, _counts_frame(ds, table, "genus", gen["g"].tolist(), n_obs=gen["n_observations"].tolist())]
            res.facts[f"{table}_rows_collected"] = len(taxa)
    # random sample of research-grade observations with photos (stored snapshot)
    smp = _api(ds, "obs_fungi_co_random_sample200")
    if smp:
        rows = []
        for o in smp["results"]:
            taxon = o.get("taxon") or {}
            photos = o.get("observation_photos") or o.get("photos") or []
            lic = [(ph.get("photo") or ph).get("license_code") for ph in photos]
            rows.append({"observation_id": o["id"], "n_photos": len(photos), "photo_ids": [ (ph.get("photo") or ph).get("id") for ph in photos],
                         "quality_grade": o.get("quality_grade"), "observed_on": o.get("observed_on"),
                         "location": o.get("location"), "positional_accuracy": o.get("positional_accuracy"),
                         "geoprivacy": o.get("geoprivacy"), "obscured": o.get("obscured"),
                         "taxon_rank": taxon.get("rank"), "taxon_name": taxon.get("name"),
                         "species_name": taxon.get("name") if taxon.get("rank") == "species" else None,
                         "genus_name": (taxon.get("name") or "").split()[0] if taxon.get("name") else None,
                         "user": (o.get("user") or {}).get("login"), "place_guess": o.get("place_guess"),
                         "photo_license": ",".join(sorted({l or "all-rights-reserved" for l in lic})),
                         "obs_license": o.get("license_code"),
                         "lat": (o.get("location") or ",").split(",")[0] or None,
                         "lon": (o.get("location") or ",").split(",")[-1] or None})
        df = pd.DataFrame(rows)
        for c in ("lat", "lon"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        t = AuditTable(ds, "random_sample200_research_photos", df, {
            "observation_id": "observation_id", "species": "species_name", "genus": "genus_name", "date": "observed_on",
            "latitude": "lat", "longitude": "lon", "license": "photo_license", "author": "user", "region": "place_guess"},
            "observation", evidence="DERIVED", is_sample=True,
            note="MUESTRA aleatoria de 200 observaciones (fungi, Colombia, grado de investigación, con fotos); "
                 "NO es el conjunto completo. La licencia es la de las fotos de cada observación")
        res.tables.append(t)
        res.facts["sample_photos_per_observation_mean"] = round(float(df["n_photos"].mean()), 3)
        res.facts["sample_photos_per_observation_max"] = int(df["n_photos"].max())
        res.facts["sample_n_obscured"] = int(df["obscured"].fillna(False).astype(bool).sum())
        res.facts["sample_photo_license_values"] = {str(k): int(v) for k, v in df["photo_license"].value_counts().items()}
    return res


# =========================================================================== GBIF
def _gbif_facet(query_id: str) -> list[dict]:
    d = _api("gbif", query_id)
    return d["facets"][0]["counts"] if d and d.get("facets") else []


def audit_gbif() -> AdapterResult:
    res = AdapterResult("gbif")
    ds = "gbif"
    res.facts.update({
        "fungi_world_records": _count(ds, "fungi_world_all"), "fungi_world_with_stillimage": _count(ds, "fungi_world_stillimage"),
        "fungi_colombia_records": _count(ds, "fungi_co_all"), "fungi_colombia_stillimage": _count(ds, "fungi_co_stillimage"),
        "fungi_colombia_stillimage_with_coords": _count(ds, "fungi_co_stillimage_coords"),
        "fungi_colombia_stillimage_without_coords": _count(ds, "fungi_co_stillimage_no_coords"),
        "fungi_colombia_stillimage_from_inaturalist_dataset": _count(ds, "fungi_co_inat_dataset_stillimage"),
        "fungi_published_by_colombia_stillimage": _count(ds, "fungi_pubco_stillimage"),
        "license_counts_colombia_stillimage": {c["name"]: c["count"] for c in _gbif_facet("fungi_co_stillimage_facet_license")},
        "basis_of_record_colombia_stillimage": {c["name"]: c["count"] for c in _gbif_facet("fungi_co_stillimage_facet_basisOfRecord")},
    })
    titles = _api(ds, "dataset_titles_index") or {}
    res.facts["top_datasets_colombia_stillimage"] = [
        {"datasetKey": c["name"], "records": c["count"], "title": (titles.get(c["name"]) or {}).get("title"),
         "license": (titles.get(c["name"]) or {}).get("license")} for c in _gbif_facet("fungi_co_stillimage_facet_datasetKey")[:12]]
    res.facts["top_datasets_published_in_colombia_stillimage"] = [
        {"datasetKey": c["name"], "records": c["count"], "title": (titles.get(c["name"]) or {}).get("title"),
         "license": (titles.get(c["name"]) or {}).get("license")} for c in _gbif_facet("fungi_pubco_stillimage_facet_datasetKey")[:12]]
    reg = {}
    for qid in ("registry_search_mitu", "registry_search_jbb", "registry_search_hongos_col"):
        r = _api(ds, qid)
        reg[qid] = [(x["title"], x["key"]) for x in (r or {}).get("results", [])][:5]
    res.facts["registry_search_results"] = reg
    # class counts from facets (records with StillImage per taxon key)
    names = _api(ds, "top_genus_names_index") or {}
    for level, qid in (("species", "fungi_co_stillimage_facet_speciesKey"), ("genus", "fungi_co_stillimage_facet_genusKey"),
                       ("family", "fungi_co_stillimage_facet_familyKey")):
        cs = _gbif_facet(qid)
        if cs:
            taxa = [f"{(names.get(c['name']) or {}).get('canonicalName', '')} [{c['name']}]".strip() for c in cs]
            res.class_counts.append(_counts_frame(ds, "colombia_stillimage_records", level, taxa, n_obs=[c["count"] for c in cs]))
    # sample records
    recs = []
    for off in (0, 6000, 12000):
        page = _api(ds, f"fungi_co_stillimage_sample300_offset{off}")
        if page:
            recs += page["results"]
    if recs:
        rows = []
        for r in recs:
            media = r.get("media") or []
            rows.append({"gbifID": r.get("gbifID"), "datasetKey": r.get("datasetKey"), "species": r.get("species"),
                         "genus": r.get("genus"), "family": r.get("family"), "order": r.get("order"),
                         "eventDate": r.get("eventDate"), "lat": r.get("decimalLatitude"), "lon": r.get("decimalLongitude"),
                         "country": r.get("country"), "region": r.get("stateProvince"), "license": r.get("license"),
                         "recordedBy": r.get("recordedBy"), "publisher": r.get("publishingOrgKey"),
                         "basisOfRecord": r.get("basisOfRecord"), "habitat": r.get("habitat"),
                         "n_media": len(media), "n_stillimage": sum(1 for m in media if m.get("type") == "StillImage"),
                         "media_url": (media[0].get("identifier") if media else None),
                         "media_urls_all": [m.get("identifier") for m in media],
                         "media_license": (media[0].get("license") if media else None),
                         "media_creator": (media[0].get("creator") or media[0].get("rightsHolder")) if media else None})
        df = pd.DataFrame(rows)
        all_urls = pd.Series([u for lst in df["media_urls_all"] for u in lst if u])
        vc = all_urls.value_counts()
        t = AuditTable(ds, "colombia_stillimage_sample900", df.drop(columns=["media_urls_all"]), {
            "observation_id": "gbifID", "image_file": None, "url": "media_url", "species": "species", "genus": "genus",
            "family": "family", "order": "order", "date": "eventDate", "latitude": "lat", "longitude": "lon",
            "country": "country", "region": "region", "habitat": "habitat", "license": "media_license",
            "author": "media_creator", "provider": "datasetKey"}, "record", evidence="DERIVED", is_sample=True,
            note="MUESTRA de 900 registros (3 páginas de 300 en desplazamientos 0/6000/12000) de fungi con StillImage en "
                 "Colombia; NO es el conjunto completo ni una muestra aleatoria estricta")
        t.roles = {k: v for k, v in t.roles.items() if v}
        t.extra_checks.update({"media_url_repeats_in_sample": int((vc > 1).sum()), "media_items_in_sample": int(len(all_urls))})
        res.tables.append(t)
        res.facts["sample_records"] = len(df)
        res.facts["sample_images_per_record_mean"] = round(float(df["n_stillimage"].mean()), 3)
        res.facts["sample_images_per_record_max"] = int(df["n_stillimage"].max())
        res.facts["sample_records_without_species"] = int(df["species"].isna().sum())
        res.facts["sample_media_license_values"] = {str(k): int(v) for k, v in df["media_license"].value_counts(dropna=False).items()}
    return res


# =========================================================================== SiB Colombia (IPT DwC-A)
SIB_RESOURCES = {
    "sib_macrohongosmitu": ("macrohongosmitu", "Mitú-Cachivera"),
    "sib_macrohongos_jbb_2023_1": ("macrohongos_2023-1", "Fungario JBB 2023-1"),
    "sib_hongos_col2023": ("hongos_col2023", "Hongos de Colombia 2023"),
}


def _dwca_records(path: Path) -> tuple[pd.DataFrame, str, list[str]]:
    with zipfile.ZipFile(path) as z:
        occ = pd.read_csv(z.open("occurrence.txt"), sep="\t", dtype=str, keep_default_na=False, quoting=3)
        meta = z.read("meta.xml").decode("utf-8", "replace")
        eml = z.read("eml.xml").decode("utf-8", "replace") if "eml.xml" in z.namelist() else ""
        names = z.namelist()
    return occ, eml, names


def audit_sib_resource(dataset_id: str) -> AdapterResult:
    res = AdapterResult(dataset_id)
    short, _ = SIB_RESOURCES[dataset_id]
    path = RAW_DIR / dataset_id / f"{short}.dwca.zip"
    if not path.exists():
        res.problems.append(f"{rel(path)} not found")
        return res
    occ, eml, names = _dwca_records(path)
    occ = occ.copy()
    published = set(occ.columns)          # Darwin Core terms actually published in this archive
    for c in ("decimalLatitude", "decimalLongitude", "genus", "specificEpithet", "associatedMedia", "license", "catalogNumber",
              "occurrenceID", "taxonRank", "basisOfRecord", "kingdom", "country", "eventDate", "year", "month", "day",
              "verbatimEventDate", "organismName", "scientificName", "family", "order"):
        if c not in occ.columns:   # some IPT resources publish a reduced set of Darwin Core terms
            occ[c] = pd.NA
    occ = occ.fillna("")
    n_comma = int(occ["decimalLatitude"].str.contains(",", regex=False).sum() + occ["decimalLongitude"].str.contains(",", regex=False).sum())
    for c in ("decimalLatitude", "decimalLongitude"):
        occ[c + "_num"] = pd.to_numeric(occ[c].str.replace(",", ".", regex=False), errors="coerce")
    # DERIVED fields: when the standard Darwin Core terms are empty, fall back to free-text/legacy fields of the same record
    org = occ["organismName"].str.extract(r"^\s*(?P<fam>[^-]+?)\s*-\s*(?P<tax>.+?)\s*$")
    toks = org["tax"].fillna("").str.split()
    g_org = toks.str[0].fillna("").str.capitalize()
    e_org = toks.str[1].fillna("")
    sp_ok = (g_org != "") & (e_org != "") & ~e_org.str.lower().str.strip(".").isin(["sp", "spp", "cf", "aff"])
    occ["family_derived"] = occ["family"].where(occ["family"] != "", org["fam"].fillna("").str.strip().str.capitalize())
    occ["genus_derived"] = occ["genus"].where(occ["genus"] != "", g_org)
    occ["species_derived"] = np.where((occ["genus"] != "") & (occ["specificEpithet"] != ""), occ["genus"] + " " + occ["specificEpithet"],
                                      np.where(sp_ok, g_org + " " + e_org.str.lower(), ""))
    occ["taxon_label_derived"] = occ["scientificName"].where(occ["scientificName"] != "", occ["organismName"])
    ymd = occ["year"] + "-" + occ["month"].str.zfill(2) + "-" + occ["day"].str.zfill(2)
    occ["date_derived"] = occ["eventDate"].where(occ["eventDate"] != "", ymd.where(occ["year"] != "", ""))
    vm = occ["verbatimEventDate"].str.extract(r"^(?P<d>\d{1,2})[-/](?P<m>\d{1,2})[-/](?P<y>\d{4})$")
    both = vm["y"].notna() & (occ["year"] != "")
    mismatch = both & ((vm["y"] != occ["year"]) | (vm["m"].fillna("").str.lstrip("0") != occ["month"].str.lstrip("0")) | (vm["d"].fillna("").str.lstrip("0") != occ["day"].str.lstrip("0")))
    t = AuditTable(dataset_id, "occurrence", occ, {
        "observation_id": "occurrenceID", "image_file": "associatedMedia", "taxon_label": "taxon_label_derived",
        "species": "species_derived", "genus": "genus_derived", "family": "family_derived", "order": "order", "date": "date_derived",
        "latitude": "decimalLatitude_num", "longitude": "decimalLongitude_num", "country": "country",
        "region": "stateProvince", "habitat": "habitat", "license": "license", "author": "recordedBy",
        "provider": "institutionCode"}, "record",
        note="registros de ocurrencia (especímenes/observaciones) del DwC-A; NO hay extensión multimedia. Género/familia/especie/fecha "
             "DERIVADOS de organismName y year/month/day cuando los términos estándar están vacíos")
    t.extra_checks.update({
        "decimal_comma_coordinates": n_comma,
        "catalogNumber_repeats": int(occ["catalogNumber"].replace("", np.nan).dropna().duplicated().sum()),
        "occurrenceID_repeats": int(occ["occurrenceID"].replace("", np.nan).dropna().duplicated().sum()),
        "date_from_year_month_day": int(((occ["eventDate"] == "") & (occ["year"] != "")).sum()),
        "taxon_from_organismName": int(((occ["scientificName"] == "") & (occ["organismName"] != "")).sum()),
        "verbatimEventDate_disagrees_with_year_month_day": int(mismatch.sum()),
        "verbatimEventDate_comparable": int(both.sum()),
    })
    res.facts["dwc_terms_published"] = len(published)
    res.facts["date_from_year_month_day"] = t.extra_checks["date_from_year_month_day"]
    res.facts["taxon_from_organismName"] = t.extra_checks["taxon_from_organismName"]
    res.facts["verbatimEventDate_disagrees_with_year_month_day"] = t.extra_checks["verbatimEventDate_disagrees_with_year_month_day"]
    res.facts["verbatimEventDate_comparable"] = t.extra_checks["verbatimEventDate_comparable"]
    res.facts["dwc_standard_terms_absent"] = sorted(c for c in ("associatedMedia", "habitat", "license", "catalogNumber") if c not in published)
    t.roles = {k: v for k, v in t.roles.items() if v in occ.columns}
    res.tables.append(t)
    res.facts.update({
        "files_in_archive": names, "n_records": len(occ), "has_multimedia_extension": any("multimedia" in n.lower() for n in names),
        "n_associatedMedia_nonempty": int((occ["associatedMedia"] != "").sum()),
        "basisOfRecord_values": occ["basisOfRecord"].value_counts().to_dict(),
        "taxonRank_values": occ["taxonRank"].value_counts().to_dict(),
        "record_level_license_nonempty": int((occ["license"] != "").sum()),
        "eml_licence_urls": sorted(set(re.findall(r"https?://creativecommons\.org/[^\s<\"]+|https?://[^\s<\"]*creativecommons[^\s<\"]*", eml)))[:3],
        "kingdom_values": occ["kingdom"].value_counts().to_dict(), "n_genera": int(occ.loc[occ["genus_derived"] != "", "genus_derived"].nunique()),
        "n_species_derived": int(occ.loc[occ["species_derived"] != "", "species_derived"].nunique()),
        "country_values": occ["country"].value_counts().to_dict(),
    })
    return res


# =========================================================================== Mushroom Observer
def _mo(name: str, **kw) -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / "mushroom_observer" / f"{name}.csv", sep="\t", na_values=["NULL"], low_memory=False, **kw)
    df.columns = [c.strip() for c in df.columns]   # observations.csv has a header ' alt' with a leading space
    return df


def audit_mushroom_observer() -> AdapterResult:
    res = AdapterResult("mushroom_observer")
    if not (RAW_DIR / "mushroom_observer" / "observations.csv").exists():
        res.problems.append("MO CSV dumps not found; run scripts/data/acquire_datasets.py mushroom_observer")
        return res
    obs, names, cls = _mo("observations"), _mo("names"), _mo("name_classifications")
    io_, img, loc = _mo("images_observations"), _mo("images"), _mo("locations")
    res.facts.update({"n_observations_all": len(obs), "n_images_all": len(img), "n_links_image_observation": len(io_),
                      "n_names": len(names), "n_locations": len(loc), "images_columns": list(img.columns),
                      "observations_columns": list(obs.columns), "images_license_values": img["license"].value_counts(dropna=False).to_dict(),
                      "images_ok_for_export_0": int((img["ok_for_export"] == 0).sum())})
    n = names.rename(columns={"id": "name_id", "text_name": "name_text", "rank": "name_rank"})
    o = obs.rename(columns={"id": "observation_id", "when": "obs_date"}).merge(
        n[["name_id", "name_text", "name_rank", "deprecated"]], on="name_id", how="left").merge(
        cls.drop_duplicates("name_id"), on="name_id", how="left").merge(
        loc.rename(columns={"id": "location_id", "name": "location_name"})[["location_id", "location_name"]],
        on="location_id", how="left")
    o["country_derived"] = o["location_name"].astype("string").str.rsplit(",", n=1).str[-1].str.strip()
    o["species_derived"] = o["name_text"].where(o["name_rank"] == 400)            # rank 400 == species (verified empirically)
    tok = _first_token(o["name_text"])
    o["genus_derived"] = tok.where(o["name_rank"].isin([100, 200, 300, 400, 410, 500]))
    o["family_derived"] = o["family"].where(o["family"].notna() & (o["family"].astype(str) != ""))
    fungi = o[o["kingdom"] == "Fungi"].copy()
    res.facts.update({"n_observations_fungi_kingdom": len(fungi),
                      "n_observations_kingdom_missing": int(o["kingdom"].isna().sum() + (o["kingdom"] == "").sum()),
                      "n_observations_deprecated_name": int((fungi["deprecated"] == 1).sum()),
                      "kingdom_values_top": o["kingdom"].fillna("<NA>").value_counts().head(6).to_dict()})
    # images per observation
    per = io_.groupby("observation_id")["image_id"].nunique().rename("n_images")
    fungi = fungi.merge(per, on="observation_id", how="left")
    fungi["n_images"] = fungi["n_images"].fillna(0).astype(int)
    res.facts["n_fungi_observations_with_images"] = int((fungi["n_images"] > 0).sum())
    res.facts["n_fungi_observations_without_images"] = int((fungi["n_images"] == 0).sum())
    is_co = fungi["country_derived"].eq("Colombia")
    res.facts["n_locations_colombia"] = int(loc["name"].astype("string").str.endswith("Colombia").sum())
    img2 = img.rename(columns={"id": "image_id", "copyright_holder": "author"})
    link = io_.merge(img2[["image_id", "license", "author", "ok_for_export"]], on="image_id", how="left")
    fl = fungi[["observation_id", "species_derived", "genus_derived", "family_derived", "country_derived"]]
    fimg = link.merge(fl, on="observation_id", how="inner")
    res.facts["n_links_image_fungi_observation"] = len(fimg)
    roles_obs = {"observation_id": "observation_id", "taxon_label": "name_text", "species": "species_derived",
                 "genus": "genus_derived", "family": "family_derived", "order": "order", "date": "obs_date",
                 "latitude": "lat", "longitude": "lng", "country": "country_derived", "region": "location_name"}
    for name, sub_o, sub_i in (("fungi_all", fungi, fimg), ("fungi_colombia", fungi[is_co], fimg[fimg["country_derived"].eq("Colombia")])):
        t = AuditTable("mushroom_observer", f"{name}_observations", sub_o, roles_obs, "observation",
                       note="observaciones cuyo taxón está clasificado en Kingdom=Fungi; especie/género DERIVADOS de names.csv "
                            "(rank 400=especie); país DERIVADO del último término del nombre de localidad")
        t.extra_checks["observation_id_repeats"] = int(sub_o["observation_id"].duplicated().sum())
        res.tables.append(t)
        t2 = AuditTable("mushroom_observer", f"{name}_images", sub_i, {
            "observation_id": "observation_id", "image_id": "image_id", "species": "species_derived", "genus": "genus_derived",
            "family": "family_derived", "license": "license", "author": "author", "country": "country_derived"}, "image",
            note="pares imagen–observación de images_observations.csv unidos a images.csv (licencia/autor por imagen)")
        t2.extra_checks["image_ids_linked_to_more_than_one_observation"] = int(
            (sub_i.groupby("image_id")["observation_id"].nunique() > 1).sum())
        t2.extra_checks["exact_duplicate_image_observation_pairs"] = int(sub_i[["image_id", "observation_id"]].duplicated().sum())
        res.tables.append(t2)
    res.facts["n_colombia_fungi_observations"] = int(is_co.sum())
    res.facts["n_colombia_fungi_observations_with_images"] = int((fungi[is_co]["n_images"] > 0).sum())
    return res
