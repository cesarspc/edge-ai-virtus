"""Source loaders of the processed dataset (Week 3).

Each ``load_<source>(cfg, log)`` reads the raw files (read-only), applies the cleaning rules of ``cleaning.py``,
records their effect in the :class:`CleaningLog` and returns two frames in the common schema:

* observations: observation fields (without split/group/label, added later) + environmental groups;
* images: image fields (without split/label).
"""
from __future__ import annotations

import glob
import json
import zipfile
from typing import Any

import numpy as np
import pandas as pd

import adapters as A
import cleaning as K
from common import RAW_DIR, log as logger

GEO_VARS = ["latitude", "longitude", "coordinate_uncertainty_m"]
CLIMATE_VARS = ["climate_temp_mean_c", "climate_precip_mm", "climate_rel_humidity_pct", "climate_window"]


def _n(df: pd.DataFrame, obs_col: str) -> tuple[int, int]:
    return int(df[obs_col].nunique()), len(df)


def _geo_group(lat, lon, unc, source: str, method: pd.Series, invalid: pd.Series, max_unc: float,
               extra_low: pd.Series | None = None, suspect: pd.Series | None = None) -> dict[str, pd.Series]:
    idx = lat.index
    has = lat.notna() & lon.notna()
    low = has & (unc > max_unc).fillna(False)
    if extra_low is not None:
        low |= has & extra_low
    q = np.select([invalid, ~has, has & suspect if suspect is not None else pd.Series(False, index=idx), low],
                  ["INVALID_REMOVED", "MISSING", "SUSPECT", "LOW_PRECISION"], "OK")
    return {"latitude": lat.where(has), "longitude": lon.where(has), "coordinate_uncertainty_m": unc.where(has).round(1),
            "geo__source": pd.Series(source, index=idx), "geo__method": method.where(has, "NOT_AVAILABLE"),
            "geo__quality": pd.Series(q, index=idx)}


def _elevation_group(elev: pd.Series, source: str, method: str) -> tuple[dict[str, pd.Series], pd.Series, pd.Series]:
    e = pd.to_numeric(elev, errors="coerce")
    bad = e.notna() & ((e < -500) | (e > 9000))
    zero = e == 0
    e = e.where(~bad)
    q = np.select([bad, e.isna(), zero], ["INVALID_REMOVED", "MISSING", "SUSPECT"], "OK")
    return ({"elevation_m": e, "elevation__source": pd.Series(source, index=e.index),
             "elevation__method": pd.Series(method, index=e.index).where(e.notna(), "NOT_AVAILABLE"),
             "elevation__quality": pd.Series(q, index=e.index)}, bad, zero)


def log_env(log: K.CleaningLog, sid: str, obs: pd.DataFrame) -> None:
    """E-rules are counted on the final observations of each source (after exclusions)."""
    q = lambda g, v: int((obs[f"{g}__quality"] == v).sum())  # noqa: E731
    log.add("E1", sid, q("geo", "INVALID_REMOVED"))
    log.add("E2", sid, q("geo", "LOW_PRECISION"))
    log.add("E3", sid, q("event_date", "INVALID_REMOVED"))
    log.add("E4", sid, q("event_date", "LOW_PRECISION"))
    log.add("E5", sid, q("elevation", "SUSPECT") + q("elevation", "INVALID_REMOVED"),
            note=f"SUSPECT={q('elevation', 'SUSPECT')}; INVALID_REMOVED={q('elevation', 'INVALID_REMOVED')}")
    log.add("E8", sid, len(obs), note=f"climate__quality: {obs['climate__quality'].iloc[0] if len(obs) else ''}")


def _finish(obs: pd.DataFrame, img: pd.DataFrame, sid: str, log: K.CleaningLog) -> tuple[pd.DataFrame, pd.DataFrame]:
    """D7: drop observations left without images; set n_images."""
    n = img.groupby("observation_uid").size()
    empty = ~obs["observation_uid"].isin(n.index)
    log.add("D7", sid, int(empty.sum()), 0)
    obs = obs[~empty].copy()
    obs["n_images"] = obs["observation_uid"].map(n).astype(int)
    img = img[img["observation_uid"].isin(obs["observation_uid"])].copy()
    return obs.reset_index(drop=True), img.reset_index(drop=True)


# =========================================================================== FungiTastic (development)
FT_TEMPORAL = {"Train": "train", "ClosedSet-Val": "val", "OpenSet-Val": "val",
               "ClosedSet-Test": "test", "OpenSet-Test": "test", "DNA-Test": "test"}


def load_fungitastic(cfg: dict[str, Any], log: K.CleaningLog) -> tuple[pd.DataFrame, pd.DataFrame]:
    sid, classes, c = "fungitastic", cfg["classes"], cfg["cleaning"]
    frames = []
    with zipfile.ZipFile(RAW_DIR / "fungitastic" / "metadata.zip") as z:
        for name in cfg["sources"][sid]["files"]:
            with z.open(A.FT_FULL_FILES[name]) as fh:
                frames.append(pd.read_csv(fh, usecols=lambda col: col != "captions", low_memory=False).assign(_file=name))
    df = pd.concat(frames, ignore_index=True)
    logger.info("FungiTastic: %d rows in %d files", len(df), len(frames))
    # T6 kingdom
    bad = df["kingdom"] != "Fungi"
    log.add("T6", sid, int(df.loc[bad, "observationID"].nunique()), int(bad.sum()))
    df = df[~bad]
    # N1/T2/T3/T1 genus
    g, syn, unc = K.normalise_genus(df["genus"], cfg["genus_synonyms"])
    df = df.assign(genus_norm=g)
    log.add("T2", sid, int(df.loc[syn & g.isin(classes), "observationID"].nunique()), int((syn & g.isin(classes)).sum()))
    bad = unc | g.isna()
    log.add("T3", sid, int(df.loc[bad, "observationID"].nunique()), int(bad.sum()))
    df = df[~bad]
    out = ~df["genus_norm"].isin(classes)
    log.add("T1", sid, int(df.loc[out, "observationID"].nunique()), int(out.sum()),
            note="filas de los 6 CSV completos (antes de quitar las repetidas por diseño ClosedSet ⊂ OpenSet)")
    df = df[~out]
    # D1: ClosedSet ⊂ OpenSet => the same image appears in two CSV by design; keep the first file in config order
    dup = df["filename"].duplicated(keep="first")
    log.add("D1", sid, 0, int(dup.sum()), note="filas repetidas entre CSV por diseño oficial (ClosedSet ⊂ OpenSet); no son imágenes distintas")
    df = df[~dup].copy()
    # T4 conflicting genus within an observation
    conf = df.groupby("observationID")["genus_norm"].transform("nunique") > 1
    log.add("T4", sid, int(df.loc[conf, "observationID"].nunique()), int(conf.sum()))
    df = df[~conf]

    o = df.sort_values("filename").groupby("observationID", sort=True).first().reset_index()
    idx = o.index
    lat, lon, inv = K.valid_coords(o["latitude"], o["longitude"])
    env: dict[str, pd.Series] = {}
    env.update(_geo_group(lat, lon, pd.to_numeric(o["coorUncert"], errors="coerce"), f"dataset:{sid}",
                          pd.Series("POINT_REPORTED", index=idx), inv, c["max_coordinate_uncertainty_m"]))
    d, p, dinv = K.parse_dates(o["eventDate"], c["min_year"], cfg["access_date"])
    env.update(K.date_group(d, p, f"dataset:{sid}", "OBSERVER_RECORDED", dinv))
    eg, _, _ = _elevation_group(o["elevation"], f"dataset:{sid}", "SOURCE_DERIVED_UNDOCUMENTED")
    env.update(eg)
    # substrate: detailed value, else coarse metaSubstrate (E6)
    detailed = o["substrate"].map(K.SUBSTRATE_MAP)
    coarse = o["metaSubstrate"].map(K.META_SUBSTRATE_MAP)
    raw = o["substrate"].where(o["substrate"].notna(), o["metaSubstrate"])
    cls = detailed.where(o["substrate"].notna(), coarse)
    unmapped = raw.notna() & cls.isna()
    cls = cls.where(~unmapped, "other")
    q = np.select([raw.isna(), unmapped, o["substrate"].isna() & o["metaSubstrate"].isin(K.META_SUBSTRATE_TRANSLATED),
                   o["substrate"].isna()], ["MISSING", "SUSPECT", "CORRECTED", "LOW_PRECISION"], "OK")
    env.update({"substrate_class": cls, "substrate_raw": raw, "substrate__source": pd.Series(f"dataset:{sid}", index=idx),
                "substrate__method": pd.Series("OBSERVER_RECORDED", index=idx).where(raw.notna(), "NOT_AVAILABLE"),
                "substrate__quality": pd.Series(q, index=idx)})
    log.add("E6", sid, int((q == "LOW_PRECISION").sum()) + int((q == "CORRECTED").sum()),
            note=f"sin sustrato detallado: {int(o['substrate'].isna().sum())}; sin ningún sustrato: {int(raw.isna().sum())}")
    hcls = o["habitat"].map(K.HABITAT_MAP)
    hun = o["habitat"].notna() & hcls.isna()
    env.update({"habitat_class": hcls.where(~hun, "other"), "habitat_raw": o["habitat"],
                "habitat__source": pd.Series(f"dataset:{sid}", index=idx),
                "habitat__method": pd.Series("OBSERVER_RECORDED", index=idx).where(o["habitat"].notna(), "NOT_AVAILABLE"),
                "habitat__quality": pd.Series(np.select([o["habitat"].isna(), hun], ["MISSING", "SUSPECT"], "OK"), index=idx)})
    lc = pd.to_numeric(o["landcover"], errors="coerce").round().astype("Int64")
    env.update({"landcover_code": lc, "landcover__source": pd.Series(f"dataset:{sid}", index=idx),
                "landcover__method": pd.Series("SOURCE_DERIVED_UNDOCUMENTED", index=idx).where(lc.notna(), "NOT_AVAILABLE"),
                "landcover__quality": pd.Series(np.where(lc.isna(), "MISSING", "OK"), index=idx)})
    br = o["biogeographicalRegion"]
    env.update({"biogeographical_region": br, "biogeographical_region__source": pd.Series(f"dataset:{sid}", index=idx),
                "biogeographical_region__method": pd.Series("SOURCE_DERIVED_UNDOCUMENTED", index=idx).where(br.notna(), "NOT_AVAILABLE"),
                "biogeographical_region__quality": pd.Series(np.where(br.isna(), "MISSING", "OK"), index=idx)})
    env.update(K.missing_group("climate", CLIMATE_VARS, idx))   # E8: climatic.zip -> HTTP 403

    obs = pd.DataFrame({
        "observation_uid": "fungitastic:" + o["observationID"].astype(str), "source_id": sid,
        "source_observation_id": o["observationID"].astype(str), "genus": o["genus_norm"],
        "taxon_name_raw": o["scientificName"], "identification_basis": "expert_curated",
        "split_temporal": o["_file"].map(FT_TEMPORAL), "country_code": o["countryCode"], **env})
    img = pd.DataFrame({
        "image_uid": "fungitastic:" + df["filename"], "observation_uid": "fungitastic:" + df["observationID"].astype(str),
        "source_id": sid, "source_image_id": df["filename"], "locator": df["_file"] + "/" + df["filename"],
        "license": "CC-BY-NC-SA", "license_scope": "dataset", "rights_holder": pd.NA, "sha256_reported": pd.NA,
        "genus": df["genus_norm"]})
    return _finish(obs, img, sid, log)


# =========================================================================== Mushroom Observer (Colombia, external test)
def load_mushroom_observer(cfg: dict[str, Any], log: K.CleaningLog) -> tuple[pd.DataFrame, pd.DataFrame]:
    sid, classes, c = "mushroom_observer_co", cfg["classes"], cfg["cleaning"]
    scfg = cfg["sources"][sid]
    obs, names, cls = A._mo("observations"), A._mo("names"), A._mo("name_classifications")
    io_, img, loc = A._mo("images_observations"), A._mo("images"), A._mo("locations")
    loc = loc.rename(columns={"id": "location_id", "name": "location_name"})
    o = obs.rename(columns={"id": "obs_id", "when": "obs_date"}).merge(loc, on="location_id", how="left")
    o["country"] = o["location_name"].astype("string").str.rsplit(",", n=1).str[-1].str.strip()
    o = o[o["country"] == scfg["country"]].copy()
    logger.info("Mushroom Observer: %d observations in %s", len(o), scfg["country"])
    # name resolution: deprecated names -> accepted synonym of the same rank (T2)
    n = names.set_index("id")
    accepted = names[names["deprecated"] == 0].dropna(subset=["synonym_id"])
    acc_by_syn = accepted.groupby(["synonym_id", "rank"])["id"].agg(lambda s: s.iloc[0] if len(s) == 1 else np.nan)
    def resolve(nid: int) -> int:
        row = n.loc[nid]
        if row["deprecated"] == 1 and pd.notna(row["synonym_id"]):
            a = acc_by_syn.get((row["synonym_id"], row["rank"]), np.nan)
            if pd.notna(a):
                return int(a)
        return nid
    o["name_id_resolved"] = o["name_id"].map(resolve)
    o["name_raw"] = o["name_id"].map(n["text_name"])
    o["name_resolved"] = o["name_id_resolved"].map(n["text_name"])
    o["rank_resolved"] = o["name_id_resolved"].map(n["rank"])
    o["kingdom"] = o["name_id_resolved"].map(cls.drop_duplicates("name_id").set_index("name_id")["kingdom"])
    oid = "obs_id"
    bad = o["kingdom"] != "Fungi"
    log.add("T6", sid, int(bad.sum()))
    o = o[~bad]
    g_raw, _, _ = K.normalise_genus(o["name_raw"], {})
    g, syn, unc = K.normalise_genus(o["name_resolved"], cfg["genus_synonyms"])
    o = o.assign(genus_norm=g)
    changed = (g_raw != g) | syn
    log.add("T2", sid, int((changed & g.isin(classes)).sum()), note="nombres deprecated resueltos con synonym_id + tabla de sinónimos")
    bad = unc | (o["rank_resolved"] > 500) | g.isna()     # MO rank 500 = genus; > 500 = family and above
    log.add("T3", sid, int(bad.sum()))
    o = o[~bad]
    out = ~o["genus_norm"].isin(classes)
    log.add("T1", sid, int(out.sum()))
    o = o[~out]
    low = o["vote_cache"].fillna(-99) < scfg["min_vote_cache"]
    log.add("T5", sid, int(low.sum()), note=f"umbral vote_cache >= {scfg['min_vote_cache']}")
    o = o[~low].copy()

    # images
    links = io_.drop_duplicates()
    log.add("D1", sid, None, int(len(io_) - len(links)), note="pares imagen–observación repetidos en todo MO")
    multi = links.groupby("image_id")["observation_id"].transform("nunique") > 1
    li = links[links["observation_id"].isin(o[oid])]
    m_here = li["image_id"].isin(links.loc[multi, "image_id"])
    log.add("D2", sid, None, int(m_here.sum()))
    li = li[~m_here].merge(img.rename(columns={"id": "image_id"}), on="image_id", how="left")
    no_export = li["ok_for_export"] != 1
    log.add("L2", sid, None, int(no_export.sum()))
    li = li[~no_export].copy()
    li["license_norm"] = K.normalise_license(li["license"])
    bad_lic = ~li["license_norm"].isin(cfg["allowed_licenses"])
    log.add("L1", sid, None, int(bad_lic.sum()), note=str(li.loc[bad_lic, "license_norm"].value_counts().to_dict()))
    li = li[~bad_lic]

    idx = o.index
    lat, lon, inv = K.valid_coords(o["lat"], o["lng"])
    blat, blon, bunc = K.box_centroid(o["north"], o["south"], o["east"], o["west"])
    point = lat.notna() & lon.notna()
    lat2, lon2 = lat.where(point, blat.round(6)), lon.where(point, blon.round(6))
    unc = pd.Series(np.nan, index=idx).where(point, bunc)
    method = pd.Series(np.where(point, "POINT_REPORTED", "LOCALITY_CENTROID"), index=idx)
    env: dict[str, pd.Series] = {}
    env.update(_geo_group(lat2, lon2, unc, f"dataset:mushroom_observer", method, inv & ~lat2.notna(),
                          c["max_coordinate_uncertainty_m"], extra_low=None, suspect=o["is_collection_location"] == 0))
    log.add("E7", sid, int((env["geo__quality"] == "SUSPECT").sum()))
    d, p, dinv = K.parse_dates(o["obs_date"], c["min_year"], cfg["access_date"])
    env.update(K.date_group(d, p, "dataset:mushroom_observer", "DEVICE_TIMESTAMP_OR_USER", dinv))
    eg, _, _ = _elevation_group(o["alt"], "dataset:mushroom_observer", "OBSERVER_RECORDED")
    env.update(eg)
    env.update(K.not_provided("substrate", ["substrate_class", "substrate_raw"], idx))
    env.update(K.not_provided("habitat", ["habitat_class", "habitat_raw"], idx))
    env.update(K.not_provided("landcover", ["landcover_code"], idx))
    env.update(K.not_provided("biogeographical_region", ["biogeographical_region"], idx))
    env.update(K.not_provided("climate", CLIMATE_VARS, idx))
    obs_df = pd.DataFrame({
        "observation_uid": "mushroom_observer:" + o[oid].astype(str), "source_id": sid,
        "source_observation_id": o[oid].astype(str), "genus": o["genus_norm"], "taxon_name_raw": o["name_raw"],
        "identification_basis": "community_consensus", "split_temporal": pd.NA,
        "country_code": o["country"].map(K.COUNTRY_ISO2), "vote_cache": o["vote_cache"], **env})
    genus_by_obs = obs_df.set_index("source_observation_id")["genus"]
    img_df = pd.DataFrame({
        "image_uid": "mushroom_observer:" + li["image_id"].astype(str),
        "observation_uid": "mushroom_observer:" + li["observation_id"].astype(str), "source_id": sid,
        "source_image_id": li["image_id"].astype(str),
        "locator": "https://mushroomobserver.org/images/640/" + li["image_id"].astype(str) + ".jpg",
        "license": li["license_norm"], "license_scope": "image", "rights_holder": li["copyright_holder"],
        "sha256_reported": pd.NA, "genus": li["observation_id"].astype(str).map(genus_by_obs)})
    return _finish(obs_df, img_df, sid, log)


# =========================================================================== iNaturalist (Colombia, external test)
def load_inaturalist(cfg: dict[str, Any], log: K.CleaningLog) -> tuple[pd.DataFrame, pd.DataFrame]:
    sid, c = "inaturalist_co", cfg["cleaning"]
    ids = cfg["inaturalist_genus_taxon_ids"]
    rows, photos, reported = [], [], {}
    for genus, tid in ids.items():
        for f in sorted(glob.glob(str(RAW_DIR / "inaturalist_api" / "api" / f"v001_co_{genus.lower()}_p*.json"))):
            d = json.loads(open(f, encoding="utf-8").read())
            reported[genus] = d["total_results"]
            for r in d["results"]:
                t = r.get("taxon") or {}
                in_genus = t.get("id") == tid or tid in (t.get("ancestor_ids") or [])
                ofv = {v["name"]: v.get("value") for v in r.get("ofvs", [])}
                coords = (r.get("geojson") or {}).get("coordinates") or [None, None]
                rows.append({"id": r["id"], "genus": genus if in_genus else None, "taxon_name": t.get("name"),
                             "rank": t.get("rank"), "observed_on": r.get("observed_on"), "lon": coords[0], "lat": coords[1],
                             "obscured": bool(r.get("obscured")), "acc": r.get("public_positional_accuracy") or r.get("positional_accuracy"),
                             "elev": ofv.get("Elevation (m)"), "mo_url": ofv.get("Mushroom Observer URL")})
                for ph in r.get("photos", []):
                    photos.append({"obs_id": r["id"], "photo_id": ph["id"], "license_code": ph.get("license_code"),
                                   "attribution": ph.get("attribution"), "url": ph.get("url")})
    o, ph = pd.DataFrame(rows), pd.DataFrame(photos)
    logger.info("iNaturalist: %d observation rows, %d photos (API total_results=%d)", len(o), len(ph), sum(reported.values()))
    dup = o["id"].duplicated()
    log.add("D1", sid, int(dup.sum()), int(ph.duplicated(["obs_id", "photo_id"]).sum()),
            note=f"total_results declarado por la API: {sum(reported.values())}; filas leídas: {len(o)}")
    o, ph = o[~dup].copy(), ph.drop_duplicates(["obs_id", "photo_id"])
    out = o["genus"].isna()
    log.add("T1", sid, int(out.sum()))
    o = o[~out].copy()
    ph = ph[ph["obs_id"].isin(o["id"])].copy()
    ph["license_norm"] = K.normalise_license(ph["license_code"])
    bad = ~ph["license_norm"].isin(cfg["allowed_licenses"])
    log.add("L1", sid, None, int(bad.sum()), note=str(ph.loc[bad, "license_norm"].value_counts().to_dict()))
    ph = ph[~bad]

    idx = o.index
    lat, lon, inv = K.valid_coords(o["lat"], o["lon"])
    method = pd.Series(np.where(o["obscured"], "POINT_OBSCURED", "POINT_REPORTED"), index=idx)
    env: dict[str, pd.Series] = {}
    env.update(_geo_group(lat, lon, pd.to_numeric(o["acc"], errors="coerce"), "dataset:inaturalist_api", method, inv,
                          c["max_coordinate_uncertainty_m"], extra_low=o["obscured"]))
    d, p, dinv = K.parse_dates(o["observed_on"], c["min_year"], cfg["access_date"])
    env.update(K.date_group(d, p, "dataset:inaturalist_api", "DEVICE_TIMESTAMP_OR_USER", dinv))
    eg, _, _ = _elevation_group(o["elev"], "dataset:inaturalist_api", "OBSERVER_RECORDED")
    # elevation is an optional observation field in iNaturalist: absent => NOT_PROVIDED rather than MISSING
    no_field = o["elev"].isna()
    eg["elevation__quality"] = eg["elevation__quality"].where(~no_field, "NOT_PROVIDED")
    eg["elevation__source"] = eg["elevation__source"].where(~no_field, "none")
    env.update(eg)
    env.update(K.not_provided("substrate", ["substrate_class", "substrate_raw"], idx))
    env.update(K.not_provided("habitat", ["habitat_class", "habitat_raw"], idx))
    env.update(K.not_provided("landcover", ["landcover_code"], idx))
    env.update(K.not_provided("biogeographical_region", ["biogeographical_region"], idx))
    env.update(K.not_provided("climate", CLIMATE_VARS, idx))
    obs = pd.DataFrame({
        "observation_uid": "inaturalist:" + o["id"].astype(str), "source_id": sid, "source_observation_id": o["id"].astype(str),
        "genus": o["genus"], "taxon_name_raw": o["taxon_name"], "identification_basis": "research_grade",
        "split_temporal": pd.NA, "country_code": "CO", "mo_url": o["mo_url"], **env})
    g = obs.set_index("source_observation_id")["genus"]
    img = pd.DataFrame({
        "image_uid": "inaturalist:" + ph["photo_id"].astype(str), "observation_uid": "inaturalist:" + ph["obs_id"].astype(str),
        "source_id": sid, "source_image_id": ph["photo_id"].astype(str),
        "locator": ph["url"].str.replace("/square.", "/medium.", regex=False), "license": ph["license_norm"],
        "license_scope": "image", "rights_holder": ph["attribution"], "sha256_reported": pd.NA,
        "genus": ph["obs_id"].astype(str).map(g)})
    # the same photo attached to two observations would break the observation grouping
    multi = img["image_uid"].duplicated(keep=False)
    log.add("D2", sid, None, int(multi.sum()))
    img = img[~multi]
    return _finish(obs, img, sid, log)


# =========================================================================== MIND.Funga v2 (image-only external test)
def load_mind_funga(cfg: dict[str, Any], log: K.CleaningLog) -> tuple[pd.DataFrame, pd.DataFrame]:
    sid, classes = "mind_funga_v2", cfg["classes"]
    res = A.audit_mind_funga()
    f = res.tables[0].df.copy()
    logger.info("MIND.Funga: %d files", len(f))
    g, syn, unc = K.normalise_genus(f["taxon_dir"], cfg["genus_synonyms"])
    f["genus_norm"] = g
    # D3 on the whole index (conflicts must be seen across genera): identical published SHA-256
    per_hash = f.groupby("sha256_reported")["genus_norm"].transform("nunique")
    conflict = per_hash > 1
    f = f.assign(_conflict=conflict).sort_values(["sha256_reported", "filename", "file_id"])
    conflict = f.pop("_conflict")
    redundant = f["sha256_reported"].duplicated(keep="first") & ~conflict
    in_cls = f["genus_norm"].isin(classes)
    log.add("D3", sid, None, int(((conflict | redundant) & in_cls).sum()),
            note=f"en las 10 clases: {int((conflict & in_cls).sum())} por grupos que cruzan géneros, {int((redundant & in_cls).sum())} copias redundantes")
    f = f[~(conflict | redundant)]
    bv = f["filename"].str.contains(r"\b(?:branco|verde)\b", case=False, regex=True)
    log.add("D6", sid, None, int((bv & f["genus_norm"].isin(classes)).sum()))
    f = f[~bv]
    log.add("T2", sid, None, int((syn.loc[f.index] & f["genus_norm"].isin(classes)).sum()))
    bad = unc.loc[f.index] | f["genus_norm"].isna()
    log.add("T3", sid, None, int(bad.sum()))
    f = f[~bad]
    out = ~f["genus_norm"].isin(classes)
    log.add("T1", sid, None, int(out.sum()))
    f = f[~out].sort_values("file_id").reset_index(drop=True)
    idx = f.index
    env: dict[str, pd.Series] = {}
    for grp, vars_ in (("geo", GEO_VARS), ("event_date", ["event_date", "event_date_precision"]), ("elevation", ["elevation_m"]),
                       ("substrate", ["substrate_class", "substrate_raw"]), ("habitat", ["habitat_class", "habitat_raw"]),
                       ("landcover", ["landcover_code"]), ("biogeographical_region", ["biogeographical_region"]),
                       ("climate", CLIMATE_VARS)):
        env.update(K.not_provided(grp, vars_, idx))
    uid = sid + ":" + f["file_id"].astype(str)
    obs = pd.DataFrame({"observation_uid": uid, "source_id": sid, "source_observation_id": f["file_id"].astype(str),
                        "genus": f["genus_norm"], "taxon_name_raw": f["taxon_dir"], "identification_basis": "folder_label",
                        "split_temporal": pd.NA, "country_code": pd.NA, **env})
    img = pd.DataFrame({"image_uid": uid, "observation_uid": uid, "source_id": sid, "source_image_id": f["file_id"].astype(str),
                        "locator": f["taxon_dir"] + "/" + f["filename"], "license": "CC-BY-NC", "license_scope": "dataset",
                        "rights_holder": pd.NA, "sha256_reported": f["sha256_reported"], "genus": f["genus_norm"]})
    return _finish(obs, img, sid, log)


LOADERS = {
    "fungitastic": load_fungitastic,
    "mushroom_observer_co": load_mushroom_observer,
    "inaturalist_co": load_inaturalist,
    "mind_funga_v2": load_mind_funga,
}
