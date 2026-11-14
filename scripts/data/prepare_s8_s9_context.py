"""Build a train/val-only context sidecar without changing published v002 tables.

python scripts/data/prepare_s8_s9_context.py
Reads only Train and ClosedSet-Val members, never test images or test metadata.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd

from common import REPO_ROOT, rel, sha256_file, utc_now
import cleaning as K
from curation_sources import _elevation_group, _geo_group
from adapters import FT_FULL_FILES

BIO = [f"Bio{i}" for i in range(1, 20)]
RAW_COLUMNS = ["observationID", "filename", "genus", "substrate", "metaSubstrate", "habitat", "elevation",
               "eventDate", "latitude", "longitude", "coorUncert"]


def read_cohort(metadata, splits):
    """Reproduce the curator's sorted first-non-null aggregation on frozen IDs."""
    frames, image_frames, consistency = [], [], []
    with zipfile.ZipFile(metadata) as z:
        for split, member in (("train", "Train"), ("val", "ClosedSet-Val")):
            ids = set(splits.loc[splits.split.eq(split), "observation_uid"])
            kept = []
            with z.open(FT_FULL_FILES[member]) as handle:
                for chunk in pd.read_csv(handle, usecols=RAW_COLUMNS, chunksize=50000):
                    chunk["observation_uid"] = "fungitastic:" + chunk.observationID.astype(str)
                    kept.append(chunk[chunk.observation_uid.isin(ids)])
            raw = pd.concat(kept, ignore_index=True)
            if raw.empty:
                raise ValueError(f"empty raw cohort: {split}")
            raw["split"] = split
            for field in RAW_COLUMNS:
                if field in ("observationID", "filename"):
                    continue
                conflicts = raw.groupby("observation_uid")[field].nunique(dropna=False).gt(1)
                consistency.append({"split": split, "field": field,
                                    "observations_with_discordant_metadata": int(conflicts.sum())})
            image_frames.append(raw[["observation_uid", "filename", "split"]].drop_duplicates())
            frames.append(raw.sort_values("filename").groupby("observation_uid", sort=True).first().reset_index())
    raw = pd.concat(frames, ignore_index=True)
    out = splits.merge(raw.drop(columns="split"), on="observation_uid", how="left", validate="one_to_one",
                       suffixes=("", "_raw"))
    if out.observationID.isna().any() or not out.genus.eq(out.genus_raw).all():
        raise ValueError("raw metadata missing or genus disagrees with frozen cohort")
    return out, pd.concat(image_frames, ignore_index=True), pd.DataFrame(consistency)


def clean_context(raw):
    idx = raw.index
    lat, lon, invalid = K.valid_coords(raw.latitude, raw.longitude)
    out = raw[["observation_uid", "split", "label_id"]].copy()
    geo = _geo_group(lat, lon, pd.to_numeric(raw.coorUncert, errors="coerce"), "dataset:fungitastic",
                     pd.Series("POINT_REPORTED", index=idx), invalid, 10000)
    for key, value in geo.items():
        out[key] = value
    return out


def audit_metadata(raw):
    """Same quality rules as v002; no source assertions inferred from correlation."""
    out = raw.copy()
    out["substrate_class"] = raw.substrate.map(K.SUBSTRATE_MAP).where(
        raw.substrate.notna(), raw.metaSubstrate.map(K.META_SUBSTRATE_MAP))
    value = raw.substrate.where(raw.substrate.notna(), raw.metaSubstrate)
    unmapped = value.notna() & out.substrate_class.isna()
    out["substrate_class"] = out.substrate_class.where(~unmapped, "other")
    out["substrate__quality"] = np.select(
        [value.isna(), unmapped, raw.substrate.isna() & raw.metaSubstrate.isin(K.META_SUBSTRATE_TRANSLATED),
         raw.substrate.isna()], ["MISSING", "SUSPECT", "CORRECTED", "LOW_PRECISION"], "OK")
    habitat = raw.habitat.map(K.HABITAT_MAP)
    out["habitat__quality"] = np.select([raw.habitat.isna(), habitat.isna()], ["MISSING", "SUSPECT"], "OK")
    elev, _, _ = _elevation_group(raw.elevation, "dataset:fungitastic", "SOURCE_DERIVED_UNDOCUMENTED")
    out["elevation__quality"] = elev["elevation__quality"]
    date, precision, invalid = K.parse_dates(raw.eventDate, 1800, "2026-10-01")
    dates = K.date_group(date, precision, "dataset:fungitastic", "OBSERVER_RECORDED", invalid)
    out["event_date__quality"] = dates["event_date__quality"]
    out["year"] = pd.to_numeric(date.str[:4], errors="coerce").astype("Int64")
    out["detailed_substrate_missing"] = raw.substrate.isna()
    return out


def join_climate(context, average):
    climate = pd.read_csv(average, usecols=["observationID", *BIO])
    climate["observation_uid"] = "fungitastic:" + climate.observationID.astype(str)
    climate = climate[climate.observation_uid.isin(context.observation_uid)]
    if climate.observation_uid.duplicated().any():
        raise ValueError("duplicate climate observation IDs")
    result = context.merge(climate.drop(columns="observationID"), on="observation_uid", how="left",
                           sort=False, validate="one_to_one")
    result[BIO] = result[BIO].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    result["climate__source"] = "CHELSA:FungiTastic"
    result["climate__method"] = "BIOCLIMATIC_NORMALS_1981_2010_RAW_SCALE"
    result["climate__quality"] = np.where(result[BIO].notna().all(axis=1), "SOURCE_DERIVED", "MISSING_OR_PARTIAL")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", default="data/processed/s8_s9_context_v002")
    args = ap.parse_args()
    output = REPO_ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    reports = REPO_ROOT / "reports/s9/audit"
    reports.mkdir(parents=True, exist_ok=True)
    splits_path = REPO_ROOT / "data/versions/v002/splits.csv"
    dataset_path = REPO_ROOT / "data/versions/v002/manifest.json"
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    expected = next(e["sha256"] for e in dataset["outputs"] if e["name"] == "splits.csv")
    if sha256_file(splits_path) != expected:
        raise ValueError("frozen splits hash mismatch")
    splits = pd.read_csv(splits_path)
    splits = splits[splits.split.isin(["train", "val"])].copy()
    metadata = REPO_ROOT / "data/raw/fungitastic/metadata.zip"
    expected = next(e["sha256"] for e in dataset["inputs"] if e["path"].endswith("metadata.zip"))
    if sha256_file(metadata) != expected:
        raise ValueError("metadata.zip differs from v002 source")
    average = REPO_ROOT / "data/raw/fungitastic/FungiTastic-Climatic-Average.csv"
    # Pinned bytes extracted from the CRC-verified official climaticData.zip.
    if sha256_file(average) != "d5f9217d65d71bbadc362d41f8bd5f0e18c2e8b5d06840af7e49b9412ad53cd7":
        raise ValueError("climate average differs from the acquired official archive")
    print("Reading original train/val metadata...", flush=True)
    raw, images, consistency = read_cohort(metadata, splits)
    consistency.to_csv(reports / "within_observation_consistency.csv", index=False, lineterminator="\n")
    context = join_climate(clean_context(raw), average)
    audit = audit_metadata(raw)
    context = context.merge(audit[["observation_uid", "substrate_class", "substrate__quality"]],
                            on="observation_uid", how="left", validate="one_to_one", sort=False)
    context.to_csv(output / "observations.csv", index=False, lineterminator="\n")
    # Detailed rows stay local: exact coordinates and raw observer metadata are not published.
    audit.to_csv(output / "metadata_audit.csv", index=False, lineterminator="\n")
    images.to_csv(output / "raw_images.csv", index=False, lineterminator="\n")
    for group in ["substrate", "habitat", "elevation", "event_date"]:
        counts = audit.groupby(["split", f"{group}__quality"], dropna=False).size().rename("observations").reset_index()
        counts.to_csv(reports / f"{group}_quality.csv", index=False, lineterminator="\n")
    for dimension in ["year", "genus", "metaSubstrate"]:
        (audit.groupby(["split", dimension, "substrate__quality"], dropna=False).size().rename("observations")
         .reset_index().to_csv(reports / f"substrate_by_{dimension}.csv", index=False, lineterminator="\n"))
    rows = []
    for split, group in context.groupby("split", sort=True):
        rows.append({"split": split, "observations": len(group), "geo_ok": int(group["geo__quality"].eq("OK").sum()),
                     "climate_complete": int(group[BIO].notna().all(axis=1).sum()),
                     "climate_partial": int((group[BIO].notna().any(axis=1) & group[BIO].isna().any(axis=1)).sum()),
                     "climate_absent": int(group[BIO].isna().all(axis=1).sum())})
    pd.DataFrame(rows).to_csv(reports / "context_coverage.csv", index=False, lineterminator="\n")
    cells = context[["latitude", "longitude"]].round(1).astype(str).agg(":".join, axis=1)
    known = set(cells[context.split.eq("train") & context["geo__quality"].eq("OK")])
    overlap = context[context.split.eq("val")].assign(seen_train_cell=cells[context.split.eq("val")].isin(known))
    (overlap.groupby("seen_train_cell").size().rename("val_observations").reset_index()
     .to_csv(reports / "geo_cell_overlap.csv", index=False, lineterminator="\n"))
    temporal = audit.groupby("split").year.agg(["min", "max"]).reset_index()
    temporal["observations_before_2011"] = temporal["split"].map(
        audit[audit.year.lt(2011)].groupby("split").size()).fillna(0).astype(int)
    temporal.to_csv(reports / "climate_temporal_audit.csv", index=False, lineterminator="\n")
    def entry(path):
        return {"path": rel(path), "sha256": sha256_file(path)}
    inputs = [metadata, average, splits_path, dataset_path, Path(__file__),
              Path(__file__).with_name("cleaning.py"), Path(__file__).with_name("curation_sources.py")]
    manifest = {"created_utc": utc_now(), "dataset_fingerprint": dataset["content_fingerprint"],
                "splits": ["train", "val"], "climate_product": "CHELSA normals 1981-2010, not ERA5-Land",
                "climate_units": "original stored scale; standardized on train; no physical-unit conversion asserted",
                "temporal_limitation": "Retrospective normals; may include future climate relative to observations before 2011.",
                "inputs": [entry(p) for p in inputs],
                "outputs": [entry(output / p) for p in ("observations.csv", "metadata_audit.csv", "raw_images.csv")]}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (reports / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False))
    print(audit.groupby(["split", "substrate__quality"]).size().to_string())


if __name__ == "__main__":
    main()
