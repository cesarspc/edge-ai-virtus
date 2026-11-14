"""Verify the climatic archive hash, table keys, and coverage of FungiTastic v002."""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data/raw/fungitastic/FungiTastic"
PROCESSED = ROOT / "data/processed/v002/observations.csv"
MANIFEST = DATA_DIR / "climatic_download_manifest.json"
INSPECTION = DATA_DIR / "climatic_inspection.json"
REPORT = DATA_DIR / "climatic_coverage_v002.json"


def atomic_json(path: Path, value: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_ids(path: Path, target_ids: set[str], split_by_id: dict[str, str], desc: str) -> dict:
    found: set[str] = set()
    duplicates: set[str] = set()
    chunks = pd.read_csv(path, usecols=["observationID"], dtype={"observationID": "string"},
                         chunksize=50_000, low_memory=False)
    for chunk in tqdm(chunks, desc=desc, unit="bloque"):
        ids = chunk["observationID"].dropna().str.strip()
        repeated = ids[ids.isin(found) | ids.duplicated()]
        duplicates.update(repeated.tolist())
        found.update(ids.tolist())
    matched = target_ids & found
    missing_ids = sorted(target_ids - found)
    split_counts = {str(k): int(v) for k, v in
                    pd.Series([split_by_id[i] for i in missing_ids]).value_counts().to_dict().items()}
    return {
        "rows_unique_ids": len(found),
        "duplicate_ids": len(duplicates),
        "v002_observations_matched": len(matched),
        "v002_observations_missing": len(missing_ids),
        "coverage_pct": round(100 * len(matched) / len(target_ids), 4) if target_ids else None,
        "missing_ids": missing_ids,
        "missing_by_split": split_counts,
    }


def inspect_numeric_values(path: Path, columns: list[str]) -> dict:
    nulls = {column: 0 for column in columns}
    minima = {column: float("inf") for column in columns}
    maxima = {column: float("-inf") for column in columns}
    rows = 0
    non_finite = 0
    chunks = pd.read_csv(path, usecols=columns, chunksize=50_000, low_memory=False)
    for chunk in tqdm(chunks, desc="Revisando valores BioClim", unit="bloque"):
        values = chunk.apply(pd.to_numeric, errors="coerce")
        rows += len(values)
        non_finite += int(np.isinf(values.to_numpy(dtype=float)).sum())
        values = values.where(np.isfinite(values))
        for column in columns:
            series = values[column]
            nulls[column] += int(series.isna().sum())
            if series.notna().any():
                minima[column] = min(minima[column], float(series.min()))
                maxima[column] = max(maxima[column], float(series.max()))
    return {
        "rows": rows,
        "missing_values_total": sum(nulls.values()),
        "infinite_values_total": non_finite,
        "missing_by_feature": {k: v for k, v in nulls.items() if v},
        "min_by_feature": {k: v for k, v in minima.items() if v != float("inf")},
        "max_by_feature": {k: v for k, v in maxima.items() if v != float("-inf")},
    }


def main() -> int:
    if not all(path.is_file() for path in (MANIFEST, INSPECTION, PROCESSED)):
        raise SystemExit("Faltan el manifiesto, el reporte de inspección o observations.csv de v002.")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8-sig"))
    archive = Path(manifest["archive"])
    actual_hash = sha256(archive)
    if actual_hash != manifest["sha256"]:
        raise SystemExit(f"SHA-256 incorrecto: {actual_hash} != {manifest['sha256']}")

    observations = pd.read_csv(PROCESSED, usecols=["source_id", "source_observation_id", "split"],
                               dtype={"source_observation_id": "string"}, low_memory=False)
    observations = observations[(observations.source_id == "fungitastic") &
                                observations.split.isin(["train", "val", "test"])]
    target_ids = set(observations.source_observation_id.dropna().str.strip())
    if len(target_ids) != len(observations):
        raise SystemExit("Hay source_observation_id repetidos o vacíos en FungiTastic v002.")

    inspected = json.loads(INSPECTION.read_text(encoding="utf-8"))
    split_by_id = observations.set_index("source_observation_id")["split"].to_dict()
    report = {
        "archive": {"path": str(archive), "bytes": archive.stat().st_size,
                    "sha256": actual_hash, "hash_matches_manifest": True},
        "v002_source_observations": len(target_ids),
        "files": {},
    }
    atomic_json(REPORT, report)
    for entry in inspected["files"]:
        path = DATA_DIR / entry["name"]
        result = inspect_ids(path, target_ids, split_by_id, f"IDs {path.name}")
        if "Timeseries" in path.name:
            columns = entry["columns"]
            years = sorted({int(m.group(1)) for c in columns if (m := re.search(r"_(\d{4})$", c))})
            result["time_series_years"] = [min(years), max(years)] if years else []
            result["time_series_feature_columns"] = len(columns) - 2
        else:
            features = [c for c in entry["columns"] if re.fullmatch(r"Bio\d+", c)]
            result["bioclim_feature_columns"] = len(features)
            result["numeric_values"] = inspect_numeric_values(path, features)
        report["files"][entry["name"]] = result
        atomic_json(REPORT, report)

    report["validation_criteria"] = {
        "archive_sha256_matches_manifest": True,
        "no_duplicate_climate_observation_ids": True,
        "minimum_coverage_pct_per_climate_table": 99.0,
        "missing_rows_are_listed_for_explicit_imputation": True,
        "no_infinite_values_in_bioclim_features": True,
    }
    report["validation_passed"] = all(
        result["duplicate_ids"] == 0 and result["coverage_pct"] >= 99.0
        for result in report["files"].values()
    ) and report["files"]["FungiTastic-Climatic-Average.csv"]["numeric_values"]["infinite_values_total"] == 0
    atomic_json(REPORT, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Reporte de cobertura: {REPORT}")
    return 0 if report["validation_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
