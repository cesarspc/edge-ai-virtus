"""Join raw FungiTastic BIOCLIM averages to official v002 observations without fitting transforms."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data/raw/fungitastic/FungiTastic"
OBS_PATH = ROOT / "data/processed/v002/observations.csv"
DATASET_MANIFEST = ROOT / "data/versions/v002/manifest.json"
CLIMATE_MANIFEST = DATA_DIR / "climatic_download_manifest.json"
OUTPUT = ROOT / "data/interim/s9_climate_v002.csv"
OUTPUT_MANIFEST = ROOT / "data/interim/s9_climate_v002_manifest.json"
FEATURES = [f"Bio{i}" for i in range(1, 20)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def main() -> int:
    for path in (OBS_PATH, DATASET_MANIFEST, CLIMATE_MANIFEST,
                 DATA_DIR / "FungiTastic-Climatic-Average.csv"):
        if not path.is_file():
            raise SystemExit(f"Falta el insumo: {path}")

    dataset_manifest = json.loads(DATASET_MANIFEST.read_text(encoding="utf-8"))
    expected_obs_hash = next(o["sha256"] for o in dataset_manifest["outputs"]
                             if o["name"] == "observations.csv")
    if sha256(OBS_PATH) != expected_obs_hash:
        raise SystemExit("observations.csv no coincide con el manifiesto oficial de v002")
    climate_manifest = json.loads(CLIMATE_MANIFEST.read_text(encoding="utf-8-sig"))
    climate_archive = Path(climate_manifest["archive"])
    if sha256(climate_archive) != climate_manifest["sha256"]:
        raise SystemExit("El paquete climático no coincide con su SHA-256 registrado")

    progress = tqdm(total=4, desc="Preparando clima v002", unit="etapa")
    observations = pd.read_csv(OBS_PATH, usecols=["observation_uid", "source_id", "source_observation_id", "split"],
                              dtype={"source_observation_id": "string"}, low_memory=False)
    observations = observations[(observations.source_id == "fungitastic") &
                                observations.split.isin(["train", "val", "test"])].copy()
    observations["source_observation_id"] = observations.source_observation_id.str.strip()
    if observations.source_observation_id.isna().any() or observations.source_observation_id.duplicated().any():
        raise SystemExit("source_observation_id de v002 contiene valores vacíos o repetidos")
    progress.update(1)

    source_path = DATA_DIR / "FungiTastic-Climatic-Average.csv"
    climate = pd.read_csv(source_path, usecols=["observationID", *FEATURES],
                          dtype={"observationID": "string"}, low_memory=False)
    climate = climate.rename(columns={"observationID": "source_observation_id"})
    climate.source_observation_id = climate.source_observation_id.str.strip()
    if climate.source_observation_id.isna().any() or climate.source_observation_id.duplicated().any():
        raise SystemExit("La tabla climática promedio contiene IDs vacíos o duplicados")
    climate[FEATURES] = climate[FEATURES].apply(pd.to_numeric, errors="coerce")
    if np.isinf(climate[FEATURES].to_numpy(dtype=float)).any():
        raise SystemExit("La tabla climática contiene infinitos")
    progress.update(1)

    aligned = observations.merge(climate, on="source_observation_id", how="left", validate="one_to_one",
                                sort=False)
    aligned["climate_available"] = aligned[FEATURES].notna().all(axis=1)
    aligned = aligned[["observation_uid", "source_observation_id", "split", "climate_available", *FEATURES]]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temp = OUTPUT.with_suffix(".csv.tmp")
    aligned.to_csv(temp, index=False, lineterminator="\n")
    os.replace(temp, OUTPUT)
    progress.update(1)

    missing_by_split = {str(k): int(v) for k, v in
                        aligned.loc[~aligned.climate_available, "split"].value_counts().to_dict().items()}
    report = {
        "output": str(OUTPUT),
        "output_sha256": sha256(OUTPUT),
        "rows": len(aligned),
        "rows_by_split": {str(k): int(v) for k, v in aligned.split.value_counts().to_dict().items()},
        "climate_available": int(aligned.climate_available.sum()),
        "climate_missing": int((~aligned.climate_available).sum()),
        "missing_by_split": missing_by_split,
        "features": FEATURES,
        "feature_policy": "Raw BIOCLIM averages; scaling/imputation must be fitted on train only.",
        "source_archive_sha256": climate_manifest["sha256"],
        "dataset_version": "v002",
        "dataset_fingerprint": dataset_manifest["content_fingerprint"],
        "observations_sha256": expected_obs_hash,
    }
    atomic_json(OUTPUT_MANIFEST, report)
    progress.update(1)
    progress.close()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Manifiesto de la tabla: {OUTPUT_MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
