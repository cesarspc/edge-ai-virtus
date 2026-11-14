"""Inspect climate CSV schemas and persist an incremental JSON report."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data/raw/fungitastic/FungiTastic"
REPORT = DATA_DIR / "climatic_inspection.json"
MANIFEST = DATA_DIR / "climatic_download_manifest.json"


def save_checkpoint(report: dict) -> None:
    temporary = REPORT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, REPORT)


def main() -> int:
    if not MANIFEST.is_file():
        raise SystemExit(f"No se encontró el manifiesto de descarga: {MANIFEST}")
    files = sorted(DATA_DIR.glob("FungiTastic-Climatic-*.csv"))
    if not files:
        raise SystemExit(f"No se encontraron CSV climáticos en {DATA_DIR}")
    report = {
        "archive_manifest": json.loads(MANIFEST.read_text(encoding="utf-8-sig")),
        "inspection": "schemas_and_sample_rows_only",
        "files": [],
    }
    save_checkpoint(report)
    for path in tqdm(files, desc="Inspeccionando archivos climáticos", unit="archivo"):
        sample = pd.read_csv(path, nrows=3, low_memory=False)
        entry = {
            "name": path.name,
            "bytes": path.stat().st_size,
            "columns": sample.columns.tolist(),
            "sample_rows": sample.astype(object).where(sample.notna(), None).to_dict(orient="records"),
        }
        report["files"].append(entry)
        save_checkpoint(report)
    print(f"Informe guardado: {REPORT}")
    for entry in report["files"]:
        print(f"{entry['name']} ({entry['bytes']:,} bytes): {entry['columns']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
