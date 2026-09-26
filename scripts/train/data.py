"""Dataset tables for training (no PyTorch dependency, so it can be checked without a training environment).

Joins the processed dataset version (data/processed/<version>/images.csv) with the image download manifest and
returns one frame per split. It also verifies that the processed tables are the published version (SHA-256 against
data/versions/<version>/manifest.json), so a run can never silently use a different dataset.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SPLITS = ("train", "val", "test")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


@dataclass
class DatasetTables:
    version: str
    fingerprint: str
    classes: list[str]
    frames: dict[str, pd.DataFrame]      # split -> rows (image_uid, observation_uid, label_id, genus, path)
    missing: pd.DataFrame                # rows whose image is not available locally
    image_root: Path


def load_tables(cfg: dict[str, Any]) -> DatasetTables:
    d = cfg["data"]
    version = d["version"]
    proc = REPO_ROOT / "data" / "processed" / version
    ver = REPO_ROOT / "data" / "versions" / version
    manifest = json.loads((ver / "manifest.json").read_text(encoding="utf-8"))
    for name in ("images.csv", "observations.csv"):
        path = proc / name
        if not path.exists():
            raise SystemExit(f"{path} not found: run `python scripts/data/build_dataset.py` (dataset {version})")
        expected = next(o["sha256"] for o in manifest["outputs"] if o["name"] == name)
        if sha256_file(path) != expected:
            raise SystemExit(f"{path} does not match data/versions/{version}/manifest.json; "
                             "rebuild it with build_dataset.py (and check with --check)")
    classes = pd.read_csv(ver / "classes.csv").sort_values("label_id")["genus"].tolist()
    img = pd.read_csv(proc / "images.csv", low_memory=False)
    obs = pd.read_csv(proc / "observations.csv", low_memory=False, usecols=["observation_uid", "split", "split_temporal"])
    img = img[img["source_id"] == d["source"]].copy()
    split_col = d.get("split_column", "split")
    if split_col != "split":   # alternative partition (observation level), e.g. split_temporal
        img = img.drop(columns="split").merge(obs[["observation_uid", split_col]].rename(columns={split_col: "split"}),
                                              on="observation_uid", how="left")
    img = img[img["split"].isin(SPLITS)]

    root = resolve(d["image_root"])
    man_path = resolve(d["download_manifest"])
    if man_path.exists():
        man = pd.read_csv(man_path, dtype=str, keep_default_na=False)[["image_uid", "filename", "status"]]
        img = img.merge(man, on="image_uid", how="left")
    else:
        img["filename"], img["status"] = img["source_image_id"], ""
    img["path"] = img["filename"].fillna(img["source_image_id"]).map(lambda f: str(root / "300p" / f))
    available = img["status"].isin(["ok", "too_small"]) & img["path"].map(lambda p: Path(p).exists())
    missing = img[~available]
    img = img[available]
    cols = ["image_uid", "observation_uid", "label_id", "genus", "path"]
    frames = {s: img[img["split"] == s][cols].sort_values("image_uid").reset_index(drop=True) for s in SPLITS}
    return DatasetTables(version, manifest["content_fingerprint"], classes, frames,
                         missing[["image_uid", "split", "status"]], root)


def summary(t: DatasetTables) -> pd.DataFrame:
    rows = []
    for s, f in t.frames.items():
        rows.append({"split": s, "images_available": len(f), "observations": f["observation_uid"].nunique(),
                     "images_missing": int((t.missing["split"] == s).sum())})
    return pd.DataFrame(rows)
