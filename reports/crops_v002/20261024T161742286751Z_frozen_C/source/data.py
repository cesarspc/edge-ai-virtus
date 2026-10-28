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
from progress_ui import bar, stage

REPO_ROOT = Path(__file__).resolve().parents[2]
SPLITS = ("train", "val", "test")


def sha256_file(path: Path, show_progress: bool = False) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        if not show_progress:
            while chunk := fh.read(1 << 20):
                h.update(chunk)
        else:
            with bar(total=path.stat().st_size, desc=f"SHA-256 {path.name}",
                     unit="B", unit_scale=True, disable=False) as progress:
                while chunk := fh.read(1 << 20):
                    h.update(chunk)
                    progress.update(len(chunk))
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


def load_tables(cfg: dict[str, Any], show_progress: bool = False,
                splits: tuple[str, ...] = SPLITS) -> DatasetTables:
    if not splits or not set(splits).issubset(SPLITS) or len(set(splits)) != len(splits):
        raise ValueError("splits must be unique train/val/test values")
    d = cfg["data"]
    version = d["version"]
    proc = REPO_ROOT / "data" / "processed" / version
    ver = REPO_ROOT / "data" / "versions" / version
    if not (ver / "manifest.json").is_file():
        raise SystemExit(f"dataset {version} is not published yet; finish acquisition, image download and build_dataset.py")
    manifest = json.loads((ver / "manifest.json").read_text(encoding="utf-8"))
    if show_progress:
        print(f"Dataset {version}: comprobando manifiestos y tablas...",flush=True)
    for name in ("images.csv", "observations.csv"):
        path = proc / name
        if not path.exists():
            raise SystemExit(f"{path} not found: run `python scripts/data/build_dataset.py` (dataset {version})")
        expected = next(o["sha256"] for o in manifest["outputs"] if o["name"] == name)
        if sha256_file(path, show_progress=show_progress) != expected:
            raise SystemExit(f"{path} does not match data/versions/{version}/manifest.json; "
                             "rebuild it with build_dataset.py (and check with --check)")
    for name in ("classes.csv", "image_integrity.csv"):
        entry = next((o for o in manifest["outputs"] if o["name"] == name), None)
        if entry and (not (ver / name).is_file() or sha256_file(ver / name, show_progress=show_progress) != entry["sha256"]):
            raise SystemExit(f"{ver / name} does not match the published dataset")
    with stage("Leyendo clases e imagenes del dataset", active=show_progress):
        classes = pd.read_csv(ver / "classes.csv").sort_values("label_id")["genus"].tolist()
        img = pd.read_csv(proc / "images.csv", low_memory=False)
    with stage("Leyendo observaciones del dataset", active=show_progress):
        obs = pd.read_csv(proc / "observations.csv", low_memory=False, usecols=["observation_uid", "split", "split_temporal"])
    img = img[img["source_id"] == d["source"]].copy()
    split_col = d.get("split_column", "split")
    if split_col != "split":   # alternative partition (observation level), e.g. split_temporal
        img = img.drop(columns="split").merge(obs[["observation_uid", split_col]].rename(columns={split_col: "split"}),
                                              on="observation_uid", how="left")
    # Exclude inactive partitions before opening/hash-checking any photographs.
    img = img[img["split"].isin(splits)]

    root = resolve(d["image_root"])
    man_path = resolve(d["download_manifest"])
    if man_path.exists():
        with stage("Leyendo manifiesto de imagenes descargadas", active=show_progress):
            man = pd.read_csv(man_path, dtype=str, keep_default_na=False)[["image_uid", "filename", "status"]]
        if man["image_uid"].duplicated().any():
            raise SystemExit("download manifest contains duplicated image_uid values")
        img = img.merge(man, on="image_uid", how="left", validate="one_to_one")
    else:
        img["filename"], img["status"] = img["source_image_id"], ""
    img["path"] = img["filename"].fillna(img["source_image_id"]).map(lambda f: str(root / "300p" / f))
    img["status"] = img["status"].fillna("")
    def visible_rows(rows, description):
        if show_progress:
            return bar(rows, desc=description, unit="imagen")
        return rows

    if show_progress:
        print(f"Verificando disponibilidad de {len(img):,} imagenes locales...", flush=True)
    exists = pd.Series([Path(p).is_file() for p in visible_rows(img["path"], "Archivos locales")], index=img.index)
    available = img["status"].isin(["ok", "too_small"]) & exists
    integrity_path = ver / "image_integrity.csv"
    if any(o["name"] == "image_integrity.csv" for o in manifest["outputs"]):
        integrity = pd.read_csv(integrity_path, dtype=str).set_index("image_uid")["sha256"]
        expected = img["image_uid"].map(integrity)
        if show_progress:
            print("Verificando SHA-256 de las imagenes; puede tardar varios minutos...", flush=True)
        paths = img.loc[available, "path"]
        actual = pd.Series([sha256_file(Path(p)) for p in visible_rows(paths, "Integridad SHA-256")], index=paths.index, dtype=str)
        bad = available & ~expected.eq(actual.reindex(img.index))
        img.loc[bad, "status"] = "hash_mismatch"
        available &= ~bad
    missing = img[~available]
    img = img[available]
    cols = ["image_uid", "observation_uid", "label_id", "genus", "path"]
    frames = {s: img[img["split"] == s][cols].sort_values("image_uid").reset_index(drop=True) for s in splits}
    return DatasetTables(version, manifest["content_fingerprint"], classes, frames,
                         missing[["image_uid", "split", "status"]], root)


def summary(t: DatasetTables) -> pd.DataFrame:
    rows = []
    for s, f in t.frames.items():
        rows.append({"split": s, "images_available": len(f), "observations": f["observation_uid"].nunique(),
                     "images_missing": int((t.missing["split"] == s).sum())})
    return pd.DataFrame(rows)
