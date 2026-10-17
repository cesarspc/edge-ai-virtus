"""S6: genus -> family -> order -> class -> phylum mapping (and species per observation) for a dataset version.

Reads only the raw FungiTastic metadata already registered in ``data/manifests/sources/fungitastic.yaml`` and the
published partition table of the version. It never modifies ``data/versions/<version>/`` (immutable); outputs go to
``data/taxonomy/<version>/``:

    genus_taxonomy.csv       one row per class (label_id, genus, family, order, class, phylum, counts per split)
    observation_species.csv  observation_uid, split, genus, species (for error analysis below genus)
    manifest.json            input/output SHA-256, rules and checks

The build fails (instead of guessing) if a genus maps to more than one higher taxon, an observation has more than
one species, or any observation of the version is missing from the raw files.

Usage:
    python scripts/data/build_taxonomy.py               # version from configs/dataset_v002.yaml
    python scripts/data/build_taxonomy.py --check       # rebuild in memory and compare with the published files
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
RANKS = ["genus", "family", "order", "class", "phylum"]
RAW_FILES = {"Train": "FungiTastic/FungiTastic-Train.csv", "ClosedSet-Val": "FungiTastic/FungiTastic-ClosedSet-Val.csv",
             "ClosedSet-Test": "FungiTastic/FungiTastic-ClosedSet-Test.csv"}
RAW_COLUMNS = ["observationID", "genus", "species", "family", "order", "class", "phylum"]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(path: Path) -> str:
    """SHA-256 of a versioned text file with LF line endings (Git ``core.autocrlf`` may check it out as CRLF)."""
    return sha256_bytes(path.read_bytes().replace(b"\r\n", b"\n"))


def csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def published_sha(manifest: dict, name: str) -> str:
    return next(o["sha256"] for o in manifest["outputs"] if o["name"] == name)


def build(dataset_cfg_path: Path) -> tuple[dict[str, bytes], dict]:
    cfg = yaml.safe_load(dataset_cfg_path.read_text(encoding="utf-8"))
    version = cfg["version"]
    ver = REPO_ROOT / "data" / "versions" / version
    manifest = json.loads((ver / "manifest.json").read_text(encoding="utf-8"))
    for name in ("splits.csv", "classes.csv"):
        if sha256_text(ver / name) != published_sha(manifest, name):
            raise SystemExit(f"{ver / name} differs from the published {version} manifest")
    splits = pd.read_csv(ver / "splits.csv", usecols=["observation_uid", "genus", "label_id", "split"])
    classes = pd.read_csv(ver / "classes.csv").sort_values("label_id")

    source = yaml.safe_load((REPO_ROOT / "data/manifests/sources/fungitastic.yaml").read_text(encoding="utf-8"))
    entry = next(f for f in source["local_files"] if f["path"].endswith("fungitastic/metadata.zip"))
    zip_path = REPO_ROOT / entry["path"]
    if not zip_path.is_file():
        raise SystemExit(f"{zip_path} not found: run `python scripts/data/acquire_datasets.py fungitastic`")
    zip_sha = sha256_file(zip_path)
    if zip_sha != entry["sha256"]:
        raise SystemExit(f"{zip_path} does not match the SHA-256 registered in data/manifests/sources/fungitastic.yaml")
    files = cfg["sources"]["fungitastic"]["files"]
    frames = []
    with zipfile.ZipFile(zip_path) as z:
        for name in files:
            with z.open(RAW_FILES[name]) as fh:
                frames.append(pd.read_csv(fh, usecols=RAW_COLUMNS, low_memory=False).assign(raw_file=name))
    raw = pd.concat(frames, ignore_index=True)
    raw["observation_uid"] = "fungitastic:" + raw["observationID"].astype(str)
    raw = raw[raw["observation_uid"].isin(splits["observation_uid"])].rename(columns={"genus": "genus_raw"})

    # Observation level: one species and one set of higher ranks per observation (all images agree).
    per_obs = raw.groupby("observation_uid")[["species", "family", "order", "class", "phylum", "genus_raw"]].nunique(dropna=False)
    conflicts = per_obs[(per_obs > 1).any(axis=1)]
    if len(conflicts):
        raise SystemExit(f"{len(conflicts)} observations with conflicting taxonomy across images, e.g. {conflicts.index[:3].tolist()}")
    obs = raw.drop_duplicates("observation_uid").drop(columns=["observationID"])
    obs = splits.merge(obs, on="observation_uid", how="left", validate="one_to_one")
    if obs["species"].isna().any() or obs[["family", "order", "class", "phylum"]].isna().any().any():
        missing = int(obs[["species", "family", "order", "class", "phylum"]].isna().any(axis=1).sum())
        raise SystemExit(f"{missing} observations of {version} without complete taxonomy in the raw files")

    # Genus level: the label (already synonym-normalized in the version) must have a single higher lineage.
    lineage = obs.groupby("genus")[["family", "order", "class", "phylum"]].nunique()
    ambiguous = lineage[(lineage > 1).any(axis=1)]
    if len(ambiguous):
        raise SystemExit(f"genera mapped to several higher taxa: {ambiguous.to_dict('index')}")
    counts = obs.pivot_table(index="genus", columns="split", values="observation_uid", aggfunc="count", fill_value=0)
    taxonomy = (classes.merge(obs.drop_duplicates("genus")[["genus", "family", "order", "class", "phylum"]], on="genus", how="left")
                .merge(obs.groupby("genus")["species"].nunique().rename("n_species").reset_index(), on="genus"))
    for split in ("train", "val", "test"):
        taxonomy[f"n_obs_{split}"] = taxonomy["genus"].map(counts[split]).astype(int)
    if taxonomy[["family", "order", "class", "phylum"]].isna().any().any():
        raise SystemExit("a class of the version has no taxonomy")
    synonyms = obs.loc[obs["genus_raw"] != obs["genus"], ["genus_raw", "genus"]].value_counts().reset_index()

    species = obs[["observation_uid", "split", "genus", "species"]].sort_values("observation_uid").reset_index(drop=True)
    outputs = {"genus_taxonomy.csv": csv_bytes(taxonomy[["label_id", "genus", "family", "order", "class", "phylum",
                                                         "n_species", "n_obs_train", "n_obs_val", "n_obs_test"]]),
               "observation_species.csv": csv_bytes(species)}
    meta = {
        "version": version, "fingerprint": manifest["content_fingerprint"],
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "builder": {"command": "python scripts/data/build_taxonomy.py", "sha256": sha256_file(Path(__file__)),
                    "python": sys.version.split()[0], "pandas": pd.__version__},
        "inputs": [{"path": entry["path"], "sha256": zip_sha, "members": [RAW_FILES[f] for f in files],
                    "columns": RAW_COLUMNS},
                   {"path": f"data/versions/{version}/splits.csv", "sha256": published_sha(manifest, "splits.csv")},
                   {"path": f"data/versions/{version}/classes.csv", "sha256": published_sha(manifest, "classes.csv")}],
        "rules": ["Higher ranks (family, order, class, phylum) are taken as published by FungiTastic; no manual edits.",
                  "The label genus is the synonym-normalized genus of the version; raw genus differences are listed.",
                  "An observation with images that disagree on species or higher ranks would abort the build."],
        "checks": {"observations": int(len(obs)), "observations_without_taxonomy": 0,
                   "observations_with_conflicting_taxonomy": 0, "ambiguous_genera": 0,
                   "raw_genus_synonyms": synonyms.to_dict("records"),
                   "distinct": {r: int(obs[r].nunique()) for r in ["species", *RANKS]}},
        "outputs": [{"name": n, "path": f"data/taxonomy/{version}/{n}", "sha256": sha256_bytes(b)} for n, b in outputs.items()],
    }
    return outputs, meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/dataset_v002.yaml")
    ap.add_argument("--check", action="store_true", help="rebuild in memory and compare with the published outputs")
    args = ap.parse_args()
    outputs, meta = build(REPO_ROOT / args.config)
    target = REPO_ROOT / "data" / "taxonomy" / meta["version"]
    if args.check:
        published = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
        bad = [o["name"] for o in meta["outputs"]
               if sha256_text(target / o["name"]) != o["sha256"]
               or next(p["sha256"] for p in published["outputs"] if p["name"] == o["name"]) != o["sha256"]]
        print("OK: taxonomy is reproducible" if not bad else f"MISMATCH: {bad}")
        return 1 if bad else 0
    target.mkdir(parents=True, exist_ok=True)
    for name, payload in outputs.items():
        (target / name).write_bytes(payload)
    (target / "manifest.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(pd.read_csv(io.BytesIO(outputs["genus_taxonomy.csv"])).to_string(index=False))
    print(json.dumps(meta["checks"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
