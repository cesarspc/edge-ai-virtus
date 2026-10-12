"""Validate downloaded bytes and remove exact cross-partition copies, protecting test."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd

from common import REPO_ROOT, sha256_file

EXCLUSION_COLUMNS = ["image_uid", "observation_uid", "split", "genus", "sha256", "protected_split", "reason"]
NEAR_COLUMNS = ["image_uid_a", "image_uid_b", "split_a", "split_b", "genus_a", "genus_b", "hamming"]


def near_pairs(images: pd.DataFrame, radius: int) -> pd.DataFrame:
    """Exact Hamming search with radius+1 disjoint blocks (pigeonhole principle).

    Report only cross-partition pairs with different byte hashes; never exclude by pHash.
    """
    if not 0 <= radius < 64:
        raise ValueError("near_duplicate_hamming must be in [0, 63]")
    blocks = radius + 1
    widths = [64 // blocks + (i < 64 % blocks) for i in range(blocks)]
    buckets = [defaultdict(list) for _ in widths]
    previous, pairs = [], []
    for row in images.sort_values("image_uid").to_dict("records"):
        value = int(row["phash64"], 16)
        keys, offset = [], 0
        for width in widths:
            keys.append((value >> offset) & ((1 << width) - 1))
            offset += width
        candidates = set()
        for bucket, key in zip(buckets, keys):
            candidates.update(bucket[key])
        for i in sorted(candidates):
            other, other_hash = previous[i]
            if row["split"] == other["split"] or row["sha256"] == other["sha256"]:
                continue
            distance = (value ^ other_hash).bit_count()
            if distance <= radius:
                pairs.append({"image_uid_a": other["image_uid"], "image_uid_b": row["image_uid"],
                              "split_a": other["split"], "split_b": row["split"],
                              "genus_a": other["genus"], "genus_b": row["genus"], "hamming": distance})
        for bucket, key in zip(buckets, keys):
            bucket[key].append(len(previous))
        previous.append((row, value))
    return pd.DataFrame(pairs, columns=NEAR_COLUMNS)


def curate(obs: pd.DataFrame, img: pd.DataFrame, policy: dict):
    if policy["cross_split_exact_duplicates"] != "protect_test":
        raise ValueError("unsupported exact duplicate policy")
    path = REPO_ROOT / policy["download_manifest"]
    if not path.exists():
        raise SystemExit("Download images after build_dataset.py --prepare-images, then build again.")
    manifest = pd.read_csv(path, dtype=str, keep_default_na=False)
    if manifest["image_uid"].duplicated().any():
        raise SystemExit("download manifest has duplicated image_uid values")
    joined = img.merge(manifest[["image_uid", "filename", "status", "sha256", "phash64", "width", "height"]],
                       on="image_uid", how="left", validate="one_to_one")
    report_small = policy.get("small_images") == "report"
    valid = joined["status"].isin(["ok", "too_small"] if report_small else ["ok"])
    valid &= joined["sha256"].fillna("").str.fullmatch(r"[0-9a-f]{64}")
    valid &= joined["phash64"].fillna("").str.fullmatch(r"[0-9a-f]{16}")
    valid &= joined["filename"].eq(joined["source_image_id"])
    for side in ("width", "height"):
        valid &= pd.to_numeric(joined[side], errors="coerce").ge(1 if report_small else policy["min_side_px"])
    if not valid.all():
        raise SystemExit(f"{int((~valid).sum())} images missing, invalid or below minimum size; retry download before publishing")
    root = path.parent / str(policy["fungitastic_resolution"])
    for row in joined.itertuples(index=False):
        filename = Path(row.filename)
        if filename.name != row.filename or not (root / filename).is_file():
            raise SystemExit(f"missing/invalid image path for {row.image_uid}")
        if sha256_file(root / filename) != row.sha256:
            raise SystemExit(f"image bytes differ from manifest: {row.image_uid}; repair download before publishing")
    priorities = {"train": 0, "val": 1, "test": 2}
    rank = joined["split"].map(priorities)
    protected_rank = rank.groupby(joined["sha256"]).transform("max")
    remove = rank < protected_rank
    exclusions = joined.loc[remove, ["image_uid", "observation_uid", "split", "genus", "sha256"]].copy()
    exclusions["protected_split"] = protected_rank[remove].map({v: k for k, v in priorities.items()})
    exclusions["reason"] = "exact_cross_split_duplicate"
    exclusions = exclusions.reindex(columns=EXCLUSION_COLUMNS).sort_values("image_uid").reset_index(drop=True)
    near = near_pairs(joined.loc[~remove], int(policy["near_duplicate_hamming"]))
    kept = img.loc[~img["image_uid"].isin(exclusions["image_uid"])].copy()
    counts = kept.groupby("observation_uid").size()
    observations = obs.loc[obs["observation_uid"].isin(counts.index)].copy()
    observations["n_images"] = observations["observation_uid"].map(counts).astype(int)
    integrity = joined.loc[~remove, ["image_uid", "filename", "sha256"]].sort_values("image_uid").reset_index(drop=True)
    small_mask = pd.to_numeric(joined["width"]).lt(policy["min_side_px"]) | pd.to_numeric(joined["height"]).lt(policy["min_side_px"])
    small = joined.loc[small_mask & ~remove, ["image_uid", "observation_uid", "split", "genus", "width", "height"]]
    small = small.sort_values("image_uid").reset_index(drop=True)
    return observations, kept, exclusions, near, integrity, small
