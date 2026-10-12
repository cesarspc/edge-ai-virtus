"""Reproducible train/val/test partitions (Week 3).

Scheme ``grouped_hash``:

1. Every observation gets a ``group_id``. Observations with a valid day-precision date and valid coordinates are
   grouped by *site-day*: same date and same ~1 km cell (coordinates rounded to ``group_coordinate_decimals``).
   Observations without date or coordinates form their own group (``obs:<observation_uid>``).
2. Each group is assigned to a split by hashing ``salt|group_id`` with SHA-256 and mapping it to [0, 1).
   All the observations — and therefore all the images — of a group land in the same split.

Properties: no image, observation or site-day is shared between splits; the assignment of a group never depends on
the other groups, so adding data in a later version does not move existing groups (stable across versions).
"""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd


def group_ids(obs: pd.DataFrame, decimals: int) -> pd.Series:
    ok = (obs["event_date_precision"] == "day") & obs["geo__quality"].isin(["OK", "LOW_PRECISION", "SUSPECT"])
    fmt = f"{{:.{decimals}f}}"
    site_day = ("sd:" + obs["event_date"].astype(str) + ":" + obs["latitude"].map(lambda v: fmt.format(v) if pd.notna(v) else "")
                + ":" + obs["longitude"].map(lambda v: fmt.format(v) if pd.notna(v) else ""))
    return site_day.where(ok, "obs:" + obs["observation_uid"])


def hash_unit(keys: pd.Series, salt: str) -> pd.Series:
    """Deterministic uniform value in [0, 1) for each key."""
    return keys.map(lambda k: int(hashlib.sha256(f"{salt}|{k}".encode("utf-8")).hexdigest()[:15], 16) / 16 ** 15)


def assign(obs: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    sp = cfg["splits"]
    obs = obs.copy()
    obs["group_id"] = group_ids(obs, sp["group_coordinate_decimals"])
    if sp["scheme"] == "fungitastic_official":
        if set(obs["source_id"]) != {"fungitastic"} or not obs["split_temporal"].isin(["train", "val", "test"]).all():
            raise ValueError("official splits require FungiTastic with a known original partition")
        obs["split"] = obs["split_temporal"]
        return obs
    if sp["scheme"] != "grouped_hash":
        raise ValueError(f"unknown split scheme: {sp['scheme']}")
    u = hash_unit(obs["group_id"], sp["salt"])
    r = sp["ratios"]
    dev_split = np.select([u < r["train"], u < r["train"] + r["val"]], ["train", "val"], "test")
    roles = obs["source_id"].map({k: v["role"] for k, v in cfg["sources"].items()})
    obs["split"] = np.where(roles == "development", dev_split, "test_external")
    return obs


def leakage_checks(obs: pd.DataFrame, img: pd.DataFrame) -> dict[str, int]:
    """Counts that must all be 0."""
    dev = obs[obs["split"] != "test_external"]
    g = dev.groupby("group_id")["split"].nunique()
    per_obs = img.groupby("observation_uid")["split"].nunique()
    return {
        "groups_in_more_than_one_split": int((g > 1).sum()),
        "observations_with_images_in_more_than_one_split": int((per_obs > 1).sum()),
        "images_in_more_than_one_split": int(img.groupby("image_uid")["split"].nunique().gt(1).sum()),
        "observation_uid_duplicated": int(obs["observation_uid"].duplicated().sum()),
        "image_uid_duplicated": int(img["image_uid"].duplicated().sum()),
    }
