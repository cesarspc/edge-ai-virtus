"""Explicit incremental modalities: S8 GPS only, S9 GPS + CHELSA normals.

No labels, dates, habitat, elevation, substrate or IDs enter the encoder.
The legacy EnvironmentalEncoder remains available to read exploratory runs.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

BIO = tuple(f"Bio{i}" for i in range(1, 20))
GEO = ("latitude", "longitude", "coordinate_uncertainty_m", "geo__quality")


def broad_substrate(frame):
    """Use the same coarse vocabulary for detailed and LOW_PRECISION wood records."""
    values = frame["substrate_class"].fillna("").astype(str)
    values = values.replace({"wood_dead": "wood", "wood_living": "wood", "wood_unspecified": "wood"})
    quality = frame["substrate__quality"]
    return values.where(quality.isin(["OK", "LOW_PRECISION", "CORRECTED"]), "")


def geo_valid(frame):
    lat = pd.to_numeric(frame.latitude, errors="coerce")
    lon = pd.to_numeric(frame.longitude, errors="coerce")
    return (lat.between(-90, 90) & lon.between(-180, 180) & ~(lat.eq(0) & lon.eq(0))
            & frame["geo__quality"].eq("OK"))


@dataclass
class ContextEncoder:
    feature_set: str
    climate_mean: list[float]
    climate_std: list[float]
    coordinate_decimals: int = 1
    substrate_categories: list[str] | None = None

    @classmethod
    def fit(cls, observations, feature_set="geo", coordinate_decimals=1):
        if observations.empty or not observations.split.eq("train").all():
            raise ValueError("context encoder requires nonempty train observations only")
        if feature_set not in ("geo", "geo_climate", "geo_climate_substrate"):
            raise ValueError("unsupported incremental feature_set")
        if coordinate_decimals != 1:
            raise ValueError("protocol fixes location rounding at 0.1 degree")
        mean, std = [], []
        if feature_set != "geo":
            values = observations[list(BIO)].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
            mean = values.mean().fillna(0).tolist()
            std = values.std(ddof=0).fillna(1).where(lambda s: s > 0, 1).tolist()
        categories = None
        if feature_set == "geo_climate_substrate":
            categories = sorted(set(broad_substrate(observations)) - {""})
        return cls(feature_set, mean, std, coordinate_decimals, categories)

    @property
    def fields(self):
        return (list(GEO) + (list(BIO) if self.feature_set != "geo" else []) +
                (["substrate_class", "substrate__quality"] if self.substrate_categories is not None else []))

    @property
    def feature_names(self):
        return ["geo_x", "geo_y", "geo_z", "geo_missing"] + (
            [f"{b}_z" for b in BIO] + [f"{b}_missing" for b in BIO]
            if self.feature_set != "geo" else []) + (
                [f"substrate_{v}" for v in self.substrate_categories] + ["substrate_unknown"]
                if self.substrate_categories is not None else [])

    def transform(self, frame):
        valid = geo_valid(frame).to_numpy()
        lat = np.radians(pd.to_numeric(frame.latitude, errors="coerce").where(valid, 0).round(self.coordinate_decimals))
        lon = np.radians(pd.to_numeric(frame.longitude, errors="coerce").where(valid, 0).round(self.coordinate_decimals))
        xyz = np.column_stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
        blocks = [np.where(valid[:, None], xyz, 0), (~valid)[:, None]]
        if self.feature_set != "geo":
            values = frame[list(BIO)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            missing = ~np.isfinite(values)
            z = (values - np.asarray(self.climate_mean)) / np.asarray(self.climate_std)
            blocks.extend([np.where(missing, 0, z), missing])
        if self.substrate_categories is not None:
            values = broad_substrate(frame)
            blocks.append(np.column_stack([(values == v).to_numpy() for v in self.substrate_categories] +
                                          [(~values.isin(self.substrate_categories)).to_numpy()]))
        result = np.concatenate(blocks, axis=1).astype(np.float32)
        if not np.isfinite(result).all():
            raise ValueError("non-finite context features")
        return result

    def state_dict(self):
        return {"feature_set": self.feature_set, "climate_mean": self.climate_mean,
                "climate_std": self.climate_std, "coordinate_decimals": self.coordinate_decimals,
                "substrate_categories": self.substrate_categories}

    @classmethod
    def from_state_dict(cls, state):
        return cls(**state)


def perturb_context(frame, condition, seed, feature_set):
    """Shuffle one complete modality per observation, preserving image row order."""
    result = frame.copy()
    groups = {"geo": list(GEO), "climate": list(BIO), "substrate": ["substrate_class", "substrate__quality"]}
    active = ["geo"] + (["climate"] if feature_set != "geo" else [])
    if feature_set == "geo_climate_substrate":
        active.append("substrate")
    if condition == "normal":
        return result
    if condition == "imprecise_geo":
        # Fixed one-degree rounding; sensitivity only, not simulated Android accuracy.
        result["latitude"] = result.latitude.round(0)
        result["longitude"] = result.longitude.round(0)
        return result
    action, _, modality = condition.partition("_")
    selected = active if modality == "all" else [modality]
    if action not in ("missing", "shuffled") or not set(selected).issubset(active):
        raise ValueError(f"invalid context condition: {condition}")
    for group in selected:
        fields = groups[group]
        if action == "missing":
            result[fields] = np.nan
            if group == "geo":
                result["geo__quality"] = "MISSING"
            elif group == "substrate":
                result["substrate__quality"] = "MISSING"
        else:
            obs = result.drop_duplicates("observation_uid").sort_values("observation_uid")
            shuffled = obs[fields].iloc[np.random.default_rng(seed).permutation(len(obs))].copy()
            shuffled.index = obs.observation_uid
            result[fields] = shuffled.reindex(result.observation_uid).to_numpy()
    return result


def dropout_context(metadata, probability):
    """Train on absent modalities without changing validation preprocessing."""
    if not 0 <= probability < 1:
        raise ValueError("metadata_dropout must be in [0,1)")
    result = metadata.clone()
    drop = torch.rand(len(result), device=result.device) < probability
    result[drop, :3] = 0
    result[drop, 3] = 1
    if result.shape[1] > 4:
        drop = torch.rand(len(result), device=result.device) < probability
        result[drop, 4:23] = 0
        result[drop, 23:42] = 1
    if result.shape[1] > 42:
        drop = torch.rand(len(result), device=result.device) < probability
        result[drop, 42:] = 0
        result[drop, -1] = 1
    return result


class ContextFusion(nn.Module):
    """Exact fallback to the trained visual branch when every modality is absent."""
    def __init__(self, visual_model, metadata_features, num_classes):
        super().__init__()
        self.visual_model = visual_model
        self.metadata_head = nn.Linear(metadata_features, num_classes)
        nn.init.zeros_(self.metadata_head.weight)
        nn.init.zeros_(self.metadata_head.bias)

    def forward(self, image, metadata):
        available = metadata[:, 3] < 0.5
        if metadata.shape[1] > 4:
            available = available | (metadata[:, 23:42] < 0.5).any(dim=1)
        if metadata.shape[1] > 42:
            available = available | metadata[:, 42:-1].gt(0).any(dim=1)
        return self.visual_model(image) + self.metadata_head(metadata) * available[:, None]


def load_context(cfg, repo_root, fingerprint, sha256_file):
    """Validate immutable sidecar provenance before joining it to image tables."""
    root = Path(repo_root) / cfg["context"]["directory"]
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest["dataset_fingerprint"] != fingerprint or manifest["splits"] != ["train", "val"]:
        raise ValueError("context sidecar does not match the train/val dataset")
    for entry in manifest["inputs"] + manifest["outputs"]:
        if sha256_file(Path(repo_root) / entry["path"]) != entry["sha256"]:
            raise ValueError(f"context provenance mismatch: {entry['path']}")
    frame = pd.read_csv(root / "observations.csv", low_memory=False)
    if not frame.split.isin(["train", "val"]).all() or frame.observation_uid.duplicated().any():
        raise ValueError("invalid context observation cohort")
    return frame, sha256_file(root / "manifest.json")
