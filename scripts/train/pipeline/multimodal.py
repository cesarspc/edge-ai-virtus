"""Train-only tabular encoding and image/metadata fusion for S8 experiments.

The encoder is explicitly fitted on train observations and must be saved with the
model. Missing or unavailable information is represented by separate indicators.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset
from PIL import Image


@dataclass
class EnvironmentalEncoder:
    categories: dict[str, list[str]]
    elevation_mean: float
    elevation_std: float

    CATEGORICAL = ("habitat_class", "substrate_class")

    @classmethod
    def fit(cls, observations: pd.DataFrame) -> "EnvironmentalEncoder":
        if observations.empty or not observations["split"].eq("train").all():
            raise ValueError("encoder must be fitted exclusively on nonempty train observations")
        categories = {name: sorted(observations[name].dropna().astype(str).unique().tolist())
                      for name in cls.CATEGORICAL}
        elevation = pd.to_numeric(observations["elevation_m"], errors="coerce")
        valid = elevation.where(observations["elevation__quality"].eq("OK"))
        mean = float(valid.mean()) if valid.notna().any() else 0.0
        std = float(valid.std(ddof=0)) if valid.notna().any() else 1.0
        return cls(categories, mean, std if math.isfinite(std) and std > 0 else 1.0)

    def transform(self, observations: pd.DataFrame) -> np.ndarray:
        blocks = []
        for name in self.CATEGORICAL:
            values = observations[name].fillna("").astype(str)
            known = self.categories[name]
            blocks.append(np.column_stack([(values == category).to_numpy(dtype=np.float32)
                                           for category in known] +
                                          [(~values.isin(known)).to_numpy(dtype=np.float32)]))
        elevation = pd.to_numeric(observations["elevation_m"], errors="coerce")
        valid = elevation.notna() & observations["elevation__quality"].eq("OK")
        blocks.append(np.column_stack([((elevation.where(valid, self.elevation_mean) - self.elevation_mean)
                                       / self.elevation_std).to_numpy(dtype=np.float32),
                                      (~valid).to_numpy(dtype=np.float32)]))
        date = pd.to_datetime(observations["event_date"], errors="coerce")
        valid_date = date.notna() & observations["event_date__quality"].eq("OK")
        angle = 2 * np.pi * (date.dt.dayofyear.fillna(1) - 1) / 365.25
        blocks.append(np.column_stack([np.where(valid_date, np.sin(angle), 0).astype(np.float32),
                                       np.where(valid_date, np.cos(angle), 0).astype(np.float32),
                                       (~valid_date).to_numpy(dtype=np.float32)]))
        return np.concatenate(blocks, axis=1).astype(np.float32)

    def state_dict(self) -> dict:
        return {"categories": self.categories, "elevation_mean": self.elevation_mean,
                "elevation_std": self.elevation_std}

    @classmethod
    def from_state_dict(cls, state: dict) -> "EnvironmentalEncoder":
        return cls(**state)


@dataclass
class GeolocationEncoder:
    """Train-only encoder for latitude/longitude; no quality flags enter B as features."""
    mean: list[float]
    std: list[float]

    @staticmethod
    def _coordinates(observations: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        lat = pd.to_numeric(observations["latitude"], errors="coerce").to_numpy(dtype=np.float64)
        lon = pd.to_numeric(observations["longitude"], errors="coerce").to_numpy(dtype=np.float64)
        quality = observations["geo__quality"].astype("string")
        valid = (np.isfinite(lat) & np.isfinite(lon) & quality.isin(["OK", "LOW_PRECISION"]).to_numpy()
                 & (lat >= -90) & (lat <= 90) & (lon >= -180) & (lon <= 180))
        lat_rad, lon_rad = np.deg2rad(np.where(valid, lat, 0)), np.deg2rad(np.where(valid, lon, 0))
        trig = np.column_stack([np.sin(lat_rad), np.cos(lat_rad), np.sin(lon_rad), np.cos(lon_rad)])
        return trig, valid

    @classmethod
    def fit(cls, observations: pd.DataFrame) -> "GeolocationEncoder":
        if observations.empty or not observations["split"].eq("train").all():
            raise ValueError("geolocation encoder must be fitted exclusively on nonempty train observations")
        trig, valid = cls._coordinates(observations)
        selected = trig[valid]
        mean = selected.mean(axis=0) if len(selected) else np.zeros(4)
        std = selected.std(axis=0) if len(selected) else np.ones(4)
        std = np.where(np.isfinite(std) & (std > 0), std, 1.0)
        return cls(mean.astype(float).tolist(), std.astype(float).tolist())

    def transform(self, observations: pd.DataFrame) -> np.ndarray:
        trig, valid = self._coordinates(observations)
        mean, std = np.asarray(self.mean), np.asarray(self.std)
        scaled = (np.where(valid[:, None], trig, mean) - mean) / std
        return scaled.astype(np.float32)

    def state_dict(self) -> dict:
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_state_dict(cls, state: dict) -> "GeolocationEncoder":
        return cls(**state)


@dataclass
class GeoClimateEncoder:
    """Train-only location and BIOCLIM encoder with explicit climate-missing mask."""
    geo: GeolocationEncoder
    climate_mean: list[float]
    climate_std: list[float]

    CLIMATE_FEATURES = tuple(f"Bio{i}" for i in range(1, 20))

    @staticmethod
    def _climate_available(observations: pd.DataFrame) -> np.ndarray:
        available = observations["climate_available"].fillna(False)
        if available.dtype == object or str(available.dtype).startswith("string"):
            available = available.astype(str).str.lower().isin(["true", "1", "yes"])
        # Pandas may expose a read-only NumPy view for an already-boolean Series.
        # The fit/transform methods combine this mask in place, so return an owned copy.
        return available.to_numpy(dtype=bool, copy=True)

    @classmethod
    def fit(cls, observations: pd.DataFrame) -> "GeoClimateEncoder":
        if observations.empty or not observations["split"].eq("train").all():
            raise ValueError("geo-climate encoder must be fitted exclusively on nonempty train observations")
        geo = GeolocationEncoder.fit(observations)
        climate = observations.loc[:, list(cls.CLIMATE_FEATURES)].apply(pd.to_numeric, errors="coerce")
        available = cls._climate_available(observations)
        available &= np.isfinite(climate.to_numpy(dtype=float)).all(axis=1)
        values = climate.to_numpy(dtype=np.float64)[available]
        if not len(values):
            raise ValueError("no complete BIOCLIM rows are available in train")
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        std = np.where(np.isfinite(std) & (std > 0), std, 1.0)
        return cls(geo, mean.astype(float).tolist(), std.astype(float).tolist())

    def transform(self, observations: pd.DataFrame) -> np.ndarray:
        geo = self.geo.transform(observations)
        climate = observations.loc[:, list(self.CLIMATE_FEATURES)].apply(pd.to_numeric, errors="coerce")
        values = climate.to_numpy(dtype=np.float64)
        available = self._climate_available(observations)
        available &= np.isfinite(values).all(axis=1)
        mean, std = np.asarray(self.climate_mean), np.asarray(self.climate_std)
        scaled = (np.where(available[:, None], values, mean) - mean) / std
        return np.column_stack([geo, scaled, (~available).astype(np.float32)]).astype(np.float32)

    def state_dict(self) -> dict:
        return {"geo": self.geo.state_dict(), "climate_mean": self.climate_mean,
                "climate_std": self.climate_std}

    @classmethod
    def from_state_dict(cls, state: dict) -> "GeoClimateEncoder":
        state = dict(state)
        state["geo"] = GeolocationEncoder.from_state_dict(state["geo"])
        return cls(**state)


class VisualEnvironmentalModel(nn.Module):
    """Add a metadata residual to a pretrained visual model's logits.

    Zero initialization preserves visual predictions exactly before fine-tuning.
    """
    def __init__(self, visual_model: nn.Module, metadata_features: int, num_classes: int):
        super().__init__()
        self.visual_model = visual_model
        self.metadata_head = nn.Linear(metadata_features, num_classes)
        nn.init.zeros_(self.metadata_head.weight)
        nn.init.zeros_(self.metadata_head.bias)

    def forward(self, image: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        return self.visual_model(image) + self.metadata_head(metadata)


class EnvironmentalOnlyModel(nn.Module):
    def __init__(self, metadata_features: int, num_classes: int):
        super().__init__()
        self.head = nn.Linear(metadata_features, num_classes)

    def forward(self, image: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        return self.head(metadata)


class MultimodalTable(Dataset):
    def __init__(self, frame: pd.DataFrame, metadata: np.ndarray, transform, load_images: bool = True):
        if len(frame) != len(metadata):
            raise ValueError("metadata and image rows must have the same length")
        self.paths = frame["path"].tolist()
        self.labels = frame["label_id"].astype(int).tolist()
        self.metadata = torch.from_numpy(metadata)
        self.transform = transform
        self.load_images = load_images

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        if self.load_images:
            with Image.open(self.paths[i]) as image:
                x = self.transform(image.convert("RGB"))
        else:
            x = torch.zeros(1)
        return x, self.metadata[i], self.labels[i]


class VisualOnlyModel(nn.Module):
    def __init__(self, visual_model: nn.Module):
        super().__init__()
        self.visual_model = visual_model

    def forward(self, image: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        return self.visual_model(image)
