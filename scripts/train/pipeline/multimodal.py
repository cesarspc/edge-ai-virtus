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
