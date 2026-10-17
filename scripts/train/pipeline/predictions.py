"""Read the predictions CSV written by ``pipeline.evaluation.save_predictions`` (no PyTorch dependency)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ID_COLUMNS = ["image_uid", "observation_uid", "label_id", "pred_label_id"]


def load_predictions(path: Path, classes: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    """Return the id columns and the ``(n_images, n_classes)`` probability matrix in ``classes`` order."""
    df = pd.read_csv(path)
    expected = [f"p_{c}" for c in classes]
    if [c for c in df.columns if c.startswith("p_")] != expected:
        raise ValueError(f"{path}: probability columns do not match the class list {classes}")
    if df["image_uid"].duplicated().any():
        raise ValueError(f"{path}: duplicated image_uid")
    return df[ID_COLUMNS].copy(), df[expected].to_numpy(dtype=float)
