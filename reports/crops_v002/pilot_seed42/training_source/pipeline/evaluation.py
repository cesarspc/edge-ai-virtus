"""Inference and evaluation: softmax probabilities, image/observation metrics and the predictions CSV."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from metrics import aggregate_observations, classification_metrics
from progress_ui import bar


@torch.no_grad()
def predict_batch(model: nn.Module, x: torch.Tensor, device: torch.device, amp: bool) -> np.ndarray:
    """One photo score per row; forward each view separately to respect the GPU batch limit."""
    if model.training:
        raise ValueError("prediction requires eval mode (including BatchNorm)")
    if x.ndim not in (4, 5):
        raise ValueError("expected B,C,H,W images or B,V,C,H,W views")
    views = [x] if x.ndim == 4 else x.unbind(dim=1)
    photo_probs = []
    for view in views:
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(view.contiguous().to(device, non_blocking=True))
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Non-finite logits during evaluation; reduce evaluation.batch_size or disable AMP.")
        probs = torch.softmax(logits.float(), dim=1)
        if not torch.isfinite(probs).all():
            raise FloatingPointError("Non-finite probabilities during evaluation; reduce evaluation.batch_size or disable AMP.")
        photo_probs.append(probs.cpu().numpy())
    return photo_probs[0] if len(photo_probs) == 1 else np.stack(photo_probs).mean(0)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device, amp: bool,
            description: str = "Evaluacion") -> np.ndarray:
    model.eval()
    out = []
    for x, _ in bar(loader, desc=description, unit="lote"):
        out.append(predict_batch(model, x, device, amp))
    return np.concatenate(out) if out else np.zeros((0, 0))


def evaluate(frame: pd.DataFrame, probs: np.ndarray, classes: list[str], top_k: list[int]) -> dict[str, Any]:
    """Image-level metrics and observation-level metrics (mean of the image probabilities per observation)."""
    y = frame["label_id"].to_numpy()
    img = classification_metrics(y, probs, classes, top_k)
    yo, po = aggregate_observations(frame["image_uid"], frame["observation_uid"], y, probs)
    return {"image_level": img, "observation_level": classification_metrics(yo, po, classes, top_k)}


def save_predictions(path: Path, frame: pd.DataFrame, probs: np.ndarray, classes: list[str]) -> None:
    df = frame[["image_uid", "observation_uid", "label_id"]].copy()
    df["pred_label_id"] = probs.argmax(axis=1)
    for i, c in enumerate(classes):
        df[f"p_{c}"] = probs[:, i].round(6)
    df.to_csv(path, index=False, lineterminator="\n")
