"""Evaluation metrics of the project scope (§4.1, §4.3): macro-F1 (main), accuracy, balanced accuracy, top-k,
per-class metrics and confusion matrix, at image and at observation level (no PyTorch dependency)."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)


def topk_accuracy(probs: np.ndarray, y: np.ndarray, k: int) -> float:
    k = min(k, probs.shape[1])
    top = np.argsort(-probs, axis=1)[:, :k]
    return float((top == y[:, None]).any(axis=1).mean())


def aggregate_observations(image_ids: pd.Series, obs_ids: pd.Series, y: np.ndarray, probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean of the image probabilities of each observation (all images of an observation share the label)."""
    df = pd.DataFrame(probs).assign(_obs=obs_ids.to_numpy(), _y=y)
    g = df.groupby("_obs", sort=True)
    return g["_y"].first().to_numpy(), g[list(range(probs.shape[1]))].mean().to_numpy()


def classification_metrics(y: np.ndarray, probs: np.ndarray, classes: list[str], top_k: list[int]) -> dict[str, Any]:
    labels = list(range(len(classes)))
    pred = probs.argmax(axis=1)
    p, r, f, n = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    out: dict[str, Any] = {
        "n": int(len(y)),
        "macro_f1": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
    }
    for k in top_k:
        out[f"top{k}_accuracy"] = topk_accuracy(probs, y, k)
    out["per_class"] = [{"label_id": i, "genus": c, "precision": float(p[i]), "recall": float(r[i]),
                         "f1": float(f[i]), "support": int(n[i])} for i, c in enumerate(classes)]
    out["confusion_matrix"] = confusion_matrix(y, pred, labels=labels).tolist()
    return out
