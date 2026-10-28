"""S6: metrics at higher taxonomic levels computed from saved genus predictions (no PyTorch, no images).

The model predicts the 10 genera. For a level L (family, order, class, phylum) each genus belongs to exactly one
taxon of L (``data/taxonomy/<version>/genus_taxonomy.csv``).

Primary rule (``top1_genus``): the predicted taxon is the taxon of the predicted (top-1) genus, i.e. the decision the
model actually returns, judged at a coarser level. A correct genus implies correct higher ranks, so accuracy never
decreases when moving up. At the genus level this reproduces the saved ``metrics_<split>.json`` exactly.

Secondary rule (``marginal``): the probability of a taxon is the sum of the probabilities of its genera and the
prediction is its argmax. Reported for comparison only: with label smoothing every genus keeps a probability floor,
so summing favours taxa that group many genera (e.g. Agaricomycetes, 7 of 10 genera).

Observation level: genus probabilities are averaged over the images of the observation (``mean_prob``, as in
training) before applying either rule. Confidence intervals are percentile bootstrap over observations (the
statistical unit), never over images.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from metrics import aggregate_observations, classification_metrics

LEVELS = ["genus", "family", "order", "class", "phylum"]


def load_taxonomy(path: Path, classes: list[str]) -> pd.DataFrame:
    tax = pd.read_csv(path).sort_values("label_id").reset_index(drop=True)
    if tax["genus"].tolist() != classes:
        raise ValueError(f"{path}: genera {tax['genus'].tolist()} do not match the model classes {classes}")
    if tax[LEVELS].isna().any().any():
        raise ValueError(f"{path}: incomplete taxonomy")
    return tax


def level_index(tax: pd.DataFrame, level: str) -> tuple[list[str], np.ndarray]:
    """Sorted taxon names of ``level`` and, for each genus label_id, the index of its taxon."""
    names = sorted(tax[level].unique())
    position = {n: i for i, n in enumerate(names)}
    return names, tax[level].map(position).to_numpy()


def marginalize(probs: np.ndarray, genus_to_taxon: np.ndarray, n_taxa: int) -> np.ndarray:
    out = np.zeros((probs.shape[0], n_taxa))
    for g, t in enumerate(genus_to_taxon):
        out[:, t] += probs[:, g]
    return out


def macro_f1(y: np.ndarray, pred: np.ndarray, n: int) -> float:
    """Macro-F1 over all ``n`` labels; an undefined F1 (label absent from y and pred) counts as 0, like sklearn."""
    cm = np.bincount(y * n + pred, minlength=n * n).reshape(n, n)
    tp = np.diag(cm).astype(float)
    denom = cm.sum(axis=0) + cm.sum(axis=1)
    f1 = np.divide(2 * tp, denom, out=np.zeros(n), where=denom > 0)
    return float(f1.mean())


def bootstrap_macro_f1(y: np.ndarray, pred: np.ndarray, n: int, resamples: int, seed: int,
                       level: float = 0.95) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=(resamples, len(y)))
    scores = np.array([macro_f1(y[i], pred[i], n) for i in idx])
    a = (1 - level) / 2
    return float(np.quantile(scores, a)), float(np.quantile(scores, 1 - a))


def level_metrics(y_genus: np.ndarray, probs_genus: np.ndarray, tax: pd.DataFrame, level: str,
                  top_k: list[int]) -> dict[str, Any]:
    """Primary metrics (taxon of the top-1 genus) plus the marginalization rule under ``marginal``."""
    names, g2t = level_index(tax, level)
    n = len(names)
    y = g2t[y_genus]
    pred = g2t[probs_genus.argmax(axis=1)]
    m = classification_metrics(y, np.eye(n)[pred], names, [], name_key="taxon")   # one-hot: metrics of `pred`
    m.update(level=level, rule="top1_genus", n_classes=n,
             macro_precision=float(np.mean([c["precision"] for c in m["per_class"]])),
             macro_recall=float(np.mean([c["recall"] for c in m["per_class"]])),
             genera_per_taxon={t: tax.loc[g2t == i, "genus"].tolist() for i, t in enumerate(names)})
    marginal = classification_metrics(y, marginalize(probs_genus, g2t, n), names, [k for k in top_k if k < n])
    m["marginal"] = {k: v for k, v in marginal.items() if k not in ("per_class", "confusion_matrix", "n")}
    m["marginal"]["disagreements_with_top1_genus"] = int((marginalize(probs_genus, g2t, n).argmax(axis=1) != pred).sum())
    return m


def taxonomic_report(pred: pd.DataFrame, probs: np.ndarray, tax: pd.DataFrame, top_k: list[int],
                     resamples: int = 1000, seed: int = 42) -> dict[str, Any]:
    """All levels at image and observation level for one predictions table."""
    y = pred["label_id"].to_numpy().astype(int)
    yo, po = aggregate_observations(pred["image_uid"], pred["observation_uid"], y, probs)
    yo = yo.astype(int)
    out: dict[str, Any] = {"image_level": {}, "observation_level": {}}
    for level in LEVELS:
        out["image_level"][level] = level_metrics(y, probs, tax, level, top_k)
        m = level_metrics(yo, po, tax, level, top_k)
        _, g2t = level_index(tax, level)
        lo, hi = bootstrap_macro_f1(g2t[yo], g2t[po.argmax(axis=1)], m["n_classes"], resamples, seed)
        m["macro_f1_ci95"] = [lo, hi]
        m["ci_method"] = f"percentile bootstrap over observations, {resamples} resamples, seed {seed}"
        out["observation_level"][level] = m
    return out


def species_breakdown(pred: pd.DataFrame, probs: np.ndarray, species: pd.DataFrame, classes: list[str]) -> pd.DataFrame:
    """Observation-level genus recall per true species, with the most frequent wrong genus."""
    y = pred["label_id"].to_numpy().astype(int)
    obs_ids = pred.groupby("observation_uid", sort=True).size().index
    yo, po = aggregate_observations(pred["image_uid"], pred["observation_uid"], y, probs)
    frame = pd.DataFrame({"observation_uid": obs_ids, "label_id": yo.astype(int), "pred": po.argmax(axis=1)})
    frame = frame.merge(species[["observation_uid", "species"]], on="observation_uid", how="left", validate="one_to_one")
    if frame["species"].isna().any():
        raise ValueError("observations without species in the taxonomy table")
    rows = []
    for (label, sp), g in frame.groupby(["label_id", "species"], sort=True):
        top_wrong = g.loc[g["pred"] != label, "pred"].value_counts()
        rows.append({"genus": classes[label], "species": sp, "observations": len(g),
                     "correct_genus": int((g["pred"] == label).sum()),
                     "genus_recall": float((g["pred"] == label).mean()),
                     "most_common_wrong_genus": classes[top_wrong.index[0]] if len(top_wrong) else "",
                     "most_common_wrong_count": int(top_wrong.iloc[0]) if len(top_wrong) else 0})
    return pd.DataFrame(rows)
