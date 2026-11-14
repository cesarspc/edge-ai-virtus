"""Audit saved validation predictions; never load test predictions or change labels.

Defaults to the historical S8 exploratory runs. Pass corrected run directories
with --runs once A/B/C finish to regenerate the same error review artifacts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/train"))
from data import sha256_file
from metrics import classification_metrics


def observation_errors(predictions, classes):
    columns = [f"p_{c}" for c in classes]
    if predictions.image_uid.duplicated().any():
        raise ValueError("duplicate image predictions")
    if predictions.groupby("observation_uid").label_id.nunique().gt(1).any():
        raise ValueError("conflicting labels within an observation")
    probs = predictions[columns].to_numpy(dtype=float)
    if (not np.isfinite(probs).all() or (probs < 0).any() or (probs > 1).any()
            or not np.allclose(probs.sum(axis=1), 1, atol=2e-5)):
        raise ValueError("invalid probability rows")
    frame = predictions.copy()
    frame["image_prediction"] = probs.argmax(axis=1)
    frame["image_entropy"] = -(probs * np.log(np.clip(probs, 1e-12, 1))).sum(axis=1)
    grouped = frame.groupby("observation_uid", sort=True)
    means = grouped[columns].mean()
    avg = means.to_numpy()
    out = means.copy()
    out["label_id"] = grouped.label_id.first()
    out["pred_label_id"] = avg.argmax(axis=1)
    out["true_genus"] = [classes[i] for i in out.label_id]
    out["pred_genus"] = [classes[i] for i in out.pred_label_id]
    out["correct"] = out.label_id.eq(out.pred_label_id)
    out["confidence"] = avg.max(axis=1)
    out["true_probability"] = avg[np.arange(len(out)), out.label_id.to_numpy()]
    out["n_images"] = grouped.size()
    out["distinct_image_predictions"] = grouped.image_prediction.nunique()
    out["disagreement_rate"] = 1 - grouped.image_prediction.agg(lambda s: s.value_counts().iloc[0] / len(s))
    out["entropy"] = -(avg * np.log(np.clip(avg, 1e-12, 1))).sum(axis=1)
    out["view_js_divergence"] = np.maximum(0, out.entropy - grouped.image_entropy.mean())
    out["image_uids"] = grouped.image_uid.agg(lambda s: "|".join(s))
    return out.reset_index()


def markdown_table(frame):
    headers = list(frame.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for values in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in values) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+")
    ap.add_argument("--context", default="data/processed/s8_s9_context_v002")
    ap.add_argument("--output", default="reports/s9/validation")
    args = ap.parse_args()
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    context = ROOT / args.context
    sidecar_manifest = json.loads((context / "manifest.json").read_text(encoding="utf-8"))
    for entry in sidecar_manifest["inputs"] + sidecar_manifest["outputs"]:
        if sha256_file(ROOT / entry["path"]) != entry["sha256"]:
            raise ValueError(f"changed sidecar input/output: {entry['path']}")
    audit = pd.read_csv(context / "metadata_audit.csv", low_memory=False)
    context_rows = pd.read_csv(context / "observations.csv", low_memory=False)
    cells = context_rows[["latitude", "longitude"]].round(1).astype(str).agg(":".join, axis=1)
    known_cells = set(cells[context_rows.split.eq("train") & context_rows["geo__quality"].eq("OK")])
    context_rows["seen_train_cell"] = cells.isin(known_cells)
    context_rows["climate_complete"] = context_rows[[f"Bio{i}" for i in range(1, 20)]].notna().all(axis=1)
    val = audit[audit.split.eq("val")]
    train_support = audit[audit.split.eq("train")].groupby("genus").size()
    classes = pd.read_csv(ROOT / "data/versions/v002/classes.csv").sort_values("label_id").genus.tolist()
    runs = [ROOT / p for p in args.runs] if args.runs else sorted((ROOT / "runs/s8_multimodal_v002").glob("*"))
    summary, comparisons, provenance, report = [], [], [], ["# S9: revisión de errores de validación", "",
        "Las corridas marcadas legacy_environment usan hábitat, sustrato, elevación y fecha: "
        "no estiman el aporte de GPS ni de clima. Las corridas corregidas declaran geo o geo_climate. No se leyó test.", "",
        "La cola prioriza errores confiados, errores de clases con poco train y vistas discordantes. "
        "Una discrepancia entre vistas no demuestra error de etiqueta; la adjudicación humana sigue PENDING. "
        "La inspección visual preliminar de seis casos se documenta aparte en ../visual_review/REVIEW.md; "
        "la cola completa no está adjudicada.", ""]
    reference = None
    previous = None
    for run in runs:
        if not (run / "completed.json").is_file():
            raise ValueError(f"run is not completed: {run}")
        metadata = json.loads((run / "run.json").read_text(encoding="utf-8"))
        if metadata["dataset_fingerprint"] != sidecar_manifest["dataset_fingerprint"]:
            raise ValueError("run dataset differs from sidecar")
        if metadata["classes"] != classes:
            raise ValueError("class order mismatch")
        predictions_path = run / "predictions_val.csv"
        predictions = pd.read_csv(predictions_path)
        if set(predictions.observation_uid) != set(val.observation_uid):
            raise ValueError("prediction observation cohort differs from frozen val")
        labels = val.set_index("observation_uid").label_id
        if not predictions.label_id.eq(predictions.observation_uid.map(labels)).all():
            raise ValueError("predictions disagree with frozen labels")
        # Check every image, not only the set of observations.
        raw_images = pd.read_csv(context / "raw_images.csv")
        expected_images = raw_images[raw_images.split.eq("val")].copy()
        expected_images["image_uid"] = "fungitastic:" + expected_images.filename
        # Frozen v002 integrity list excludes any curated image removals.
        integrity = pd.read_csv(ROOT / "data/versions/v002/image_integrity.csv", usecols=["image_uid"])
        expected_images = expected_images[expected_images.image_uid.isin(integrity.image_uid)]
        expected_pairs = expected_images.set_index("image_uid").observation_uid.sort_index()
        actual_pairs = predictions.set_index("image_uid").observation_uid.sort_index()
        if not actual_pairs.equals(expected_pairs):
            raise ValueError("image/observation mapping or image cohort mismatch")
        errors = observation_errors(predictions, classes)
        local = out / run.name
        local.mkdir(exist_ok=True)
        errors.to_csv(local / "observations.csv", index=False, lineterminator="\n")
        metrics = classification_metrics(errors.label_id.to_numpy(), errors[[f"p_{c}" for c in classes]].to_numpy(), classes, [1, 3])
        per_class = pd.DataFrame(metrics["per_class"])
        per_class["train_observations"] = per_class.genus.map(train_support)
        per_class["small_train_class"] = per_class.train_observations.le(train_support.quantile(.25))
        per_class.to_csv(local / "per_class.csv", index=False, lineterminator="\n")
        confusion = (errors[~errors.correct].groupby(["true_genus", "pred_genus"]).size()
                     .rename("errors").reset_index().sort_values("errors", ascending=False))
        support = errors.true_genus.value_counts()
        confusion["fraction_of_true_class"] = confusion.errors / confusion.true_genus.map(support)
        confusion.to_csv(local / "confusions.csv", index=False, lineterminator="\n")
        annotated = errors.merge(val[["observation_uid", "substrate__quality", "habitat__quality", "elevation__quality",
                                       "event_date__quality"]], on="observation_uid", validate="one_to_one")
        quality = []
        for field in ["substrate", "habitat", "elevation", "event_date"]:
            for name, group in annotated.groupby(f"{field}__quality"):
                quality.append({"field": field, "quality": name, "observations": len(group),
                                "errors": int((~group.correct).sum()), "accuracy": float(group.correct.mean())})
        pd.DataFrame(quality).to_csv(local / "quality_strata.csv", index=False, lineterminator="\n")
        context_errors = errors.merge(context_rows[["observation_uid", "seen_train_cell", "climate_complete"]],
                                     on="observation_uid", validate="one_to_one")
        strata = []
        for field in ("seen_train_cell", "climate_complete"):
            for name, group in context_errors.groupby(field):
                strata.append({"field": field, "value": bool(name), "observations": len(group),
                               "represented_classes": int(group.label_id.nunique()),
                               "errors": int((~group.correct).sum()), "accuracy": float(group.correct.mean()),
                               "macro_f1_fixed_10_classes": f1_score(group.label_id, group.pred_label_id,
                                   labels=list(range(len(classes))), average="macro", zero_division=0)})
        pd.DataFrame(strata).to_csv(local / "context_strata.csv", index=False, lineterminator="\n")
        wrong = errors[~errors.correct].sort_values("confidence", ascending=False)
        discordant = errors[errors.n_images.gt(1) & errors.distinct_image_predictions.gt(1)].sort_values("view_js_divergence", ascending=False)
        queue = pd.concat([wrong.head(25), discordant.head(25),
                           wrong[wrong.true_genus.isin(per_class.loc[per_class.small_train_class, "genus"])].head(25)])
        queue = queue.drop_duplicates("observation_uid").copy()
        for field in ["review_status", "suspected_cause", "reviewer", "notes"]:
            queue[field] = "PENDING" if field == "review_status" else ""
        queue.to_csv(local / "manual_review_queue.csv", index=False, lineterminator="\n")
        summary.append({"run": run.name, "feature_set": metadata["config"].get("context", {}).get("feature_set", "legacy_environment"),
                        "mode": metadata["config"]["mode"], "observations": len(errors), "errors": len(wrong),
                        "macro_f1": metrics["macro_f1"], "accuracy": metrics["accuracy"],
                        "multi_image_observations": int(errors.n_images.gt(1).sum()),
                        "discordant_observations": len(discordant)})
        current = (run.name, errors[["observation_uid", "correct"]].rename(columns={"correct": "reference_correct"}),
                   metrics["macro_f1"], metrics["accuracy"])
        if reference is None:
            reference = current
        else:
            references = [reference] if previous[0] == reference[0] else [reference, previous]
            for ref_name, ref_errors, ref_f1, ref_accuracy in references:
                pair = errors.merge(ref_errors, on="observation_uid", validate="one_to_one")
                comparisons.append({"run": run.name, "reference": ref_name,
                                    "delta_macro_f1": metrics["macro_f1"] - ref_f1,
                                    "delta_accuracy": metrics["accuracy"] - ref_accuracy,
                                    "fixed_errors": int((pair.correct & ~pair.reference_correct).sum()),
                                    "introduced_errors": int((~pair.correct & pair.reference_correct).sum())})
        previous = current
        provenance.append({"run": str(run.relative_to(ROOT)), "predictions_sha256": sha256_file(predictions_path),
                           "run_json_sha256": sha256_file(run / "run.json")})
        report.extend([f"## {run.name}", "", markdown_table(per_class[["genus", "train_observations", "support", "recall", "f1"]]),
                       "", "Confusiones más frecuentes (por observación):", "", markdown_table(confusion.head(8)), ""])
    pd.DataFrame(summary).to_csv(out / "summary.csv", index=False, lineterminator="\n")
    pd.DataFrame(comparisons).to_csv(out / "paired_errors.csv", index=False, lineterminator="\n")
    (out / "provenance.json").write_text(json.dumps({"context_manifest_sha256": sha256_file(context / "manifest.json"),
                                                  "runs": provenance}, indent=2) + "\n", encoding="utf-8")
    report[4:4] = [markdown_table(pd.DataFrame(summary)), ""]
    (out / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
