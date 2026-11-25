"""Reproduce the six S11 modality variants and export diagnostic figures."""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.nn import functional as F
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from data import resolve, REPO_ROOT, sha256_file
from pipeline.context import BIO, ContextEncoder, broad_substrate, geo_valid, load_context, perturb_context
from pipeline.models import build_model
from pipeline.transforms import build_transforms
from run_s9_s10_frozen import aligned_visual, mix_scores, paired_delta, score_group, write_json

VARIANTS = {
    "A": (), "B": ("geo",), "C": ("geo", "climate"),
    "D": ("geo", "climate", "substrate"), "E": ("climate",), "F": ("geo", "substrate"),
}
FULL = "geo_climate_substrate"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def available(frame, encoder):
    return {
        "geo": geo_valid(frame).to_numpy(),
        "climate": np.isfinite(frame[list(BIO)].to_numpy(float)).any(axis=1),
        "substrate": broad_substrate(frame).isin(encoder.substrate_categories).to_numpy(),
    }


def model_scores(train, frame, visual, encoder, modalities, cfg):
    feature_groups = [
        "geo" if name.startswith("geo_") else "substrate" if name.startswith("substrate_") else "climate"
        for name in encoder.feature_names
    ]
    columns = np.isin(feature_groups, modalities)
    features = encoder.transform(train)[:, columns]
    scaler = StandardScaler().fit(features)
    classifier = LogisticRegression(**cfg["logistic_regression"], random_state=cfg["seed"])
    classifier.fit(scaler.transform(features), train.label_id)
    if not np.array_equal(classifier.classes_, np.arange(visual.shape[1])):
        raise ValueError("Unexpected class order")

    def predict(data, image_scores):
        probabilities = classifier.predict_proba(scaler.transform(encoder.transform(data)[:, columns]))
        flags = available(data, encoder)
        has_context = np.logical_or.reduce([flags[m] for m in modalities])
        return mix_scores(image_scores, probabilities, has_context, cfg["context_weight"])

    yield "normal", predict(frame, visual), predict
    for modality in modalities:
        for action in ("missing", "shuffled"):
            condition = f"{action}_{modality}"
            yield condition, predict(perturb_context(frame, condition, cfg["seed"], FULL), visual), predict
    if "geo" in modalities:
        yield "imprecise_geo", predict(perturb_context(frame, "imprecise_geo", cfg["seed"], FULL), visual), predict
    yield "missing_all", predict(perturb_context(frame, "missing_all", cfg["seed"], FULL), visual), predict


def confusion_plot(metrics, classes, path):
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout="constrained", sharex=True, sharey=True)
    for (name, scores), ax in zip(metrics.items(), axes.flat):
        cm = np.array(scores["confusion_matrix"])
        # Show error counts with a common colour scale. Diagonal values are printed separately in grey.
        errors = cm.copy()
        np.fill_diagonal(errors, 0)
        image = ax.imshow(errors, cmap="Blues", vmin=0, vmax=12)
        ax.set_title(f"{name}   ·   {cm.sum() - cm.trace()} errores")
        for i, j in zip(*np.nonzero(cm)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                    color="#555555" if i == j else "white" if cm[i, j] >= 8 else "#17375a")
        ax.set_xticks(range(len(classes)), classes, rotation=55, ha="right", fontsize=8)
        ax.set_yticks(range(len(classes)), classes, fontsize=8)
    fig.supxlabel("Género predicho")
    fig.supylabel("Género real")
    fig.colorbar(image, ax=axes, label="Número de errores", shrink=.85, extend="max")
    fig.savefig(path, dpi=185)
    plt.close(fig)


def training_plot(run, visual_cfg, classes, path):
    history = pd.read_csv(run / "history.csv")
    best = history.loc[history.val_obs_macro_f1.idxmax()]
    predictions = pd.read_csv(run / "predictions_val.csv")
    weights = np.array(read_json(run / "class_balance.json")["loss_weights"])
    y = predictions.label_id.to_numpy()
    p = predictions[[f"p_{c}" for c in classes]].to_numpy()
    p /= p.sum(axis=1, keepdims=True)
    smooth = visual_cfg["training"]["label_smoothing"]
    targets = np.eye(len(classes))[y] * (1 - smooth) + smooth / len(classes)
    loss = -(targets * weights * np.log(np.clip(p, 5e-7, 1))).sum() / weights[y].sum()
    accuracy = history.val_img_accuracy * 100
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.35), layout="constrained")
    axes[0].plot(history.epoch, history.train_loss, color="#176478", label="Entrenamiento")
    axes[0].scatter([best.epoch], [loss], color="#bf4a27", zorder=3, label="Validación · punto recuperado")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("Entropía cruzada ponderada")
    axes[1].plot(history.epoch, accuracy, color="#176478")
    axes[1].scatter([best.epoch], [best.val_img_accuracy * 100], color="#bf4a27", zorder=3)
    axes[1].set_ylabel("Exactitud de validación por imagen (%)")
    for ax in axes:
        ax.set_xlabel("Época")
        ax.grid(alpha=.22)
    fig.savefig(path, dpi=185)
    plt.close(fig)
    return {"epochs": len(history), "best_epoch": int(best.epoch), "reconstructed_val_loss": float(loss),
            "train_loss_first": history.train_loss.iloc[0], "train_loss_last": history.train_loss.iloc[-1],
            "val_accuracy_first": history.val_img_accuracy.iloc[0], "val_accuracy_last": history.val_img_accuracy.iloc[-1],
            "val_accuracy_best_checkpoint": best.val_img_accuracy}


def gradcam(run, cfg, external, classes, destination):
    model = build_model(cfg, pretrained=False).eval()
    model.load_state_dict(torch.load(run / "best.pt", map_location="cpu", weights_only=True)["model"])
    _, transform = build_transforms(cfg)
    root = resolve("data/raw/images/external_v001")
    download = pd.read_csv(root / "download_manifest.csv").set_index("image_uid")
    integrity = pd.read_csv(resolve("data/versions/ext_full_001/image_integrity.csv")).set_index("image_uid")
    cases = external[external.image_uid.str.startswith("inaturalist:")].sort_values("image_uid")
    correct = cases.label_id.eq(cases.pred_label_id)
    # One correct example and one error from each of two distinct real genera.
    cases = pd.concat([cases[correct].head(1), cases[~correct].drop_duplicates("label_id").head(2)])
    fig, axes = plt.subplots(2, len(cases), figsize=(9, 6), layout="constrained")
    metadata = []
    for col, case in enumerate(cases.itertuples()):
        path = root / download.loc[case.image_uid, "filename"]
        if sha256_file(path) != integrity.loc[case.image_uid, "sha256"]:
            raise ValueError(f"Image hash mismatch: {case.image_uid}")
        with Image.open(path) as image:
            tensor = transform(image.convert("RGB")).unsqueeze(0)
        activation = model.features(tensor)
        logits = model.classifier(torch.flatten(model.avgpool(activation), 1))
        target = int(logits.argmax(1).item())
        if target != case.pred_label_id:
            raise ValueError(f"Grad-CAM prediction differs from archived result: {case.image_uid}")
        gradient, = torch.autograd.grad(logits[0, target], activation)
        cam = (gradient.mean(dim=(2, 3), keepdim=True) * activation).sum(1, keepdim=True).relu()
        cam = F.interpolate(cam, tensor.shape[-2:], mode="bilinear", align_corners=False).detach().squeeze().numpy()
        peak = float(cam.max())
        if peak <= 0:
            raise ValueError(f"No positive Grad-CAM response: {case.image_uid}")
        cam /= peak
        mean = np.array(cfg["input"]["mean"])
        std = np.array(cfg["input"]["std"])
        rgb = np.clip(tensor[0].permute(1, 2, 0).numpy() * std + mean, 0, 1)
        axes[0, col].imshow(rgb)
        axes[0, col].set_title(f"Real · {classes[case.label_id]}\nPredicción · {classes[target]}", fontsize=10)
        axes[1, col].imshow(rgb)
        axes[1, col].imshow(cam, cmap="inferno", vmin=0, vmax=1, alpha=.65 * cam)
        axes[1, col].set_xlabel(case.image_uid, fontsize=8)
        for ax in axes[:, col]:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(False)
        y, x = np.unravel_index(cam.argmax(), cam.shape)
        metadata.append({"image_uid": case.image_uid, "actual": classes[case.label_id],
                         "predicted": classes[target], "peak_xy_224": [int(x), int(y)]})
    fig.savefig(destination / "gradcam.png", dpi=165)
    plt.close(fig)
    return metadata


def main():
    torch.set_num_threads(4)
    cfg_path = resolve("configs/experiments/s9_s10_frozen_context.yaml")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    destination = resolve("reports/week_11")
    destination.mkdir(exist_ok=True)
    run = resolve(cfg["visual_run"])
    if sha256_file(run / "best.pt") != cfg["visual_checkpoint_sha256"]:
        raise ValueError("Unexpected visual checkpoint")
    manifest = read_json(resolve("data/versions/v002/manifest.json"))
    context, context_hash = load_context({"context": {"directory": cfg["context_directory"]}},
                                         REPO_ROOT, manifest["content_fingerprint"], sha256_file)
    train = context[context.split.eq("train")].sort_values("observation_uid")
    val = context[context.split.eq("val")].sort_values("observation_uid")
    classes = pd.read_csv(resolve("data/versions/v002/classes.csv")).sort_values("label_id").genus.tolist()
    visual = aligned_visual(run / "predictions_val.csv", val, classes)
    encoder = ContextEncoder.fit(train, FULL)

    ext_id = cfg["external_set"]
    ext_manifest = read_json(resolve(f"data/versions/{ext_id}/manifest.json"))
    obs_path = resolve(f"data/processed/{ext_id}/observations.csv")
    expected = next(o["sha256"] for o in ext_manifest["outputs"] if o["name"] == "observations.csv")
    archive = resolve(f"reports/external/{ext_id}/{run.name}")
    evaluation = read_json(archive / "evaluation.json")
    if (sha256_file(obs_path) != expected or evaluation["set_fingerprint"] != ext_manifest["content_fingerprint"]
            or evaluation["checkpoint_sha256"] != cfg["visual_checkpoint_sha256"]):
        raise ValueError("External provenance mismatch")
    ext_path = archive / "predictions_external.csv"
    ext_images = pd.read_csv(ext_path)
    ext = pd.read_csv(obs_path, low_memory=False)
    ext = ext[ext.observation_uid.isin(ext_images.observation_uid)].sort_values("observation_uid").copy()
    ext = ext.assign(**dict.fromkeys(BIO, np.nan), substrate_class="", substrate__quality="NOT_PROVIDED")
    ext_visual = aligned_visual(ext_path, ext, classes)

    metrics, external_metrics, rows, external_rows, deltas = {}, {}, [], [], []
    for name, modalities in VARIANTS.items():
        print(f"Evaluando {name}...", flush=True)
        conditions = model_scores(train, val, visual, encoder, modalities, cfg) if modalities else [("normal", visual, None)]
        for condition, probabilities, predict in conditions:
            scores = score_group(val, probabilities, classes)
            rows.append({"model": name, "condition": condition, **{k: scores[k] for k in ("n", "macro_f1", "accuracy")}})
            if condition == "normal":
                metrics[name] = scores
                external_probs = predict(ext, ext_visual) if predict else ext_visual
                external_metrics[name] = {}
                for group, mask in {"total": np.ones(len(ext), dtype=bool),
                                    "colombia": ext.source_id.isin(["inaturalist_co", "mushroom_observer_co"]),
                                    "orinoquia": ext.region.eq("orinoquia")}.items():
                    scores_ext = score_group(ext.loc[mask], external_probs[mask], classes)
                    external_metrics[name][group] = scores_ext
                    external_rows.append({"model": name, "group": group, **{k: scores_ext[k] for k in ("n", "macro_f1", "accuracy")}})
                if name != "A":
                    deltas.append({"model": name, **paired_delta(val.label_id.to_numpy(), visual.argmax(1),
                                   probabilities.argmax(1), cfg["bootstrap_resamples"], cfg["seed"])})
            if condition == "missing_all" and not np.array_equal(probabilities, visual):
                raise ValueError("Expected exact visual fallback")
    results = pd.DataFrame(rows)
    old = pd.read_csv(resolve(cfg["output"]) / "validation_summary.csv")
    overlap = old.merge(results, on=["model", "condition"], suffixes=("_s10", "_s11"), validate="one_to_one")
    if len(overlap) != len(old) or not np.allclose(overlap.macro_f1_s10, overlap.macro_f1_s11, atol=1e-12, rtol=0):
        raise ValueError("A–D no longer reproduce S9–S10")
    results.to_csv(destination / "validacion.csv", index=False)
    pd.DataFrame(external_rows).to_csv(destination / "externo.csv", index=False)
    pd.DataFrame([{"cohort": cohort, "model": name, **row}
                  for cohort, tables in [("validacion", metrics), ("externo", {m: s["total"] for m, s in external_metrics.items()})]
                  for name, scores in tables.items() for row in scores["per_class"]]).to_csv(
        destination / "metricas_por_genero.csv", index=False)
    confusion_plot(metrics, classes, destination / "confusion.png")
    confusion_plot({m: s["total"] for m, s in external_metrics.items()}, classes, destination / "confusion_externo.png")
    visual_cfg = read_json(run / "run.json")["config"]
    training = training_plot(run, visual_cfg, classes, destination / "entrenamiento.png")
    cams = gradcam(run, visual_cfg, ext_images, classes, destination)
    write_json(destination / "evidencia.json", {"variants": VARIANTS, "train_observations": len(train),
               "context_manifest_sha256": context_hash, "checkpoint_sha256": cfg["visual_checkpoint_sha256"],
               "external_predictions_sha256": sha256_file(ext_path), "training": training, "gradcam": cams,
               "validation_by_model": metrics, "external_by_model": external_metrics, "paired_vs_A": deltas})


if __name__ == "__main__":
    main()
