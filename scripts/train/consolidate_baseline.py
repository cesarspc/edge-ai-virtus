"""Archive and describe a completed run using saved predictions/metrics only.

No model is instantiated, trained or used for inference. Loading best.pt reads
checkpoint metadata so the selected epoch and dataset can be verified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def preserve(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and sha(source) != sha(target):
        raise SystemExit(f"Refusing to replace archived content: {target}")
    if not target.exists():
        shutil.copy2(source, target)
    if sha(source) != sha(target):
        raise SystemExit(f"Archive copy differs: {target}")


def savefig(fig, path):
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def matrix_plot(metrics, level, labels, normalized, destination, best_epoch):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.7), layout="constrained")
    for ax, split in zip(axes, ("val", "test")):
        counts = np.asarray(metrics[split][level]["confusion_matrix"])
        values = counts / counts.sum(axis=1, keepdims=True) * 100 if normalized else counts
        image = ax.imshow(values, cmap="Blues", vmin=0, vmax=100 if normalized else counts.max())
        for y, x in np.ndindex(counts.shape):
            content = f"{values[y,x]:.1f}%" if normalized else str(counts[y,x])
            ax.text(x, y, content, ha="center", va="center", fontsize=7.5,
                    color="white" if values[y,x] > image.norm.vmax * .55 else "#1e293b")
        ax.set_xticks(range(10), labels, rotation=55, ha="right", fontsize=9)
        ax.set_yticks(range(10), labels, fontsize=9)
        ax.set_xlabel("Género predicho")
        ax.set_ylabel("Género real")
        ax.set_title(f"{split.upper()} · {'observación' if level == 'observation_level' else 'imagen'}")
        fig.colorbar(image, ax=ax, shrink=.7, label="Porcentaje por género real (%)" if normalized else "Conteo")
    fig.suptitle(f"ResNet-18 · checkpoint de la época {best_epoch}", fontsize=14)
    savefig(fig, destination)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="runs/20261004T133123181686Z_ma_resnet18_v002")
    args = ap.parse_args()
    run = (ROOT / args.run).resolve()
    info = json.loads((run / "run.json").read_text(encoding="utf-8"))
    completed = json.loads((run / "completed.json").read_text(encoding="utf-8"))
    if completed["test_evaluations"] != 1 or completed["evaluated_splits"] != ["val", "test"]:
        raise SystemExit("A completed run with exactly one final test evaluation is required.")
    sources = {p.name: sha(p) for p in run.iterdir() if p.is_file()}
    hist = pd.read_csv(run / "history.csv")
    checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    best_epoch = int(hist.loc[hist.val_obs_macro_f1.idxmax(), "epoch"])
    if checkpoint["epoch"] != best_epoch or checkpoint["dataset_fingerprint"] != info["dataset"]["content_fingerprint"]:
        raise SystemExit("Checkpoint metadata differs from the run/history.")
    if checkpoint["config"] != info["config"]:
        raise SystemExit("Checkpoint configuration differs from the saved run.")
    classes = checkpoint["classes"]
    metrics = {s: json.loads((run / f"metrics_{s}.json").read_text(encoding="utf-8")) for s in ("val", "test")}
    if not np.isclose(metrics["val"]["observation_level"]["macro_f1"], hist.val_obs_macro_f1.max(), atol=1e-12, rtol=0):
        raise SystemExit("Final val metrics do not match the selected epoch.")
    out, figures = ROOT / "reports/baseline_v002", ROOT / "reports/figures"
    archive = ROOT / "artifacts/baselines/resnet18_v002" / run.name
    out.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    archive_names = ["best.pt", "run.json", "history.csv", "metrics_val.json", "metrics_test.json",
                     "predictions_val.csv", "predictions_test.csv", "completed.json", "test_started.json", "RESULTS.md"]
    for name in archive_names:
        preserve(run / name, archive / name)
    effective = archive / "training_config_effective.yaml"
    text = yaml.safe_dump(info["config"], sort_keys=False, allow_unicode=True)
    if effective.exists() and effective.read_text(encoding="utf-8") != text:
        raise SystemExit("Archived effective config differs.")
    effective.write_text(text, encoding="utf-8")
    dataset_config = ROOT / "configs/dataset_v002.yaml"
    manifest_path = ROOT / "data/versions/v002/manifest.json"
    dataset_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sha(dataset_config) != dataset_manifest["config"]["sha256"]:
        raise SystemExit("Dataset config differs from the published manifest.")
    preserve(dataset_config, archive / "dataset_config.yaml")
    preserve(manifest_path, archive / "dataset_manifest.json")

    rows, overall, confusions = [], [], []
    for split, levels in metrics.items():
        for level, m in levels.items():
            overall.append(dict(split=split, level=level, **{k:v for k,v in m.items() if k not in ("per_class", "confusion_matrix")}))
            cm = np.asarray(m["confusion_matrix"])
            if int(cm.sum()) != m["n"]:
                raise SystemExit("Confusion matrix total differs from saved metrics.")
            for c in m["per_class"]:
                i = c["label_id"]
                rows.append(dict(split=split, level=level, **c, true_positives=int(cm[i,i]),
                                 false_negatives=int(cm[i,:].sum()-cm[i,i]),
                                 false_positives=int(cm[:,i].sum()-cm[i,i])))
            for i,j in np.ndindex(cm.shape):
                if i != j and cm[i,j]:
                    confusions.append(dict(split=split, level=level, true_genus=classes[i],
                                           predicted_genus=classes[j], count=int(cm[i,j]),
                                           true_support=int(cm[i].sum()), fraction_of_true_class=float(cm[i,j]/cm[i].sum())))
    table = pd.DataFrame(rows)
    table.to_csv(out / "metrics_by_genus.csv", index=False, lineterminator="\n")
    pd.DataFrame(overall).to_csv(out / "metrics_summary.csv", index=False, lineterminator="\n")
    pd.DataFrame(confusions).sort_values(["split", "level", "count"], ascending=[True,True,False]).to_csv(out / "confusion_pairs.csv", index=False, lineterminator="\n")

    # Detailed error inspection uses VAL ONLY. Exported probabilities have six decimal places.
    predictions = pd.read_csv(run / "predictions_val.csv")
    probabilities = ["p_" + c for c in classes]
    predictions["true_genus"] = predictions.label_id.map(dict(enumerate(classes)))
    predictions["predicted_genus"] = predictions.pred_label_id.map(dict(enumerate(classes)))
    predictions["predicted_probability"] = predictions[probabilities].to_numpy().max(axis=1)
    image_errors = predictions[predictions.label_id != predictions.pred_label_id].sort_values("predicted_probability", ascending=False)
    image_errors.to_csv(out / "validation_image_errors.csv", index=False, lineterminator="\n")
    if predictions.groupby("observation_uid").label_id.nunique().gt(1).any():
        raise SystemExit("Conflicting labels in saved val predictions.")
    grouped = predictions.groupby("observation_uid", sort=True)
    observations = grouped[probabilities].mean()
    observations["label_id"] = grouped.label_id.first()
    observations["pred_label_id"] = observations[probabilities].to_numpy().argmax(axis=1)
    observations["n_images"] = grouped.size()
    observations["image_uids"] = grouped.image_uid.agg(";".join)
    observations["true_genus"] = observations.label_id.map(dict(enumerate(classes)))
    observations["predicted_genus"] = observations.pred_label_id.map(dict(enumerate(classes)))
    observations["predicted_probability"] = observations[probabilities].max(axis=1)
    cm = np.zeros((10,10), dtype=int)
    np.add.at(cm, (observations.label_id.to_numpy(), observations.pred_label_id.to_numpy()), 1)
    if not np.array_equal(cm, metrics["val"]["observation_level"]["confusion_matrix"]):
        raise SystemExit("Rounded saved probabilities do not reproduce observation predictions.")
    obs_errors = observations[observations.label_id != observations.pred_label_id].sort_values("predicted_probability", ascending=False)
    obs_errors.to_csv(out / "validation_observation_errors.csv", lineterminator="\n")
    obs = metrics["val"]["observation_level"]
    minor = []
    for c in obs["per_class"]:
        if c["support"] < 100:
            n, r, z = c["support"], c["recall"], 1.96
            center = (r+z*z/(2*n))/(1+z*z/n)
            half = z*np.sqrt(r*(1-r)/n+z*z/(4*n*n))/(1+z*z/n)
            minor.append(dict(genus=c["genus"], support=n, recall=r, f1=c["f1"],
                              recall_wilson95_lower=max(0,center-half), recall_wilson95_upper=min(1,center+half)))
    pd.DataFrame(minor).to_csv(out / "small_class_support.csv", index=False, lineterminator="\n")

    plt.rcParams.update({"font.size":10, "axes.spines.top":False, "axes.spines.right":False})
    fig, axes = plt.subplots(1,3,figsize=(14,4.4),layout="constrained")
    axes[0].plot(hist.epoch,hist.train_loss,"o-",color="#2563eb",label="Pérdida train")
    axes[0].set_ylabel("Entropía cruzada (sin unidad)")
    axes[1].plot(hist.epoch,hist.val_obs_macro_f1,"o-",color="#0f766e",label="Macro-F1 observación val")
    axes[1].plot(hist.epoch,hist.val_img_macro_f1,"s-",color="#2563eb",label="Macro-F1 imagen val")
    axes[1].set_ylabel("Macro-F1 (escala 0–1)")
    axes[1].legend(fontsize=8,loc="lower right")
    axes[2].plot(hist.epoch,hist.lr,"o-",color="#7c3aed")
    axes[2].set_ylabel("Tasa de aprendizaje")
    for ax in axes:
        ax.axvline(best_epoch,color="#dc2626",linestyle="--",linewidth=1)
        ax.axvspan(best_epoch+.5,hist.epoch.max()+.5,color="#fee2e2",alpha=.5)
        ax.set_xlabel("Época")
        ax.set_xticks([1,4,8,11,16])
        ax.grid(alpha=.2)
    fig.suptitle(f"ResNet-18 · mejor época {best_epoch} · early stopping {int(hist.epoch.max())}")
    savefig(fig,figures / "baseline_v002_training_curves.png")
    for level, short in (("image_level","image"),("observation_level","observation")):
        for normalized in (False,True):
            matrix_plot(metrics,level,classes,normalized,figures / f"baseline_v002_confusion_{short}_{'percent' if normalized else 'counts'}.png",best_epoch)
    fig, axes = plt.subplots(1,2,figsize=(13,5.5),layout="constrained",sharey=True)
    positions = np.arange(10)
    for ax, level, title in zip(axes,("image_level","observation_level"),("Por imagen","Por observación")):
        for offset,split,color in ((-.18,"val","#2563eb"),(.18,"test","#0f766e")):
            values = table[(table.split==split)&(table.level==level)].sort_values("label_id")
            bars=ax.barh(positions+offset,values.f1,height=.34,label=split.upper(),color=color)
            for bar,value,n in zip(bars,values.f1,values.support):
                ax.text(.02,bar.get_y()+bar.get_height()/2,f"F1={value:.3f} · n={n}",va="center",fontsize=8,color="white")
        ax.set_yticks(positions,classes)
        ax.set_xlim(0,1.04)
        ax.set_xlabel("F1 (escala 0–1); n = soporte de la clase")
        ax.set_title(title,loc="left",pad=15)
        ax.legend(loc="lower right",bbox_to_anchor=(1,1.01),ncol=2)
    axes[0].invert_yaxis()
    savefig(fig,figures / "baseline_v002_f1_by_genus.png")

    # Local-only gallery: source photographs stay excluded from Git.
    selected = image_errors.groupby("true_genus",sort=False).head(1).head(10)
    fig, axes = plt.subplots(4,3,figsize=(11,13),layout="constrained")
    from PIL import Image
    root = ROOT / info["config"]["data"]["image_root"] / "300p"
    for ax, row in zip(axes.flat,selected.itertuples()):
        with Image.open(root / row.image_uid.split(":",1)[1]) as image:
            ax.imshow(image.convert("RGB"))
        ax.set_title(f"Real: {row.true_genus}\nPredicho: {row.predicted_genus} ({row.predicted_probability:.3f})",fontsize=9)
        ax.axis("off")
    for ax in list(axes.flat)[len(selected):]:
        ax.axis("off")
    fig.suptitle("Errores de validación · una imagen por género · probabilidades guardadas")
    savefig(fig,out / "validation_examples_local_only.png")

    overview = dict(run_id=run.name,epochs_completed=int(hist.epoch.max()),best_epoch=best_epoch,
                    best_val_obs_macro_f1=float(hist.val_obs_macro_f1.max()),
                    test_evaluations=completed["test_evaluations"],
                    validation_image_errors=len(image_errors),validation_observation_errors=len(obs_errors),
                    checkpoint_sha256=sha(run/"best.pt"),dataset_fingerprint=info["dataset"]["content_fingerprint"],
                    total_recorded_epoch_seconds=float(hist.seconds.sum()),
                    mean_recorded_epoch_seconds=float(hist.seconds.mean()),
                    plots_use_saved_metrics=True,model_inference_performed=False,
                    observation_errors_reconstructed_from_rounded_saved_probabilities=True)
    (out/"summary.json").write_text(json.dumps(overview,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    archive_manifest = dict(**overview,source_run=str(run.relative_to(ROOT)),
                            files={p.name:sha(p) for p in archive.iterdir() if p.is_file() and p.name!="archive_manifest.json"})
    (archive/"archive_manifest.json").write_text(json.dumps(archive_manifest,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    (out/"preservation_manifest.json").write_text(json.dumps(archive_manifest,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    consolidation = dict(source_run=str(run.relative_to(ROOT)), source_sha256=sources,
                         model_inference_performed=False, test_evaluations_added=0,
                         generator_sha256=sha(Path(__file__)),
                         output_sha256={str(p.relative_to(ROOT)):sha(p) for p in
                                        list(out.glob("*.csv"))+list(figures.glob("baseline_v002_*.png"))})
    (out/"consolidation_manifest.json").write_text(json.dumps(consolidation,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    if sources != {p.name:sha(p) for p in run.iterdir() if p.is_file()}:
        raise SystemExit("Source run changed during consolidation.")
    print(json.dumps(overview,indent=2))
    print(f"Archive: {archive}\nAnalysis: {out}\nFigures: {figures}")


if __name__ == "__main__":
    main()
