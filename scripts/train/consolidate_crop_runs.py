"""Consolidate completed runs, reusing inference ONLY after exact model-state verification.

No train/test images or GPU inference are used. Saved val predictions are checked against
the published cohort and the completed runs. Bootstrap and progress have durable checkpoints.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from progress_ui import bar, stage

with stage("Cargando consolidador de recortes", active=__name__ == "__main__"):
    import numpy as np
    import pandas as pd
    import torch
    from data import REPO_ROOT, resolve, sha256_file
    from metrics import aggregate_observations
    from pipeline.artifacts import ProgressFile, code_hashes
    from pipeline.crop_state import atomic_json, signature, state_digest
    from pipeline.evaluation import evaluate
    from pipeline.predictions import load_predictions
    from analyze_crop_experiments import paired_bootstrap


def verify_identical_checkpoints(original, resumed):
    """Archive hashes may change on serialization; every parameter AND buffer must be exact."""
    a = torch.load(original, map_location="cpu", weights_only=True)
    b = torch.load(resumed, map_location="cpu", weights_only=True)
    for name in ("classes", "dataset_fingerprint", "config", "epoch"):
        if name not in a or a[name] != b.get(name):
            raise ValueError(f"checkpoint metadata changed: {name}; rerun evaluate_crop_pair.py")
    if a["model"].keys() != b["model"].keys():
        raise ValueError("model tensors changed; rerun evaluate_crop_pair.py")
    for name, value in a["model"].items():
        other = b["model"][name]
        if (value.dtype != other.dtype or value.shape != other.shape
                or not torch.isfinite(value).all() or not torch.equal(value, other)):
            raise ValueError(f"model state changed: {name}; rerun evaluate_crop_pair.py")
    return {"tensors_identical": True, "tensors_checked": len(a["model"]),
            "includes_parameters_and_buffers": True, "best_epoch": int(a["epoch"]),
            "state_sha256": state_digest(a["model"]),
            "original_archive_sha256": sha256_file(original), "completed_archive_sha256": sha256_file(resumed),
            "classes": a["classes"], "dataset_fingerprint": a["dataset_fingerprint"]}


def validate_predictions(path, classes, expected):
    frame, probs = load_predictions(path, classes)
    order = np.argsort(frame.image_uid.to_numpy())
    frame, probs = frame.iloc[order].reset_index(drop=True), probs[order]
    pd.testing.assert_frame_equal(frame[expected.columns], expected, check_dtype=False)
    if not np.array_equal(frame.pred_label_id, probs.argmax(1)):
        raise ValueError(f"saved decisions differ from probabilities: {path}")
    return frame, probs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--control", required=True)
    ap.add_argument("--cropped", required=True)
    ap.add_argument("--reuse-evaluation", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    torch.set_num_threads(1)
    source, out = resolve(args.reuse_evaluation), resolve(args.output)
    runs = {"control": resolve(args.control), "cropped": resolve(args.cropped)}
    if out == source or any(out.is_relative_to(run) for run in runs.values()):
        ap.error("use a new report directory outside the source evaluation and training runs")
    out.mkdir(parents=True, exist_ok=True)
    progress = ProgressFile(out, out.name)
    progress("verifying_checkpoints")
    old = json.loads((source / "pair.json").read_text(encoding="utf-8"))
    if code_hashes() != old["code"]:
        raise ValueError("current pipeline differs from the cached evaluation; evaluate again")
    if not json.loads((source / "completed.json").read_text())["model_states_unchanged_during_evaluation"]:
        raise ValueError("source evaluation did not preserve model state")
    infos, histories, proofs, input_hashes, drifts = {}, {}, {}, {}, []

    def track(path):
        input_hashes[str(path.relative_to(REPO_ROOT))] = sha256_file(path)
        return path

    track(source / "pair.json"); track(source / "completed.json")
    track(Path(__file__)); track(Path(__file__).with_name("analyze_crop_experiments.py"))
    if sha256_file(Path(__file__).with_name("analyze_crop_experiments.py")) != old["analysis_script_sha256"]:
        raise ValueError("analysis helper differs from source evaluation")
    head_rows_path = track(source / "classifier_rows_per_genus.csv")
    with stage("Verificando pesos y finalizacion de ambas ramas"):
        for branch, run in bar(runs.items(), desc="Checkpoints completados", unit="rama"):
            info = json.loads(track(run / "run.json").read_text(encoding="utf-8"))
            done = json.loads(track(run / "completed.json").read_text())
            if done["evaluated_splits"] != ["val"] or done["test_evaluations"] != 0:
                raise ValueError("this consolidation requires completed val-only development runs")
            if info["config"] != old["configs"][branch] or info["code_sha256"] != old["code"]:
                raise ValueError("recipe or pipeline differs from the source evaluation; evaluate again")
            previous = resolve(old["runs"][branch]) / "best.pt"
            if sha256_file(previous) != old["checkpoints"][branch]:
                raise ValueError("original checkpoint archive changed since source evaluation")
            proofs[branch] = verify_identical_checkpoints(track(previous), track(run / "best.pt"))
            init_record = json.loads(track(run / "initialization.json").read_text())
            expected_init = old["initialization"][list(runs).index(branch)]
            if init_record["state_sha256"] != expected_init["state_sha256"] or init_record["sha256"] != expected_init["sha256"]:
                raise ValueError("initial reference differs from source evaluation")
            init = torch.load(track(run / "initial.pt"), map_location="cpu", weights_only=True)
            if state_digest(init["model"]) != init_record["state_sha256"]:
                raise ValueError("saved initial weights differ from the drift reference")
            del init
            history = pd.read_csv(track(run / "history.csv"))
            if history.epoch.tolist() != list(range(1, len(history)+1)):
                raise ValueError("training history is not contiguous")
            best_epoch = int(history.loc[history.val_obs_macro_f1.idxmax(), "epoch"])
            if best_epoch != proofs[branch]["best_epoch"]:
                raise ValueError("checkpoint is not the first best epoch recorded in history")
            histories[branch], infos[branch] = history, info
            drifts.append(pd.read_csv(track(run / "parameter_drift_summary.csv")).assign(branch=branch))
    classes = proofs["control"]["classes"]
    fingerprint = proofs["control"]["dataset_fingerprint"]
    if classes != old["classes"] or fingerprint != old["dataset"]:
        raise ValueError("source evaluation has a different class order or dataset")
    if any(p["classes"] != classes or p["dataset_fingerprint"] != fingerprint for p in proofs.values()):
        raise ValueError("paired checkpoint labels or dataset differ")
    version = infos["control"]["config"]["data"]["version"]
    manifest = json.loads(track(REPO_ROOT / f"data/versions/{version}/manifest.json").read_text())
    if manifest["content_fingerprint"] != fingerprint:
        raise ValueError("published dataset fingerprint changed")
    class_path = track(REPO_ROOT / f"data/versions/{version}/classes.csv")
    if sha256_file(class_path) != next(o["sha256"] for o in manifest["outputs"] if o["name"] == "classes.csv"):
        raise ValueError("published class table changed")
    if pd.read_csv(class_path).sort_values("label_id").genus.tolist() != classes:
        raise ValueError("published class order differs from the model")
    image_path = track(REPO_ROOT / f"data/processed/{version}/images.csv")
    if sha256_file(image_path) != next(o["sha256"] for o in manifest["outputs"] if o["name"] == "images.csv"):
        raise ValueError("published image table changed")
    images = pd.read_csv(image_path, low_memory=False)
    expected = images.loc[images.split.eq("val"), ["image_uid", "observation_uid", "label_id"]].sort_values("image_uid").reset_index(drop=True)
    jobs, rows, metrics, per_genus, evaluation_outputs = {}, [], {}, [], {}
    progress("checking_saved_predictions")
    for branch, run in runs.items():
        own = "original" if branch == "control" else "center75"
        current_frame, current_p = validate_predictions(track(run / "predictions_val.csv"), classes, expected)
        current_metrics = json.loads(track(run / "metrics_val.json").read_text())
        for policy in ["original", "center75"]:
            key = f"{branch}_{policy}"
            frame, probs = validate_predictions(track(source / key / "predictions_val.csv"), classes, expected)
            metric = evaluate(frame, probs, classes, [1, 3])
            saved = json.loads(track(source / key / "metrics_val.json").read_text())
            for level in ["image_level", "observation_level"]:
                if metric[level]["confusion_matrix"] != saved[level]["confusion_matrix"]:
                    raise ValueError("saved probabilities no longer reproduce the evaluated checkpoint")
                if policy == own and metric[level]["confusion_matrix"] != current_metrics[level]["confusion_matrix"]:
                    raise ValueError("completed run decisions differ from cached evaluation")
            if policy == own and not np.allclose(probs, current_p, atol=2e-6, rtol=0):
                raise ValueError("completed run probabilities differ materially from source evaluation")
            evaluation_outputs[key] = metric
            y, po = aggregate_observations(frame.image_uid, frame.observation_uid, frame.label_id.to_numpy(), probs)
            jobs[key] = po.argmax(1); metrics[key] = metric
            history, config = histories[branch], infos[branch]["config"]
            last_epoch = int(history.epoch.iloc[-1]); best_epoch = proofs[branch]["best_epoch"]
            stopped_early = last_epoch < config["training"]["epochs"] and last_epoch-best_epoch >= config["training"]["early_stopping_patience"]
            rows.append(dict(branch=branch, inference_policy=policy, epochs_executed=last_epoch, best_epoch=best_epoch,
                             max_epochs=config["training"]["epochs"], early_stopping=stopped_early,
                             macro_f1_observation=metric["observation_level"]["macro_f1"],
                             macro_f1_image=metric["image_level"]["macro_f1"], accuracy_observation=metric["observation_level"]["accuracy"]))
            per_genus.extend({"branch": branch, "inference_policy": policy, **row} for row in metric["observation_level"]["per_class"])
    sig = signature({"inputs": input_hashes, "script": sha256_file(Path(__file__))})
    if (out / "reuse_audit.json").exists() and json.loads((out / "reuse_audit.json").read_text())["signature"] != sig:
        raise ValueError("consolidation inputs changed; choose a new output directory")
    atomic_json(out / "reuse_audit.json", {"signature": sig, "checkpoint_proofs": proofs, "source_sha256": input_hashes,
                "source_evaluation": str(source.relative_to(REPO_ROOT)), "inference_reexecuted": False,
                "saved_predictions_reproduce_latest_own_policy": True, "test_evaluations": 0})
    for key, metric in bar(evaluation_outputs.items(), desc="Consolidando resultados val", unit="condicion"):
        target = out / key
        target.mkdir(exist_ok=True)
        shutil.copyfile(source / key / "predictions_val.csv", target / "predictions_val.csv")
        atomic_json(target / "metrics_val.json", metric)
    progress("bootstrap")
    scores = paired_bootstrap(y, jobs, len(classes), 10000, 42, out / "bootstrap_state.npz", sig)
    comparisons = []
    for policy in ["original", "center75"]:
        delta = scores[f"cropped_{policy}"]-scores[f"control_{policy}"]
        low, high = np.quantile(delta, [.025, .975]).tolist()
        point = metrics[f"cropped_{policy}"]["observation_level"]["macro_f1"]-metrics[f"control_{policy}"]["observation_level"]["macro_f1"]
        comparisons.append(dict(inference_policy=policy, delta_cropped_minus_control=point, ci95_low=low, ci95_high=high, contains_zero=bool(low <= 0 <= high)))
    pd.DataFrame(rows).to_csv(out / "validation_2x2.csv", index=False)
    pd.DataFrame(comparisons).to_csv(out / "paired_training_effect.csv", index=False)
    pd.DataFrame(per_genus).to_csv(out / "per_genus.csv", index=False)
    drift = pd.concat(drifts, ignore_index=True)
    drift.to_csv(out / "parameter_drift_summary.csv", index=False)
    common = drifts[0].merge(drifts[1], on=["epoch", "batches_seen", "group"], suffixes=("_control", "_cropped"))
    common["same_optimizer_steps"] = common.optimizer_steps_control.eq(common.optimizer_steps_cropped)
    common.to_csv(out / "drift_at_common_batches.csv", index=False)
    pd.concat([h.assign(branch=b) for b, h in histories.items()]).to_csv(out / "training_history.csv", index=False)
    shutil.copyfile(head_rows_path, out / "classifier_rows_per_genus_best.csv")
    progress("figures")
    plot_histories(out, histories, metrics, classes)
    subprocess.run([sys.executable, "-u", str(Path(__file__).with_name("plot_crop_weights.py")), "--study", str(out)], check=True)
    selected = max(rows, key=lambda row: row["macro_f1_observation"])
    selection = {"status": "validation_candidate", "campaign": "cropping_mobilenet_v3_large_v002",
                 "run": str(runs[selected["branch"]].relative_to(REPO_ROOT)), "checkpoint": "best.pt", **selected,
                 "checkpoint_sha256": proofs[selected["branch"]]["completed_archive_sha256"],
                 "dataset_fingerprint": fingerprint, "selection_metric": "val observation macro-F1",
                 "test_used_for_selection": False, "additional_seeds_completed": False}
    atomic_json(out / "selection.json", selection)
    write_report(out, rows, comparisons)
    atomic_json(out / "completed.json", {"signature": sig, "training_runs_completed": True,
                "best_checkpoint_states_identical_to_pilot": True, "inference_reexecuted": False, "test_evaluations": 0})
    progress("completed")
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
    print(f"Informe: {out / 'RESULTS.md'}", flush=True)


def plot_histories(out, histories, metrics, classes):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    with stage("Generando curvas y matrices de validacion"):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for branch, history in histories.items():
            axes[0].plot(history.epoch, history.train_loss, marker="o", label=branch)
            axes[1].plot(history.epoch, history.val_obs_macro_f1, marker="o", label=branch)
        for ax in axes:
            ax.set_xticks(sorted(set().union(*(set(h.epoch) for h in histories.values()))))
            ax.set_xlabel("Época"); ax.grid(alpha=.2); ax.legend()
        axes[0].set(title="Pérdida de entrenamiento", ylabel="Entropía cruzada ponderada")
        axes[1].set(title="Validación con encuadre propio", ylabel="F1 macro por observación")
        fig.tight_layout(); fig.savefig(figures / "training_curves.png", dpi=160); plt.close(fig)
        fig, axes = plt.subplots(2, 2, figsize=(13, 11))
        fig.suptitle("Color: proporción por género real · Cifras: observaciones")
        for ax, (key, metric) in zip(axes.ravel(), metrics.items()):
            cm = np.array(metric["observation_level"]["confusion_matrix"])
            normalized = cm/np.maximum(1, cm.sum(1, keepdims=True))
            ax.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
            ax.set_xticks(range(len(classes)), classes, rotation=90); ax.set_yticks(range(len(classes)), classes)
            ax.set(title=key, xlabel="Predicción", ylabel="Real")
            for i, j in zip(*np.where(cm)):
                ax.text(j, i, str(cm[i,j]), ha="center", va="center", fontsize=6, color="white" if normalized[i,j]>.5 else "black")
        fig.tight_layout(rect=(0, 0, 1, .96)); fig.savefig(figures / "confusion_observations.png", dpi=150); plt.close(fig)


def write_report(out, rows, comparisons):
    lines = ["# Comparación completada: control y recortes", "",
             "Finalización de las ramas:", ""]
    for row in rows:
        if row["inference_policy"] == ("original" if row["branch"] == "control" else "center75"):
            reason = "parada temprana" if row["early_stopping"] else "límite configurado"
            lines.append(f"- {row['branch']}: {row['epochs_executed']} épocas ejecutadas de un máximo de {row['max_epochs']}; mejor época {row['best_epoch']}; finalizó por {reason}.")
    lines += ["", "Cuando se activa la parada temprana, no es necesario completar el límite de épocas de la corrida.", "",
              "| Entrenamiento | Inferencia | Épocas ejecutadas | Mejor época | F1 macro obs. | F1 macro foto |", "|---|---|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['branch']} | {row['inference_policy']} | {row['epochs_executed']} | {row['best_epoch']} | {row['macro_f1_observation']:.4f} | {row['macro_f1_image']:.4f} |")
    lines += ["", "| Inferencia común | Δ F1 recortado − control | IC 95 % pareado |", "|---|---:|---|"]
    for row in comparisons:
        lines.append(f"| {row['inference_policy']} | {row['delta_cropped_minus_control']:+.4f} | [{row['ci95_low']:+.4f}, {row['ci95_high']:+.4f}] |")
    lines += ["", "Se conserva el procesamiento original y se registra la continuación de control como candidato de esta comparación por su mayor F1 macro observado. No hay ventaja concluyente por entrenar con recortes cuando ambos modelos reciben el recorte central del 75 %.",
              "El modelo adaptado pierde rendimiento con el encuadre original. Estas conclusiones se limitan a esta receta y una semilla.", "",
              "La inferencia del piloto se reutilizó tras comparar exactamente todos los parámetros y buffers de los mejores checkpoints. Sus archivos .pt tienen hashes diferentes por serialización, pero su estado del modelo es idéntico. Las predicciones propias de las corridas completadas reproducen las matrices guardadas.",
              "Los intervalos se recalculan desde probabilidades guardadas: bootstrap pareado por observación, 10 000 remuestreos, semilla 42, sin corrección múltiple ni ajuste por sitio-día o variación entre semillas.", "",
              "Las curvas de pesos incluyen el historial completo. La tabla de filas de la cabeza por género corresponde al mejor checkpoint seleccionado, no a los pesos de la última época.",
              "La pérdida de train disminuye mientras el F1 de val cae respecto al mejor punto. Aumentar épocas con esta receta no aportó una mejora observada.",
              "Las fuentes, identificadores, clases y particiones oficiales de v002 se mantienen. No se ejecutó inferencia nueva ni test. La selección histórica de S5 permanece intacta.", "",
              "Figuras: [Entrenamiento](figures/training_curves.png) · [Cambios de pesos](figures/weight_drift.png) · [Matrices](figures/confusion_observations.png).",
              "Auditoría y trazabilidad: `reuse_audit.json`. Candidato de validación: `selection.json`."]
    (out / "RESULTS.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


if __name__ == "__main__": main()
