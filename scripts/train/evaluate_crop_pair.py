"""Resumable 2x2 validation of two crop pilot checkpoints. Never reads test photographs.

python -u scripts/train/evaluate_crop_pair.py --control <run> --cropped <run> --output <report>
Repeat the same command to resume complete inference blocks and the paired bootstrap.
"""
from __future__ import annotations

import argparse
import copy
import json
from importlib.metadata import version as package_version
from pathlib import Path

from progress_ui import stage, bar, ready_batches

with stage("Cargando evaluador del piloto", active=__name__ == "__main__"):
    import numpy as np
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader
    from data import REPO_ROOT, resolve, load_tables, sha256_file
    from metrics import aggregate_observations
    from pipeline.artifacts import ProgressFile, code_hashes, environment
    from pipeline.crop_state import signature, state_digest, atomic_json, atomic_npz, load_blocks
    from pipeline.datasets import ImageTable
    from pipeline.evaluation import predict_batch, evaluate, save_predictions
    from pipeline.models import build_model
    from pipeline.transforms import build_transforms
    from analyze_crop_experiments import paired_bootstrap


def infer(model, frame, transform, folder, sig, device, progress, description):
    folder.mkdir(exist_ok=True)
    chunks = folder / "chunks"
    chunks.mkdir(exist_ok=True)
    blocks, _, _, done = load_blocks(chunks, sig, len(frame), 64, 1, 10)
    if done < len(frame):
        loader = DataLoader(ImageTable(frame.iloc[done:], transform), batch_size=64, num_workers=2,
                            prefetch_factor=1, pin_memory=device.type == "cuda")
        with bar(total=len(frame), initial=done, desc=description, unit="foto") as pbar:
            for x, _ in ready_batches(loader, f"Preparando primer lote: {description}"):
                p = predict_batch(model, x, device, device.type == "cuda")
                end = done+len(p)
                atomic_npz(chunks / f"block_{done:06d}.npz", signature=sig, start=done, end=end,
                           probs=p[:, None, :], seconds=np.zeros(1), peak_bytes=np.zeros(1))
                blocks.append(p[:, None, :]); done = end
                pbar.update(len(p)); progress(description, photos_completed=done, photos_total=len(frame))
    return np.concatenate(blocks)[:, 0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--control", required=True)
    ap.add_argument("--cropped", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    output = resolve(args.output)
    output.mkdir(parents=True, exist_ok=True)
    progress = ProgressFile(output, output.name)
    runs = {"control": resolve(args.control), "cropped": resolve(args.cropped)}
    infos = {key: json.loads((run / "run.json").read_text(encoding="utf-8")) for key, run in runs.items()}
    cfg = infos["control"]["config"]
    cfg_crop = infos["cropped"]["config"]
    if (cfg["input"].get("extra_crop") or not cfg_crop["input"].get("extra_crop")
            or cfg["training"] != cfg_crop["training"] or cfg["data"] != cfg_crop["data"]
            or cfg["model"] != cfg_crop["model"]):
        raise ValueError("paired runs require identical training/data/architecture and original vs extra-crop inputs")
    input_without_crop = copy.deepcopy(cfg_crop["input"])
    input_without_crop.pop("extra_crop")
    if cfg["input"] != input_without_crop:
        raise ValueError("only extra_crop may differ between the two inputs")
    origins = [json.loads((run / "initialization.json").read_text(encoding="utf-8")) for run in runs.values()]
    if origins[0]["state_sha256"] != origins[1]["state_sha256"] or origins[0]["sha256"] != origins[1]["sha256"]:
        raise ValueError("paired branches must start from identical model weights")
    progress("verifying_validation")
    tables = load_tables(cfg, show_progress=True, splits=("val",))
    frame = tables.frames["val"]
    if len(tables.missing):
        raise ValueError("all val photos are required")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    identity = {"checkpoints": {key: sha256_file(run / "best.pt") for key, run in runs.items()},
                "dataset": tables.fingerprint, "classes": tables.classes,
                "frame_order": signature(frame[["image_uid", "observation_uid", "label_id"]].to_dict("list")),
                "configs": {key: info["config"] for key, info in infos.items()}, "initialization": origins,
                "code": code_hashes(), "script_sha256": sha256_file(Path(__file__)), "environment": environment(device),
                "image_libraries": {name: package_version(name) for name in ["Pillow", "numpy", "pandas"]},
                "analysis_script_sha256": sha256_file(Path(__file__).with_name("analyze_crop_experiments.py"))}
    sig = signature(identity)
    manifest_path = output / "pair.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8"))["signature"] != sig:
        raise ValueError("paired evaluation inputs changed; choose another output directory")
    atomic_json(manifest_path, {**identity, "signature": sig, "test_evaluations": 0,
                               "runs": {key: str(path.relative_to(REPO_ROOT)) for key, path in runs.items()}})
    transforms = {"original": build_transforms(cfg)[1], "center75": build_transforms(cfg_crop)[1]}
    rows, jobs, drifts, histories, head_rows = [], {}, [], [], []
    for branch, run in runs.items():
        checkpoint = torch.load(run / "best.pt", weights_only=True, map_location="cpu")
        if checkpoint["classes"] != tables.classes or checkpoint["dataset_fingerprint"] != tables.fingerprint:
            raise ValueError("pilot checkpoint does not match published dataset")
        initial = torch.load(run / "initial.pt", map_location="cpu", weights_only=True)["model"]
        if state_digest(initial) != origins[0]["state_sha256"]:
            raise ValueError("saved initial weights differ from the paired reference")
        head_key = [name for name, tensor in checkpoint["model"].items()
                    if name.startswith(("classifier.", "fc.")) and name.endswith("weight")
                    and tensor.ndim == 2 and tensor.shape[0] == len(tables.classes)][-1]
        for label, genus in enumerate(tables.classes):
            a, b = initial[head_key][label].double(), checkpoint["model"][head_key][label].double()
            n0, delta = float(torch.linalg.vector_norm(a)), float(torch.linalg.vector_norm(b-a))
            head_rows.append(dict(branch=branch, best_epoch=checkpoint["epoch"], genus=genus, tensor=head_key,
                                  initial_norm_l2=n0, absolute_delta_l2=delta, relative_delta_l2=delta/(n0+1e-12)))
        del initial
        model = build_model(infos[branch]["config"], pretrained=False).to(device)
        model.load_state_dict(checkpoint["model"], strict=True)
        model.requires_grad_(False).eval()
        before = state_digest(model.state_dict())
        for policy, transform in transforms.items():
            key = f"{branch}_{policy}"
            folder = output / key
            probs = infer(model, frame, transform, folder, signature([sig, key]), device, progress, key)
            metrics = evaluate(frame, probs, tables.classes, [1, 3])
            atomic_json(folder / "metrics_val.json", metrics)
            save_predictions(folder / "predictions_val.csv", frame, probs, tables.classes)
            y, po = aggregate_observations(frame.image_uid, frame.observation_uid, frame.label_id.to_numpy(), probs)
            jobs[key] = po.argmax(1)
            rows.append({"branch": branch, "inference_policy": policy, "best_epoch": checkpoint["epoch"],
                         "macro_f1_observation": metrics["observation_level"]["macro_f1"],
                         "macro_f1_image": metrics["image_level"]["macro_f1"],
                         "accuracy_observation": metrics["observation_level"]["accuracy"]})
        if state_digest(model.state_dict()) != before:
            raise ValueError("evaluation changed pilot weights/buffers")
        del model, checkpoint
        drift = pd.read_csv(run / "parameter_drift_summary.csv").assign(branch=branch)
        drifts.append(drift)
        histories.append(pd.read_csv(run / "history.csv").assign(branch=branch))
    scores = paired_bootstrap(y, jobs, len(tables.classes), 10000, 42, output / "bootstrap_state.npz", sig)
    comparisons = []
    for policy in transforms:
        delta = scores[f"cropped_{policy}"]-scores[f"control_{policy}"]
        point = next(r["macro_f1_observation"] for r in rows if r["branch"] == "cropped" and r["inference_policy"] == policy) - next(r["macro_f1_observation"] for r in rows if r["branch"] == "control" and r["inference_policy"] == policy)
        low, high = np.quantile(delta, [.025, .975]).tolist()
        comparisons.append(dict(inference_policy=policy, delta_cropped_minus_control=point, ci95_low=low, ci95_high=high, contains_zero=bool(low <= 0 <= high)))
    pd.DataFrame(rows).to_csv(output / "validation_2x2.csv", index=False)
    pd.DataFrame(head_rows).to_csv(output / "classifier_rows_per_genus.csv", index=False)
    pd.DataFrame(comparisons).to_csv(output / "paired_training_effect.csv", index=False)
    drift = pd.concat(drifts)
    drift.to_csv(output / "parameter_drift_summary.csv", index=False)
    pd.concat(histories).to_csv(output / "training_history.csv", index=False)
    common = drifts[0].merge(drifts[1], on=["epoch", "batches_seen", "group"], suffixes=("_control", "_cropped"))
    common["same_optimizer_steps"] = common.optimizer_steps_control.eq(common.optimizer_steps_cropped)
    common.to_csv(output / "drift_at_common_batches.csv", index=False)
    plot_drift(output, drift)
    lines = ["# Piloto pareado de ajuste con recortes", "", "Ambas ramas parten exactamente de C. Optimizador e historial reiniciados. Solo validación.", "",
             "| Entrenamiento | Inferencia | Mejor época | F1 macro obs. | F1 macro foto |", "|---|---|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['branch']} | {r['inference_policy']} | {r['best_epoch']} | {r['macro_f1_observation']:.4f} | {r['macro_f1_image']:.4f} |")
    lines += ["", "Comparar filas con la misma inferencia permite estudiar el cambio de entrenamiento. Comparar columnas de un mismo modelo mide el efecto del encuadre.",
              "", "| Inferencia común | Δ F1 cropped − control | IC 95 % pareado |", "|---|---:|---|"]
    for r in comparisons:
        lines.append(f"| {r['inference_policy']} | {r['delta_cropped_minus_control']:+.4f} | [{r['ci95_low']:+.4f}, {r['ci95_high']:+.4f}] |")
    lines += ["", "Los CSV de cambios de pesos separan backbone, cabeza y buffers de BatchNorm; indican épocas, lotes y actualizaciones reales de AdamW.",
              "Una mayor distancia de pesos no implica mejor clasificación. La comparación a lotes comunes distingue también posibles pasos omitidos por AMP.",
              "Este es un piloto de una semilla desde una inicialización ya seleccionada; una época comprueba el mecanismo, no estima el potencial final del ajuste.",
              "Los intervalos son bootstrap pareado por observación, 10 000 remuestreos, semilla 42, exploratorios sin corrección múltiple.",
              "Las ramas conservan last.pt y pueden continuar con su configuración original. No se ha consultado test ni sustituido C."]
    (output / "RESULTS.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    atomic_json(output / "completed.json", {"test_evaluations": 0, "initializations_identical": True, "model_states_unchanged_during_evaluation": True, "signature": sig})
    progress("completed")
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
    print(f"Resultados: {output / 'RESULTS.md'}", flush=True)


def plot_drift(output, drift):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, group in zip(axes, ["backbone", "classifier"]):
        for branch in ["control", "cropped"]:
            f = drift[drift.group.eq(group) & drift.branch.eq(branch)]
            ax.plot(f.epoch, f.relative_delta_l2, marker="o", label=branch)
        ax.set(title=group, xlabel="Época", ylabel="Cambio L2 relativo desde C")
        ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(figures / "weight_drift.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    main()
