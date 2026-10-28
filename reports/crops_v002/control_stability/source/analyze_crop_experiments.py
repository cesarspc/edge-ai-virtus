"""Analyze saved crop probabilities only; never opens test or performs model inference."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from metrics import aggregate_observations
from pipeline.crop_state import atomic_json, atomic_npz, signature
from pipeline.crops import POLICIES
from progress_ui import bar, stage


def paired_bootstrap(y, predictions, n_classes, samples, seed, path, source_signature, stop_after=None):
    """Paired ordinary observation bootstrap, fixed label universe, resumable every 250 draws."""
    keys = list(predictions)
    if samples < 1 or not len(y) or any(len(p) != len(y) for p in predictions.values()):
        raise ValueError("positive sample count and aligned nonempty observations required")
    sig = signature([keys, n_classes, samples, seed, source_signature])
    rng = np.random.default_rng(seed)
    scores = np.zeros((samples, len(keys)))
    done = 0
    if path.exists():
        with np.load(path, allow_pickle=False) as cache:
            if str(cache["signature"]) != sig:
                raise ValueError("bootstrap inputs changed; use a new output directory")
            scores[:] = cache["scores"]
            done = int(cache["done"])
            rng.bit_generator.state = json.loads(str(cache["rng_state"]))
    target = samples if stop_after is None else min(samples, done + stop_after)
    with bar(total=samples, initial=done, desc="Bootstrap pareado por observacion", unit="muestra") as progress:
        while done < target:
            end = min(done + 250, target)
            idx = rng.integers(0, len(y), size=(end-done, len(y)))
            for col, key in enumerate(keys):
                codes = (y*n_classes + predictions[key])[idx]
                offsets = np.arange(end-done)[:, None] * n_classes*n_classes
                cm = np.bincount((codes+offsets).ravel(), minlength=(end-done)*n_classes*n_classes).reshape(-1, n_classes, n_classes)
                tp = np.diagonal(cm, axis1=1, axis2=2)
                denom = cm.sum(1) + cm.sum(2)
                f1 = np.divide(2*tp, denom, out=np.zeros_like(tp, dtype=float), where=denom > 0)
                scores[done:end, col] = f1.mean(1)
            done = end
            atomic_npz(path, signature=sig, done=done, scores=scores, rng_state=json.dumps(rng.bit_generator.state))
            progress.update(len(idx))
    return {key: scores[:done, col] for col, key in enumerate(keys)}


def analyze(folder):
    from pipeline.evaluation import evaluate, save_predictions
    from pipeline.crop_state import validate_probs

    manifest = json.loads((folder / "study.json").read_text(encoding="utf-8"))
    frozen = json.loads((folder / "weights_unchanged.json").read_text(encoding="utf-8"))
    control_check = json.loads((folder / "control_reproduction.json").read_text(encoding="utf-8"))
    if not frozen.get("identical") or not control_check.get("identical_image_decisions") or not control_check.get("confusion_matrices_match_saved_C"):
        raise ValueError("analysis requires unchanged model state and a reproduced reference control")
    classes = manifest["classes"]
    frame = pd.read_csv(folder / "validation_images.csv")
    with np.load(folder / "probabilities_val.npz", allow_pickle=False) as archive:
        probs = archive["probs"]
        timings, peaks = archive["seconds"], archive["peak_bytes"]
    validate_probs(probs)
    if probs.shape != (len(frame), 7, len(classes)):
        raise ValueError("unexpected final probability shape")
    observation_ids = sorted(frame.observation_uid.unique())
    metrics, predictions, obs_frames = {}, {}, {}
    rows, per_genus, changes = [], [], []
    for policy, views in POLICIES.items():
        target = folder / policy
        target.mkdir(exist_ok=True)
        p = probs[:, views].mean(1)
        m = evaluate(frame, p, classes, [1, 3])
        metrics[policy] = m
        atomic_json(target / "metrics_val.json", m)
        save_predictions(target / "predictions_val.csv", frame, p, classes)
        y, po = aggregate_observations(frame.image_uid, frame.observation_uid, frame.label_id.to_numpy(), p)
        predictions[policy] = po.argmax(1)
        obs = pd.DataFrame({"observation_uid": observation_ids, "label_id": y, "pred_label_id": predictions[policy]})
        for i, name in enumerate(classes):
            obs[f"p_{name}"] = po[:, i]
        obs.to_csv(target / "predictions_observation_val.csv", index=False)
        obs_frames[policy] = obs
        rows.append({"policy": policy, "images": len(frame), "observations": len(y),
                     "macro_f1_observation": m["observation_level"]["macro_f1"],
                     "accuracy_observation": m["observation_level"]["accuracy"],
                     "macro_f1_image": m["image_level"]["macro_f1"], "views_per_photo": len(views),
                     "forward_ms_per_photo": float(timings[:, views].sum()*1000/len(frame)),
                     "peak_cuda_allocated_mib": float(peaks[:, views].max()/2**20)})
        for entry in m["observation_level"]["per_class"]:
            per_genus.append({"policy": policy, **entry})
    control = predictions["control"]
    for policy in list(POLICIES)[1:]:
        pred = predictions[policy]
        for i in np.flatnonzero(pred != control):
            changes.append({"policy": policy, "observation_uid": observation_ids[i], "true_genus": classes[y[i]],
                            "control_genus": classes[control[i]], "crop_genus": classes[pred[i]],
                            "gained_correct": bool(control[i] != y[i] and pred[i] == y[i]),
                            "lost_correct": bool(control[i] == y[i] and pred[i] != y[i])})
    pd.DataFrame(changes, columns=["policy", "observation_uid", "true_genus", "control_genus", "crop_genus", "gained_correct", "lost_correct"]).to_csv(folder / "changed_observations.csv", index=False)
    pd.DataFrame(per_genus).to_csv(folder / "per_genus.csv", index=False)
    boot = manifest["config"]["bootstrap"]
    source_sig = hashlib.sha256((folder / "probabilities_val.npz").read_bytes()).hexdigest()
    scores = paired_bootstrap(y, predictions, len(classes), boot["resamples"], boot["seed"], folder / "bootstrap_state.npz", source_sig)
    paired = []
    for row in rows:
        policy = row["policy"]
        row["ci95_low"], row["ci95_high"] = np.quantile(scores[policy], [.025, .975]).tolist()
        if policy == "control":
            continue
        lo, hi = np.quantile(scores[policy] - scores["control"], [.025, .975]).tolist()
        pred = predictions[policy]
        paired.append({"policy": policy, "reference": "control", "delta_macro_f1": row["macro_f1_observation"]-rows[0]["macro_f1_observation"],
                       "ci95_low": lo, "ci95_high": hi, "contains_zero": bool(lo <= 0 <= hi),
                       "gained_correct": int(((control != y) & (pred == y)).sum()),
                       "lost_correct": int(((control == y) & (pred != y)).sum()),
                       "n_observations": len(y), "bootstrap_samples": boot["resamples"], "seed": boot["seed"]})
    pd.DataFrame(rows).to_csv(folder / "summary.csv", index=False)
    pd.DataFrame(paired).to_csv(folder / "paired_comparisons.csv", index=False)
    # An inferentially conservative gate: prefer single-view policies if none clearly beats control.
    useful = [p for p in paired if p["delta_macro_f1"] > 0 and p["ci95_low"] > 0]
    winner = max(rows, key=lambda r: r["macro_f1_observation"])["policy"]
    decision = {"winner_point_estimate": winner, "clear_improvement_candidates": [p["policy"] for p in useful],
                "recommended_inference_policy": max(useful, key=lambda p: p["delta_macro_f1"])["policy"] if useful else "control",
                "test_used": False, "weights_changed": False,
                "inference_gate": "positive macro-F1 delta with paired percentile CI entirely above zero",
                "training_conclusion": "Frozen sensitivity does not measure the effect of fine tuning.",
                "limitations": ["exploratory comparisons without multiplicity correction", "one frozen checkpoint",
                                "bootstrap does not account for site-day dependence or training seed variation",
                                "forward timing excludes decoding/crop transforms and is not an Android measurement"]}
    atomic_json(folder / "decision.json", decision)
    plot_results(folder, classes, metrics, rows, paired)
    lines = ["# Sensibilidad a recortes: MobileNetV3-Large v002", "",
             "Validación oficial únicamente. Mismo checkpoint C congelado en todas las variantes.", "",
             "| Política | F1 macro observación | F1 macro foto | Exactitud obs. | Vistas/foto |",
             "|---|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['policy']} | {r['macro_f1_observation']:.4f} | {r['macro_f1_image']:.4f} | {r['accuracy_observation']:.4f} | {r['views_per_photo']} |")
    lines += ["", "| Recorte frente a control | Δ F1 macro obs. | IC 95 % pareado | Aciertos ganados | Aciertos perdidos |",
              "|---|---:|---|---:|---:|"]
    for r in paired:
        lines.append(f"| {r['policy']} | {r['delta_macro_f1']:+.4f} | [{r['ci95_low']:+.4f}, {r['ci95_high']:+.4f}] | {r['gained_correct']} | {r['lost_correct']} |")
    lines += ["", f"Procesamiento recomendado en esta etapa: **{decision['recommended_inference_policy']}**.", "",
              "Los recortes cambian las predicciones. Los parámetros y los buffers de BatchNorm se verificaron sin cambios.",
              "Las cinco regiones se promedian por foto; luego las fotos se promedian por observación. No se crean muestras independientes.",
              "El porcentaje de área corresponde al recorte adicional previo a Resize y CenterCrop.", "",
              f"Bootstrap pareado por observación: {boot['resamples']} remuestreos, semilla {boot['seed']}, IC percentil; comparaciones exploratorias sin corrección múltiple.",
              "El tiempo de forward excluye lectura y transformaciones; no representa latencia en Android.",
              "Estos resultados no prueban el efecto de reentrenar. Un ajuste debe contrastarse con una continuación de control desde el mismo C.",
              "No se ejecutó test ni se reemplazó la selección histórica de S5.", "",
              "Figuras: [F1 e intervalos](figures/validation_f1.png) · [Matrices de confusión](figures/confusion_observations.png).",
              "Galería de train: `gallery_train/gallery_train.png` (fotos locales, excluidas de Git)."]
    (folder / "RESULTS.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return rows, decision


def plot_results(folder, classes, metrics, rows, paired):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    target = folder / "figures"
    target.mkdir(exist_ok=True)
    with stage("Generando figuras de resultados"):
        fig, ax = plt.subplots(figsize=(7, 4))
        vals = np.array([r["macro_f1_observation"] for r in rows])
        low, high = np.array([r["ci95_low"] for r in rows]), np.array([r["ci95_high"] for r in rows])
        # Percentile CIs need not contain the point estimate: draw independent intervals.
        ax.vlines(range(len(rows)), low, high, color="steelblue")
        ax.scatter(range(len(rows)), vals, color="navy")
        ax.set_xticks(range(len(rows)), [r["policy"] for r in rows])
        ax.set(ylabel="F1 macro por observación", title="C congelado: validación e IC 95 %")
        ax.grid(axis="y", alpha=.2)
        fig.tight_layout(); fig.savefig(target / "validation_f1.png", dpi=160); plt.close(fig)
        fig, axes = plt.subplots(2, 2, figsize=(13, 11))
        fig.suptitle("Color: proporción por género real · Cifras: número de observaciones", fontsize=12)
        for ax, (policy, m) in zip(axes.ravel(), metrics.items()):
            cm = np.asarray(m["observation_level"]["confusion_matrix"])
            display = cm / np.maximum(1, cm.sum(1, keepdims=True))
            ax.imshow(display, vmin=0, vmax=1, cmap="Blues")
            ax.set_xticks(range(len(classes)), classes, rotation=90)
            ax.set_yticks(range(len(classes)), classes)
            ax.set(title=policy, xlabel="Predicción", ylabel="Real")
            for i in range(len(classes)):
                for j in range(len(classes)):
                    if cm[i, j]:
                        ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=6, color="white" if display[i,j]>.5 else "black")
        fig.tight_layout(rect=(0, 0, 1, .97)); fig.savefig(target / "confusion_observations.png", dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--study", type=Path, required=True)
    args = ap.parse_args()
    with stage("Analizando probabilidades guardadas"):
        rows, _ = analyze(args.study.resolve())
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
