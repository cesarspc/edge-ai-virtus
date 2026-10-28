"""Analyze saved validation predictions for the prespecified control seeds 42/43/44.

No image inference or test access. The selected seed 42 stays fixed; no seed cherry-picking
or probability ensembling. Observation bootstrap checkpoints resume automatically.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from progress_ui import stage, bar

with stage("Cargando analisis de estabilidad", active=__name__ == "__main__"):
    import numpy as np
    import pandas as pd
    from data import REPO_ROOT, resolve, sha256_file
    from metrics import aggregate_observations, classification_metrics
    from pipeline.artifacts import ProgressFile, code_hashes
    from pipeline.crop_state import atomic_json, signature
    from pipeline.predictions import load_predictions
    from analyze_crop_experiments import paired_bootstrap
    from run_control_stability import recipe_without_seed, read_json


def summarize(values):
    """Between-seed sample SD is descriptive; it is not a confidence interval."""
    x = np.asarray(values, dtype=float)
    if x.shape != (3,) or not np.isfinite(x).all():
        raise ValueError("three finite seed scores are required")
    return {"n_seeds": 3, "mean": float(x.mean()), "sample_sd": float(x.std(ddof=1)),
            "min": float(x.min()), "max": float(x.max()), "range": float(x.max()-x.min())}


def fixed_candidate(rows):
    """Seed 42 was selected before replications; a higher-scoring repeat cannot replace it."""
    reference = [row for row in rows if row["seed"] == 42]
    if len(rows) != 3 or len(reference) != 1 or {row["seed"] for row in rows} != {42, 43, 44}:
        raise ValueError("exactly seeds 42,43,44 required")
    return dict(reference[0])


def checked_observations(path, classes, expected, saved_metrics):
    frame, probs = load_predictions(path, classes)
    order = np.argsort(frame.image_uid.to_numpy())
    frame, probs = frame.iloc[order].reset_index(drop=True), probs[order]
    pd.testing.assert_frame_equal(frame[expected.columns], expected, check_dtype=False)
    if not np.array_equal(frame.pred_label_id.to_numpy(), probs.argmax(1)):
        raise ValueError("saved decisions differ from probability argmax")
    y, po = aggregate_observations(frame.image_uid, frame.observation_uid, frame.label_id.to_numpy(), probs)
    obs = classification_metrics(y, po, classes, [1, 3])
    img = classification_metrics(frame.label_id.to_numpy(), probs, classes, [1, 3])
    for name, metric in [("observation_level", obs), ("image_level", img)]:
        if metric["confusion_matrix"] != saved_metrics[name]["confusion_matrix"]:
            raise ValueError("CSV does not reproduce saved validation confusion matrix")
    return frame, y, po, {"observation_level": obs, "image_level": img}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--campaign", default="reports/crops_v002/control_stability")
    args = ap.parse_args()
    campaign = resolve(args.campaign)
    state, frozen = read_json(campaign / "campaign.json"), read_json(campaign / "protocol.json")
    if not state.get("training_complete") or any(state["jobs"].get(str(s), {}).get("status") != "completed" for s in [42, 43, 44]):
        raise SystemExit("Finish all three seeds with run_control_stability.py before analysis.")
    if state["signature"] != frozen["signature"]:
        raise ValueError("campaign state differs from the frozen protocol")
    if code_hashes() != frozen["code_sha256"] or sha256_file(Path(__file__).with_name("run_control_stability.py")) != frozen["runner_sha256"]:
        raise ValueError("pipeline or campaign runner changed since the protocol was frozen")
    out = campaign / "results"
    out.mkdir(exist_ok=True)
    progress = ProgressFile(out, out.name)
    progress("verifying_saved_results")
    inputs = {}

    def track(path):
        inputs[str(path.relative_to(REPO_ROOT))] = sha256_file(path)
        return path

    track(campaign / "campaign.json"); track(campaign / "protocol.json"); track(Path(__file__))
    track(Path(__file__).with_name("run_control_stability.py"))
    base = frozen["configs"][0]
    version = base["data"]["version"]
    manifest = read_json(track(REPO_ROOT / f"data/versions/{version}/manifest.json"))
    if manifest["content_fingerprint"] != frozen["dataset_fingerprint"]:
        raise ValueError("published dataset fingerprint differs from training")
    for name, path in [("images.csv", REPO_ROOT / f"data/processed/{version}/images.csv"),
                       ("classes.csv", REPO_ROOT / f"data/versions/{version}/classes.csv")]:
        if sha256_file(track(path)) != next(o["sha256"] for o in manifest["outputs"] if o["name"] == name):
            raise ValueError("published cohort or class table changed")
    classes = pd.read_csv(REPO_ROOT / f"data/versions/{version}/classes.csv").sort_values("label_id").genus.tolist()
    images = pd.read_csv(REPO_ROOT / f"data/processed/{version}/images.csv", low_memory=False)
    expected = images.loc[images.split.eq("val"), ["image_uid", "observation_uid", "label_id"]].sort_values("image_uid").reset_index(drop=True)
    reference_C = resolve(base["initialization"]["checkpoint"]).parent
    if sha256_file(track(reference_C / "best.pt")) != frozen["protocol"]["initial_checkpoint_sha256"]:
        raise ValueError("common initial C checkpoint changed")
    old_info = read_json(track(reference_C / "run.json"))
    if old_info["dataset"]["content_fingerprint"] != manifest["content_fingerprint"]:
        raise ValueError("baseline C has another dataset fingerprint")
    _, y, po, baseline = checked_observations(track(reference_C / "predictions_val.csv"), classes, expected,
                                              read_json(track(reference_C / "metrics_val.json")))
    predictions = {"C": po.argmax(1)}
    metrics = {"C": baseline}
    rows, per_genus, histories, drift_frames, origins, balances, environments = [], [], [], [], [], [], {}
    for seed in bar([42, 43, 44], desc="Verificando resultados por semilla", unit="semilla"):
        run = resolve(state["jobs"][str(seed)]["run"])
        info, done = read_json(track(run / "run.json")), read_json(track(run / "completed.json"))
        if (info["mode"] != "train" or info["config"]["training"]["seed"] != seed
                or recipe_without_seed(info["config"]) != recipe_without_seed(base)
                or info["code_sha256"] != frozen["code_sha256"]
                or info["dataset"]["content_fingerprint"] != manifest["content_fingerprint"]):
            raise ValueError("seed training recipe, pipeline or dataset differs from fixed protocol")
        if info["device"] != "cuda" or done["evaluated_splits"] != ["val"] or done["test_evaluations"] != 0:
            raise ValueError("seed must be a completed CUDA val-only run")
        environments[str(seed)] = {"versions": info["versions"], "gpu": info.get("gpu"), "amp": info["amp"]}
        origins.append(read_json(track(run / "initialization.json")))
        balances.append(read_json(track(run / "class_balance.json")))
        _, current_y, current_po, m = checked_observations(track(run / "predictions_val.csv"), classes, expected,
                                                          read_json(track(run / "metrics_val.json")))
        np.testing.assert_array_equal(y, current_y)
        predictions[str(seed)], metrics[str(seed)] = current_po.argmax(1), m
        history = pd.read_csv(track(run / "history.csv"))
        if history.epoch.tolist() != list(range(1, len(history)+1)):
            raise ValueError("history contains missing/duplicated epochs")
        best_epoch = int(history.loc[history.val_obs_macro_f1.idxmax(), "epoch"])
        best_score = float(history.loc[history.epoch.eq(best_epoch), "val_obs_macro_f1"].item())
        if abs(best_score-m["observation_level"]["macro_f1"]) > 1e-12:
            raise ValueError("best validation score is not reproduced by final saved predictions")
        last = int(history.epoch.iloc[-1])
        rows.append({"seed": seed, "run": str(run.relative_to(REPO_ROOT)), "epochs_executed": last,
                     "best_epoch": best_epoch, "early_stopping": bool(last < base["training"]["epochs"] and last-best_epoch >= base["training"]["early_stopping_patience"]),
                     "macro_f1_observation": m["observation_level"]["macro_f1"], "macro_f1_image": m["image_level"]["macro_f1"],
                     "accuracy_observation": m["observation_level"]["accuracy"], "checkpoint_sha256": sha256_file(track(run / "best.pt")),
                     "gpu": info.get("gpu"), "amp": info["amp"]})
        per_genus.extend({"seed": seed, **c} for c in m["observation_level"]["per_class"])
        histories.append(history.assign(seed=seed))
        drift_frames.append(pd.read_csv(track(run / "parameter_drift_summary.csv")).assign(seed=seed))
    if not all(origin == origins[0] for origin in origins) or not all(balance == balances[0] for balance in balances):
        raise ValueError("seed initial weights or class-loss weights differ")
    if any(env != environments["42"] for env in environments.values()):
        raise ValueError("library versions, GPU or precision differ between seeds")
    if origins[0]["sha256"] != frozen["protocol"]["initial_checkpoint_sha256"]:
        raise ValueError("seed weights did not initialize from the recorded C checkpoint")
    track(Path(__file__).with_name("analyze_crop_experiments.py"))
    sig = signature(inputs)
    if (out / "analysis_manifest.json").exists() and read_json(out / "analysis_manifest.json")["signature"] != sig:
        raise ValueError("analysis inputs changed; preserve old results and select a new campaign")
    atomic_json(out / "analysis_manifest.json", {"signature": sig, "source_sha256": inputs,
                "test_evaluations": 0, "image_inference_executed": False, "classes": classes,
                "n_observations": len(y), "n_images": len(expected), "scope": frozen["protocol"]["scope"],
                "selection_policy": frozen["selection_policy"], "initializations_identical": True,
                "class_weights_identical": True, "environments": environments})
    progress("bootstrap")
    bootstrap = frozen["protocol"]["analysis"]
    scores = paired_bootstrap(y, predictions, len(classes), bootstrap["bootstrap_samples"], bootstrap["bootstrap_seed"], out / "bootstrap_state.npz", sig)
    for row in rows:
        row["ci95_low"], row["ci95_high"] = np.quantile(scores[str(row["seed"])], [.025, .975]).tolist()
    pairs = []
    for seed in [42, 43, 44]:
        for reference in (["C"] if seed == 42 else ["C", "42"]):
            low, high = np.quantile(scores[str(seed)]-scores[reference], [.025, .975]).tolist()
            pairs.append({"seed": seed, "reference": reference,
                          "delta_macro_f1": metrics[str(seed)]["observation_level"]["macro_f1"]-metrics[reference]["observation_level"]["macro_f1"],
                          "ci95_low": low, "ci95_high": high, "contains_zero": bool(low <= 0 <= high)})
    summary = {key: summarize([row[key] for row in rows]) for key in ["macro_f1_observation", "macro_f1_image", "accuracy_observation"]}
    summary.update(baseline_C=baseline["observation_level"], candidate_seed=42, selection_policy=frozen["selection_policy"], bootstrap=bootstrap,
                   all_seed_point_estimates_above_C=all(row["macro_f1_observation"] > baseline["observation_level"]["macro_f1"] for row in rows),
                   warning="Between-seed SD is not a confidence interval; three repeats share the same selected initial C checkpoint.")
    pd.DataFrame(rows).to_csv(out / "results_by_seed.csv", index=False)
    pd.DataFrame(pairs).to_csv(out / "paired_comparisons.csv", index=False)
    genus = pd.DataFrame(per_genus)
    genus.to_csv(out / "per_genus.csv", index=False)
    genus_summary = genus.groupby("genus").agg(mean_f1=("f1", "mean"), sample_sd_f1=("f1", "std"), min_f1=("f1", "min"), max_f1=("f1", "max"), support=("support", "first"))
    genus_summary.to_csv(out / "per_genus_stability.csv")
    summary["most_variable_genera"] = genus_summary.sort_values("sample_sd_f1", ascending=False).head(3).reset_index().to_dict("records")
    history, drift = pd.concat(histories), pd.concat(drift_frames)
    history.to_csv(out / "training_history.csv", index=False)
    drift.to_csv(out / "parameter_drift_summary.csv", index=False)
    selected_drifts = pd.concat([drift.loc[drift.seed.eq(row["seed"]) & drift.epoch.eq(row["best_epoch"])] for row in rows])
    selected_drifts.to_csv(out / "selected_checkpoint_drift.csv", index=False)
    common_epochs = set.intersection(*(set(f.epoch) for f in drift_frames))
    common = drift.loc[drift.epoch.isin(common_epochs)].copy()
    common["same_optimizer_steps"] = common.groupby(["epoch", "batches_seen", "group"]).optimizer_steps.transform("nunique").eq(1)
    common.sort_values(["epoch", "group", "seed"]).to_csv(out / "drift_at_common_batches.csv", index=False)
    obs = pd.DataFrame({"observation_uid": sorted(expected.observation_uid.unique()), "label_id": y, "genus": np.array(classes)[y], "pred_C": predictions["C"]})
    for seed in [42, 43, 44]: obs[f"pred_{seed}"] = predictions[str(seed)]
    seed_preds = np.stack([predictions[str(s)] for s in [42, 43, 44]])
    obs["wrong_seed_count"] = (seed_preds != y).sum(0)
    obs["seed_disagreement"] = seed_preds.max(0) != seed_preds.min(0)
    obs.to_csv(out / "observation_errors.csv", index=False)
    error_summary = {"all_three_wrong": int(obs.wrong_seed_count.eq(3).sum()), "any_seed_wrong": int(obs.wrong_seed_count.gt(0).sum()),
                     "prediction_disagreement": int(obs.seed_disagreement.sum()), "n_observations": len(y)}
    atomic_json(out / "error_overlap.json", error_summary)
    atomic_json(out / "summary.json", summary)
    candidate = fixed_candidate(rows)
    if candidate["checkpoint_sha256"] != frozen["reference_best_sha256"]:
        raise ValueError("preselected candidate changed during replications")
    atomic_json(out / "selection.json", {"status": "validation_candidate_with_seed_study", **candidate,
                "inference_policy": "original", "checkpoint": "best.pt", "reference_checkpoint_sha256": frozen["reference_best_sha256"],
                "candidate_preserved_from_before_repeats": True, "seeds_completed": [42, 43, 44],
                "test_used_for_selection": False, "seed_score_averaging_is_not_an_ensemble": True})
    progress("figures")
    plot_results(out, rows, pairs, history, drift, baseline)
    write_report(out, rows, pairs, summary, error_summary)
    atomic_json(out / "completed.json", {"signature": sig, "seeds_completed": [42, 43, 44],
                "candidate_seed_unchanged": 42, "test_evaluations": 0, "image_inference_executed": False})
    progress("completed", seeds_completed=3)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
    print(f"Informe: {out / 'RESULTS.md'}", flush=True)


def plot_results(out, rows, pairs, history, drift, baseline):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    with stage("Generando figuras de estabilidad"):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for i, row in enumerate(rows):
            axes[0].vlines(i, row["ci95_low"], row["ci95_high"], color="steelblue")
            axes[0].scatter(i, row["macro_f1_observation"], color="navy")
            p = next(p for p in pairs if p["seed"] == row["seed"] and p["reference"] == "C")
            axes[1].vlines(i, p["ci95_low"], p["ci95_high"], color="steelblue")
            axes[1].scatter(i, p["delta_macro_f1"], color="navy")
        axes[0].axhline(baseline["observation_level"]["macro_f1"], linestyle="--", color="gray", label="C inicial")
        axes[1].axhline(0, color="gray", linestyle="--")
        for ax in axes: ax.set_xticks(range(3), ["42 (candidato)", "43", "44"]); ax.set_xlabel("Semilla del ajuste"); ax.grid(axis="y", alpha=.2)
        axes[0].set(title="Validación: F1 e IC 95 % por observación", ylabel="F1 macro observación"); axes[0].legend()
        axes[1].set(title="Diferencia frente a C: IC 95 % pareado", ylabel="Δ F1 macro")
        fig.tight_layout(); fig.savefig(figures / "validation_stability.png", dpi=160); plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for seed, f in history.groupby("seed"):
            axes[0].plot(f.epoch, f.train_loss, marker="o", label=str(seed))
            axes[1].plot(f.epoch, f.val_obs_macro_f1, marker="o", label=str(seed))
        for ax in axes: ax.set_xticks(sorted(history.epoch.unique())); ax.set_xlabel("Época"); ax.legend(title="Semilla"); ax.grid(alpha=.2)
        axes[0].set(title="Pérdida de train", ylabel="Entropía cruzada ponderada")
        axes[1].set(title="Validación: encuadre original", ylabel="F1 macro observación")
        fig.tight_layout(); fig.savefig(figures / "training_curves.png", dpi=160); plt.close(fig)
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, group, title in zip(axes, ["backbone", "classifier", "bn_running_statistics"], ["Extractor", "Cabeza", "BatchNorm (buffers)"]):
            for seed, f in drift.loc[drift.group.eq(group)].groupby("seed"):
                f = f.sort_values("epoch")
                ax.plot([0, *f.epoch], [0, *(100*f.relative_delta_l2)], marker="o", label=str(seed))
            ax.set_xticks(sorted({0, *history.epoch.astype(int)})); ax.set(title=title, xlabel="Época", ylabel="Distancia L2 desde C (%)", ylim=(0, None)); ax.legend(title="Semilla")
        fig.tight_layout(); fig.savefig(figures / "weight_trajectories.png", dpi=160); plt.close(fig)


def write_report(out, rows, pairs, summary, errors):
    lines = ["# Estabilidad del control: semillas 42, 43 y 44", "",
             "Las tres repeticiones ajustan el mismo checkpoint C con la misma receta y encuadre original. Solo cambia la semilla. Train y val conservan las particiones oficiales de v002.", "",
             "| Semilla | Épocas ejecutadas | Mejor época | F1 macro obs. | F1 macro foto | Exactitud obs. |", "|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['seed']} | {r['epochs_executed']} | {r['best_epoch']} | {r['macro_f1_observation']:.4f} | {r['macro_f1_image']:.4f} | {r['accuracy_observation']:.4f} |")
    s = summary["macro_f1_observation"]
    lines += ["", f"F1 macro por observación: media **{s['mean']:.4f}**, desviación estándar muestral **{s['sample_sd']:.4f}**, rango **{s['min']:.4f}–{s['max']:.4f}** (tres semillas). La desviación estándar no es un intervalo de confianza.",
              f"Referencia C antes del ajuste: **{summary['baseline_C']['macro_f1']:.4f}** en val."]
    below = [str(r["seed"]) for r in rows if r["macro_f1_observation"] <= summary["baseline_C"]["macro_f1"]]
    if below:
        lines += [f"Las mejores puntuaciones de las semillas {', '.join(below)} no superan el punto de referencia C. La mejora frente a C observada en la semilla 42 no apareció en las tres repeticiones; debe comunicarse junto con la dispersión entre semillas."]
    else:
        lines += ["Las tres puntuaciones observadas superan el punto de C en val. Los intervalos y las limitaciones de selección siguen siendo necesarios antes de interpretar esto como una mejora generalizable."]
    lines += ["", "| Semilla | Referencia | Δ F1 macro | IC 95 % pareado por observación |", "|---|---|---:|---|"]
    for p in pairs: lines.append(f"| {p['seed']} | {p['reference']} | {p['delta_macro_f1']:+.4f} | [{p['ci95_low']:+.4f}, {p['ci95_high']:+.4f}] |")
    lines += ["", "| Género con mayor variación | Soporte de val | F1 mínimo | F1 máximo | DE entre semillas |", "|---|---:|---:|---:|---:|"]
    for g in summary["most_variable_genera"]:
        lines.append(f"| {g['genus']} | {int(g['support'])} | {g['min_f1']:.4f} | {g['max_f1']:.4f} | {g['sample_sd_f1']:.4f} |")
    lines += ["", "La semilla 42 se mantiene como candidato fijado antes de ejecutar las repeticiones. No se elige la semilla con mayor puntuación ni se combinan las probabilidades para crear un ensemble.",
              "Cada fila utiliza su mejor checkpoint seleccionado con val. Las diferencias frente a C son comparaciones de validación, no resultados de test ni estimaciones independientes de la selección previa.",
              f"Errores: {errors['all_three_wrong']} observaciones fallan en las tres semillas; {errors['prediction_disagreement']} reciben predicciones distintas entre semillas. Véanse `observation_errors.csv` y `per_genus_stability.csv`.",
              f"Los intervalos usan bootstrap pareado por observación, {summary['bootstrap']['bootstrap_samples']} remuestreos, semilla {summary['bootstrap']['bootstrap_seed']}, sin corrección múltiple. No representan la incertidumbre total por semillas o sitios-día.",
              "El estudio describe variación del ajuste posterior condicionada a una inicialización C ya seleccionada. Con tres semillas no se afirma estabilidad universal ni se mide la variación de entrenar desde ImageNet con otras inicializaciones.",
              "Las distancias de pesos de los checkpoints seleccionados se guardan por separado de las trayectorias completas. Sus épocas pueden ser distintas; `drift_at_common_batches.csv` permite comparar épocas/lotes comunes e indica si coinciden los pasos reales del optimizador. Una distancia mayor no implica mejor clasificación.",
              "No se ejecutó inferencia nueva: se verificaron y analizaron los archivos finales de val ya guardados. No se consultó test ni se sustituyó la selección histórica de S5.", "",
              "Figuras: [Validación](figures/validation_stability.png) · [Entrenamiento](figures/training_curves.png) · [Pesos](figures/weight_trajectories.png).",
              "Trazabilidad: `analysis_manifest.json`. Candidato preservado: `selection.json`."]
    (out / "RESULTS.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


if __name__ == "__main__": main()
