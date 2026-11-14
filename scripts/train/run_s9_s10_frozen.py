"""Train contextual branches on train only, reuse frozen S5-C visual scores.

python scripts/train/run_s9_s10_frozen.py --external
This is a late-fusion experiment, not joint neural-network fine-tuning.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import yaml
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from data import REPO_ROOT, resolve, sha256_file
from metrics import classification_metrics
from pipeline.context import BIO, ContextEncoder, broad_substrate, geo_valid, load_context, perturb_context


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def paired_delta(y, reference, candidate, repeats, seed):
    def f1(labels, predictions):
        n = 10  # The protocol fixes ten classes, including classes absent from a subgroup.
        cm = np.bincount(labels * n + predictions, minlength=n * n).reshape(n, n)
        denominator = cm.sum(0) + cm.sum(1)
        return np.divide(2 * cm.diagonal(), denominator, out=np.zeros(n), where=denominator > 0).mean()
    rng = np.random.default_rng(seed)
    deltas, accuracy = [], []
    for _ in range(repeats):
        idx = rng.integers(len(y), size=len(y))
        deltas.append(f1(y[idx], candidate[idx]) - f1(y[idx], reference[idx]))
        accuracy.append(np.mean(candidate[idx] == y[idx]) - np.mean(reference[idx] == y[idx]))
    return {"delta_macro_f1": float(f1(y, candidate) - f1(y, reference)),
            "ci95_delta_macro_f1": np.quantile(deltas, [.025, .975]).tolist(),
            "delta_accuracy": float(np.mean(candidate == y) - np.mean(reference == y)),
            "ci95_delta_accuracy": np.quantile(accuracy, [.025, .975]).tolist(),
            "fixed_errors": int(((candidate == y) & (reference != y)).sum()),
            "introduced_errors": int(((candidate != y) & (reference == y)).sum())}


def aligned_visual(predictions_path, observations, classes):
    predictions = pd.read_csv(predictions_path)
    columns = [f"p_{c}" for c in classes]
    if predictions.image_uid.duplicated().any() or predictions.groupby("observation_uid").label_id.nunique().gt(1).any():
        raise ValueError("invalid visual prediction identifiers or labels")
    probabilities = predictions[columns].to_numpy(float)
    if (not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any()
            or not np.allclose(probabilities.sum(1), 1, atol=2e-5)):
        raise ValueError("invalid frozen probabilities")
    if set(predictions.observation_uid) != set(observations.observation_uid):
        raise ValueError("frozen predictions and observations differ")
    grouped = predictions.groupby("observation_uid", sort=True)
    result = grouped[columns].mean()
    result["label_id"] = grouped.label_id.first()
    result = result.reindex(observations.observation_uid)
    if not np.array_equal(result.label_id.to_numpy(), observations.label_id.to_numpy()):
        raise ValueError("frozen prediction labels disagree with cohort")
    values = result[columns].to_numpy()
    return values / values.sum(1, keepdims=True)


def context_available(frame, encoder):
    available = geo_valid(frame).to_numpy(copy=True)
    if encoder.feature_set != "geo":
        available |= np.isfinite(frame[list(BIO)].to_numpy(float)).any(axis=1)
    if encoder.substrate_categories is not None:
        available |= broad_substrate(frame).isin(encoder.substrate_categories).to_numpy()
    return available


def mix_scores(visual, contextual, available, weight):
    if not 0 <= weight <= 1:
        raise ValueError("context_weight must be a probability")
    alpha = weight * np.asarray(available, dtype=float)[:, None]
    return (1 - alpha) * visual + alpha * contextual


def context_prediction(model, frame):
    encoder, scaler, classifier = model
    return classifier.predict_proba(scaler.transform(encoder.transform(frame)))


def score_group(frame, probabilities, classes):
    scores = classification_metrics(frame.label_id.to_numpy(), probabilities, classes, [1, 3])
    present = np.unique(frame.label_id)
    scores["macro_f1_present_classes"] = float(np.mean([scores["per_class"][int(i)]["f1"] for i in present]))
    scores["classes_present"] = len(present)
    return scores


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/experiments/s9_s10_frozen_context.yaml")
    ap.add_argument("--external", action="store_true")
    args = ap.parse_args()
    config_path = resolve(args.config)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output = resolve(cfg["output"])
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(resolve("data/versions/v002/manifest.json").read_text(encoding="utf-8"))
    context_cfg = {"context": {"directory": cfg["context_directory"]}}
    context, context_hash = load_context(context_cfg, REPO_ROOT, manifest["content_fingerprint"], sha256_file)
    train = context[context.split.eq("train")].sort_values("observation_uid").reset_index(drop=True)
    val = context[context.split.eq("val")].sort_values("observation_uid").reset_index(drop=True)
    classes = pd.read_csv(resolve("data/versions/v002/classes.csv")).sort_values("label_id").genus.tolist()
    run = resolve(cfg["visual_run"])
    if sha256_file(run / "best.pt") != cfg["visual_checkpoint_sha256"]:
        raise ValueError("visual checkpoint differs from frozen protocol")
    visual_path = run / "predictions_val.csv"
    visual = aligned_visual(visual_path, val, classes)
    # Write the fixed protocol before fitting; no parameter search follows.
    write_json(output / "protocol.json", {**cfg, "config_sha256": sha256_file(config_path),
        "context_manifest_sha256": context_hash, "visual_predictions_sha256": sha256_file(visual_path),
        "script_sha256": sha256_file(Path(__file__)), "encoder_sha256": sha256_file(Path(__file__).parent / "pipeline/context.py"),
        "training_observations": len(train), "validation_observations": len(val),
        "evaluation_unit": "observation", "image_probabilities_recomputed": False,
        "internal_test_evaluations": 0})
    models = {}
    for name, feature_set in cfg["models"].items():
        print(f"Fitting {name}: {feature_set}, train only ({len(train)} observations)...", flush=True)
        encoder = ContextEncoder.fit(train, feature_set)
        scaler = StandardScaler().fit(encoder.transform(train))
        classifier = LogisticRegression(**cfg["logistic_regression"], random_state=cfg["seed"])
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            classifier.fit(scaler.transform(encoder.transform(train)), train.label_id)
        if not np.array_equal(classifier.classes_, np.arange(len(classes))):
            raise ValueError("context class order mismatch")
        models[name] = (encoder, scaler, classifier)
        write_json(output / "models" / f"{name}.json", {"encoder": encoder.state_dict(), "classes": classes,
            "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
            "coef": classifier.coef_.tolist(), "intercept": classifier.intercept_.tolist(),
            "iterations": classifier.n_iter_.tolist(), "feature_names": encoder.feature_names})

    summary, pairwise, val_predictions = [], [], {"A": visual}
    val_metrics = score_group(val, visual, classes)
    summary.append({"model": "A", "condition": "normal", **{k: val_metrics[k] for k in ["n", "macro_f1", "accuracy"]}})
    write_json(output / "val/A/metrics_normal.json", val_metrics)
    for name, model in models.items():
        encoder = model[0]
        conditions = ["normal", "missing_all", "missing_geo", "shuffled_geo", "imprecise_geo"]
        if encoder.feature_set != "geo":
            conditions += ["missing_climate", "shuffled_climate"]
        if encoder.substrate_categories is not None:
            conditions += ["missing_substrate", "shuffled_substrate"]
        for condition in conditions:
            altered = perturb_context(val, condition, cfg["seed"], encoder.feature_set)
            metadata = context_prediction(model, altered)
            probabilities = mix_scores(visual, metadata, context_available(altered, encoder), cfg["context_weight"])
            scores = score_group(val, probabilities, classes)
            write_json(output / f"val/{name}/metrics_{condition}.json", scores)
            summary.append({"model": name, "condition": condition,
                            **{k: scores[k] for k in ["n", "macro_f1", "accuracy"]}})
            if condition == "normal":
                val_predictions[name] = probabilities
                context_only = score_group(val, metadata, classes)
                write_json(output / f"val/{name}/context_only_metrics.json", context_only)
        predicted = val[["observation_uid", "label_id"]].copy()
        predicted["pred_label_id"] = val_predictions[name].argmax(1)
        for i, genus in enumerate(classes):
            predicted[f"p_{genus}"] = val_predictions[name][:, i]
        predicted.to_csv(output / f"val/{name}/predictions_normal.csv", index=False, lineterminator="\n")
    for reference, candidate in (("A", "B"), ("B", "C"), ("C", "D")):
        delta = paired_delta(val.label_id.to_numpy(), val_predictions[reference].argmax(1),
                             val_predictions[candidate].argmax(1), cfg["bootstrap_resamples"], cfg["seed"])
        pairwise.append({"comparison": f"{candidate}-{reference}", **delta})
    pd.DataFrame(summary).to_csv(output / "validation_summary.csv", index=False, lineterminator="\n")
    write_json(output / "paired_validation.json", pairwise)
    print(pd.DataFrame(summary).query("condition == 'normal'").to_string(index=False), flush=True)
    if args.external:
        evaluate_external(cfg, models, classes, train, output)
    write_json(output / "completed.json", {"models": list(models), "trained_visual_backbone": False,
        "external_evaluated": args.external, "internal_test_evaluations": 0})


def evaluate_external(cfg, models, classes, train, output):
    import evaluate_external as external
    external.configure(cfg["external_set"])
    bundle = external.load_set(external.IMAGE_ROOT)  # Fresh hashes for every external photograph.
    ext_obs = bundle["obs"].copy()
    visual_run_name = Path(cfg["visual_run"]).name
    old_report = resolve(f"reports/external/{cfg['external_set']}/{visual_run_name}")
    evaluation = json.loads((old_report / "evaluation.json").read_text(encoding="utf-8"))
    if (evaluation["checkpoint_sha256"] != cfg["visual_checkpoint_sha256"] or
            evaluation["set_fingerprint"] != bundle["manifest"]["content_fingerprint"] or
            bundle["classes"] != classes):
        raise ValueError("external scores are not from the same frozen visual model/set")
    ext_obs = ext_obs[ext_obs.observation_uid.isin(bundle["frame"].observation_uid)].sort_values("observation_uid").reset_index(drop=True)
    # Ext_full has location metadata only. Do not invent CHELSA or substrate values.
    for column in BIO:
        ext_obs[column] = np.nan
    ext_obs["substrate_class"], ext_obs["substrate__quality"] = "", "NOT_PROVIDED"
    frozen_path = old_report / "predictions_external.csv"
    predictions = pd.read_csv(frozen_path)
    pairs = predictions.set_index("image_uid").observation_uid.sort_index()
    expected = bundle["frame"].set_index("image_uid").observation_uid.sort_index()
    if not pairs.equals(expected):
        raise ValueError("external image cohort differs from frozen predictions")
    visual = aligned_visual(frozen_path, ext_obs, classes)
    qlo, qhi = cfg["geographic_guard_quantiles"]
    bounds = train.loc[geo_valid(train), ["latitude", "longitude"]].quantile([qlo, qhi])
    in_domain = (ext_obs.latitude.between(bounds.latitude.iloc[0], bounds.latitude.iloc[1]) &
                 ext_obs.longitude.between(bounds.longitude.iloc[0], bounds.longitude.iloc[1]))
    groups = {"full": np.ones(len(ext_obs), dtype=bool),
              "colombia": ext_obs.source_id.isin(["inaturalist_co", "mushroom_observer_co"]),
              "orinoquia": ext_obs.region.eq("orinoquia")}
    for source in sorted(ext_obs.source_id.unique()):
        groups[source] = ext_obs.source_id.eq(source)
    rows, deltas = [], []
    all_predictions = {"A": visual}
    for name, model in models.items():
        contextual = context_prediction(model, ext_obs)
        available = context_available(ext_obs, model[0])
        raw = mix_scores(visual, contextual, available, cfg["context_weight"])
        guarded = mix_scores(visual, contextual, available & in_domain.to_numpy(), cfg["context_weight"])
        all_predictions[name + "_raw"] = raw
        all_predictions[name + "_guarded"] = guarded
    for name, probabilities in all_predictions.items():
        predicted = ext_obs[["observation_uid", "label_id", "source_id", "region"]].copy()
        predicted["pred_label_id"] = probabilities.argmax(1)
        for i, genus in enumerate(classes):
            predicted[f"p_{genus}"] = probabilities[:, i]
        destination = output / "external"
        destination.mkdir(exist_ok=True)
        predicted.to_csv(destination / f"predictions_{name}.csv", index=False, lineterminator="\n")
        for group, mask in groups.items():
            if not np.any(mask):
                continue
            group_scores = score_group(ext_obs.loc[mask], probabilities[mask], classes)
            rows.append({"model": name, "group": group, **{k: group_scores[k] for k in
                ["n", "accuracy", "macro_f1", "macro_f1_present_classes", "classes_present"]}})
            write_json(output / f"external/{name}/metrics_{group}.json", group_scores)
    for group, mask in groups.items():
        if group not in ("full", "colombia", "orinoquia") or not np.any(mask):
            continue
        for reference, candidate in (("A", "B_raw"), ("B_raw", "C_raw"), ("C_raw", "D_raw"), ("A", "D_guarded")):
            delta = paired_delta(ext_obs.loc[mask, "label_id"].to_numpy(), all_predictions[reference][mask].argmax(1),
                all_predictions[candidate][mask].argmax(1), cfg["bootstrap_resamples"], cfg["seed"])
            deltas.append({"group": group, "comparison": f"{candidate}-{reference}", **delta})
    pd.DataFrame(rows).to_csv(output / "external_summary.csv", index=False, lineterminator="\n")
    write_json(output / "paired_external.json", deltas)
    write_json(output / "external/provenance.json", {"set_fingerprint": bundle["manifest"]["content_fingerprint"],
        "images_freshly_hash_verified": len(bundle["frame"]), "predictions_sha256": sha256_file(frozen_path),
        "original_evaluation_sha256": sha256_file(old_report / "evaluation.json"),
        "climate_available": 0, "substrate_available": 0, "geo_ok": int(geo_valid(ext_obs).sum()),
        "in_training_domain": int(in_domain.sum()), "guard_bounds": bounds.to_dict(),
        "overlapping_groups": True, "previously_evaluated_set": True, "used_for_selection": False})
    print(pd.DataFrame(rows).query("group in ['colombia','orinoquia']").to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
