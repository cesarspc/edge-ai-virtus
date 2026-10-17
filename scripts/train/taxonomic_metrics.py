"""S6: consolidate baseline metrics per taxonomic level from the saved predictions of completed runs.

Reads only versioned artifacts (no images, no checkpoints, no PyTorch):
    runs/<run_id>/{run.json, completed.json, predictions_<split>.csv, metrics_<split>.json}
    runs/<run_id>/final_test/predictions_test.csv     (S5 final test, when it exists)
    data/taxonomy/<version>/{genus_taxonomy.csv, observation_species.csv, manifest.json}
    data/versions/<version>/splits.csv                (coverage check)

Before reporting, every evaluation must (1) use the dataset fingerprint of the taxonomy, (2) cover every observation
of its split, and (3) reproduce at genus level the metrics saved by the run (confusion matrices identical).

Writes reports/s6/: taxonomic_summary.csv, taxonomic_per_class.csv, species_breakdown.csv,
taxonomic_metrics.json (full detail, confusion matrices) and README.md (tables).

Usage:
    python scripts/train/taxonomic_metrics.py
    python scripts/train/taxonomic_metrics.py --check    # recompute and compare with reports/s6/ (exit 1 on change)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import REPO_ROOT  # noqa: E402
from metrics import aggregate_observations, classification_metrics  # noqa: E402
from pipeline.predictions import load_predictions  # noqa: E402
from pipeline.taxonomy import LEVELS, load_taxonomy, species_breakdown, taxonomic_report  # noqa: E402

LEVEL_ES = {"genus": "género", "family": "familia", "order": "orden", "class": "clase", "phylum": "filo"}
UNIT_ES = {"observation_level": "observación", "image_level": "imagen"}
OUTPUTS = ("taxonomic_summary.csv", "taxonomic_per_class.csv", "species_breakdown.csv", "taxonomic_metrics.json", "README.md")


def text_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def discover(runs_dir: Path) -> list[dict]:
    """Completed full training runs and their evaluated splits (smoke / eval-only runs are skipped)."""
    found = []
    for run in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        if not (run / "completed.json").is_file():
            continue
        info = json.loads((run / "run.json").read_text(encoding="utf-8"))
        if info["mode"] != "train":
            continue
        done = json.loads((run / "completed.json").read_text(encoding="utf-8"))
        base = {"run_id": run.name, "run_name": info["config"]["run_name"],
                "architecture": info["config"]["model"]["architecture"], "info": info}
        for split in done["evaluated_splits"]:
            found.append({**base, "split": split, "predictions": run / f"predictions_{split}.csv",
                          "metrics": run / f"metrics_{split}.json", "source": "development run"})
        final = run / "final_test"
        if (final / "completed.json").is_file():
            found.append({**base, "split": "test", "predictions": final / "predictions_test.csv",
                          "metrics": final / "metrics_test.json", "source": "S5 final test"})
    return found


def reproduce_genus(pred: pd.DataFrame, probs: np.ndarray, classes: list[str], saved: dict, top_k: list[int]) -> dict:
    """Recompute the run's own genus metrics from the CSV and compare them with metrics_<split>.json."""
    y = pred["label_id"].to_numpy()
    yo, po = aggregate_observations(pred["image_uid"], pred["observation_uid"], y, probs)
    mine = {"image_level": classification_metrics(y, probs, classes, top_k),
            "observation_level": classification_metrics(yo, po, classes, top_k)}
    out = {}
    for unit in mine:
        a, b = mine[unit], saved[unit]
        keys = [k for k in ("macro_f1", "accuracy", "balanced_accuracy", *[f"top{k}_accuracy" for k in top_k]) if k in b]
        out[unit] = {"confusion_identical": a["confusion_matrix"] == b["confusion_matrix"],
                     "max_abs_diff": max(abs(a[k] - b[k]) for k in keys)}
    return out


def compute(version: str, resamples: int) -> tuple[dict[str, pd.DataFrame], dict]:
    tax_dir = REPO_ROOT / "data" / "taxonomy" / version
    tax_manifest = json.loads((tax_dir / "manifest.json").read_text(encoding="utf-8"))
    for o in tax_manifest["outputs"]:
        if text_sha256(tax_dir / o["name"]) != o["sha256"]:
            raise SystemExit(f"{tax_dir / o['name']} changed; rebuild with scripts/data/build_taxonomy.py")
    classes = pd.read_csv(REPO_ROOT / "data/versions" / version / "classes.csv").sort_values("label_id")["genus"].tolist()
    tax = load_taxonomy(tax_dir / "genus_taxonomy.csv", classes)
    species = pd.read_csv(tax_dir / "observation_species.csv")
    split_obs = pd.read_csv(REPO_ROOT / "data/versions" / version / "splits.csv", usecols=["observation_uid", "split"])

    summary, per_class, breakdown, detail = [], [], [], []
    for ev in discover(REPO_ROOT / "runs"):
        info = ev["info"]
        if info["dataset"]["version"] != version:
            continue
        if info["dataset"]["content_fingerprint"] != tax_manifest["fingerprint"]:
            raise SystemExit(f"{ev['run_id']}: dataset fingerprint differs from the taxonomy of {version}")
        top_k = info["config"]["evaluation"]["top_k"]
        pred, probs = load_predictions(ev["predictions"], classes)
        expected = set(split_obs.loc[split_obs["split"] == ev["split"], "observation_uid"])
        got = set(pred["observation_uid"])
        if got != expected:
            raise SystemExit(f"{ev['predictions']}: covers {len(got & expected)}/{len(expected)} observations of "
                             f"{ev['split']} and {len(got - expected)} foreign ones")
        check = reproduce_genus(pred, probs, classes, json.loads(ev["metrics"].read_text(encoding="utf-8")), top_k)
        if not all(c["confusion_identical"] and c["max_abs_diff"] < 1e-9 for c in check.values()):
            raise SystemExit(f"{ev['predictions']} does not reproduce its saved metrics: {check}")
        report = taxonomic_report(pred, probs, tax, top_k, resamples=resamples)
        model = f"{ev['run_name']}"
        key = {"model": model, "architecture": ev["architecture"], "split": ev["split"], "run_id": ev["run_id"]}
        for unit, levels in report.items():
            for level, m in levels.items():
                ci = m.get("macro_f1_ci95", [np.nan, np.nan])
                summary.append({**key, "unit": unit.replace("_level", ""), "level": level, "n_taxa": m["n_classes"],
                                "n": m["n"], "macro_f1": m["macro_f1"], "macro_f1_ci95_low": ci[0],
                                "macro_f1_ci95_high": ci[1], "accuracy": m["accuracy"],
                                "balanced_accuracy": m["balanced_accuracy"], "macro_precision": m["macro_precision"],
                                "macro_recall": m["macro_recall"], "marginal_macro_f1": m["marginal"]["macro_f1"],
                                "marginal_accuracy": m["marginal"]["accuracy"],
                                "rule_disagreements": m["marginal"]["disagreements_with_top1_genus"]})
                for c in m["per_class"]:
                    per_class.append({**key, "unit": unit.replace("_level", ""), "level": level, "taxon": c["taxon"],
                                      "genera": ";".join(m["genera_per_taxon"][c["taxon"]]), "support": c["support"],
                                      "precision": c["precision"], "recall": c["recall"], "f1": c["f1"]})
        sb = species_breakdown(pred, probs, species, classes)
        breakdown.append(sb.assign(**key)[[*key, *sb.columns]])
        detail.append({**key, "source": ev["source"],
                       "predictions": str(ev["predictions"].relative_to(REPO_ROOT)).replace("\\", "/"),
                       "predictions_sha256": text_sha256(ev["predictions"]), "observations": len(got),
                       "images": len(pred), "genus_reproduction": check, "metrics": report})
    if not detail:
        raise SystemExit(f"no completed runs on {version} found under runs/")
    tables = {"taxonomic_summary.csv": pd.DataFrame(summary), "taxonomic_per_class.csv": pd.DataFrame(per_class),
              "species_breakdown.csv": pd.concat(breakdown, ignore_index=True)}
    payload = {"version": version, "fingerprint": tax_manifest["fingerprint"],
               "taxonomy": {"path": f"data/taxonomy/{version}/genus_taxonomy.csv",
                            "manifest_outputs": tax_manifest["outputs"]},
               "levels": LEVELS, "prediction_rule": "taxon of the top-1 genus (secondary: sum of genus probabilities)",
               "observation_aggregation": "mean of image probabilities per observation, then the rule",
               "bootstrap_resamples": resamples, "evaluations": detail}
    return tables, payload


def readme(tables: dict[str, pd.DataFrame], payload: dict) -> str:
    s = tables["taxonomic_summary.csv"]
    tax = pd.read_csv(REPO_ROOT / payload["taxonomy"]["path"])
    lines = ["# S6: métricas base por nivel taxonómico", "",
             "Generado por `scripts/train/taxonomic_metrics.py` a partir de las predicciones guardadas; no edita a mano.",
             f"Dataset `{payload['version']}` · huella `{payload['fingerprint']}`.", "",
             "Regla principal: el taxón predicho es el del género top-1 (la decisión real del modelo, juzgada a un",
             "nivel superior). Por observación se promedian primero las probabilidades de sus imágenes (`mean_prob`).",
             "IC95 = bootstrap percentil sobre observaciones. La regla alternativa (sumar probabilidades por taxón)",
             "queda en `taxonomic_summary.csv` (`marginal_*`) solo como comparación.", "",
             "## Taxonomía usada (FungiTastic, sin ediciones)", "",
             "| Género | Familia | Orden | Clase | Filo | Especies | Obs. val | Obs. test |", "|---|---|---|---|---|---:|---:|---:|"]
    for r in tax.itertuples():
        lines.append(f"| {r.genus} | {r.family} | {r.order} | {r._5} | {r.phylum} | {r.n_species} | {r.n_obs_val} | {r.n_obs_test} |")
    for unit in ("observation", "image"):
        lines += ["", f"## Macro-F1 por {UNIT_ES[unit + '_level']}" + (" (métrica principal)" if unit == "observation" else ""), "",
                  "| Modelo | Partición | Nivel | Taxones | n | Macro-F1 | IC95 | Exactitud | Exactitud balanceada |",
                  "|---|---|---|---:|---:|---:|---|---:|---:|"]
        for r in s[s["unit"] == unit].itertuples():
            ci = f"{r.macro_f1_ci95_low:.4f}–{r.macro_f1_ci95_high:.4f}" if unit == "observation" else "—"
            lines.append(f"| {r.model} | {r.split} | {LEVEL_ES[r.level]} | {r.n_taxa} | {r.n} | {r.macro_f1:.4f} | {ci} | "
                         f"{r.accuracy:.4f} | {r.balanced_accuracy:.4f} |")
    pc = tables["taxonomic_per_class.csv"]
    pc = pc[(pc["unit"] == "observation") & pc["level"].isin(["class", "phylum"])]
    lines += ["", "## Clase y filo por observación: F1 por taxón (resto en `taxonomic_per_class.csv`)", "",
              "| Modelo | Partición | Nivel | Taxón | Géneros | Soporte | Precisión | Recall | F1 |",
              "|---|---|---|---|---|---:|---:|---:|---:|"]
    for r in pc.itertuples():
        lines.append(f"| {r.model} | {r.split} | {LEVEL_ES[r.level]} | {r.taxon} | {r.genera.replace(';', ', ')} | "
                     f"{r.support} | {r.precision:.4f} | {r.recall:.4f} | {r.f1:.4f} |")
    sp = tables["species_breakdown.csv"]
    weak = sp[(sp["observations"] >= 5) & (sp["genus_recall"] < 0.9)].sort_values(["model", "split", "genus_recall"])
    lines += ["", "## Especies con recall de género < 0,90 (≥ 5 observaciones)", "",
              "| Modelo | Partición | Especie | Obs. | Recall género | Error más frecuente |", "|---|---|---|---:|---:|---|"]
    for r in weak.itertuples():
        lines.append(f"| {r.model} | {r.split} | *{r.species}* | {r.observations} | {r.genus_recall:.3f} | "
                     f"{r.most_common_wrong_genus} ({r.most_common_wrong_count}) |")
    lines += ["", "## Evaluaciones incluidas", "", "| Modelo | Partición | Origen | Observaciones | Imágenes | Reproduce métricas guardadas |",
              "|---|---|---|---:|---:|---|"]
    for e in payload["evaluations"]:
        ok = all(c["confusion_identical"] for c in e["genus_reproduction"].values())
        lines.append(f"| {e['model']} | {e['split']} | {e['source']} | {e['observations']} | {e['images']} | {'sí' if ok else 'NO'} |")
    return "\n".join(lines) + "\n"


def serialize(tables: dict[str, pd.DataFrame], payload: dict) -> dict[str, bytes]:
    out = {n: t.to_csv(index=False, lineterminator="\n", float_format="%.6f").encode("utf-8") for n, t in tables.items()}
    out["taxonomic_metrics.json"] = (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    out["README.md"] = readme(tables, payload).encode("utf-8")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", default="v002")
    ap.add_argument("--resamples", type=int, default=1000)
    ap.add_argument("--check", action="store_true", help="recompute and compare with reports/s6 without writing")
    args = ap.parse_args()
    files = serialize(*compute(args.version, args.resamples))
    target = REPO_ROOT / "reports" / "s6"
    if args.check:
        changed = [n for n, b in files.items()
                   if not (target / n).is_file() or text_sha256(target / n) != hashlib.sha256(b).hexdigest()]
        print("OK: reports/s6 reproduced from saved predictions" if not changed else f"CHANGED: {changed}")
        return 1 if changed else 0
    target.mkdir(parents=True, exist_ok=True)
    for name, payload in files.items():
        (target / name).write_bytes(payload)
    s = pd.read_csv(target / "taxonomic_summary.csv")
    print(s[s["unit"] == "observation"][["model", "split", "level", "n_taxa", "macro_f1", "macro_f1_ci95_low",
                                         "macro_f1_ci95_high", "accuracy"]].to_string(index=False))
    print(f"written: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
