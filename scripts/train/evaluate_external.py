"""Evaluate trained checkpoints on an external set (domain shift; no training, no tuning).

    python scripts/train/evaluate_external.py --set ext_full_001 --run runs/<run_id> [--device cpu]
    python scripts/train/evaluate_external.py --set ext_co_001 --run runs/<run_id>
    python scripts/train/evaluate_external.py --set ext_full_001 --summary-only   # rebuild the README

Sets: ``ext_co_001`` = Colombia (iNaturalist CO + Mushroom Observer CO, region per observation);
``ext_full_001`` = the complete ``test_external`` of ``v001`` (Colombia + MIND.Funga v2).

Protocol
  * Each checkpoint is evaluated once per set: ``reports/external/<set>/<run_id>/`` is created exclusively and an
    existing directory blocks a repeat. The external set is never used to select models or hyperparameters.
  * For the S5 campaign only the frozen selection (``reports/s5/selected_model.json``, checkpoint SHA-256 included)
    can be evaluated, so the external result cannot feed back into the choice of trial.
  * The class list of the checkpoint must equal the set's (same 10 genera, same order). The dataset fingerprints
    differ by design (trained on v002, evaluated on a set derived from v001); both are recorded.
  * Every image is checked against ``data/versions/<set>/image_integrity.csv`` before inference.
  * Metrics at image and observation level (``mean_prob``), overall, for Colombia, by source and by region
    (Orinoquía = Meta, Casanare, Arauca, Vichada; MIND.Funga = ``no_location``). Observation macro-F1 is given over
    the 10 classes and over the classes present in the group; 95 % CIs are percentile bootstrap over observations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import REPO_ROOT, resolve, sha256_file  # noqa: E402

SET_ID = "ext_co_001"
SET_VER = REPO_ROOT / "data" / "versions" / SET_ID
SET_PROC = REPO_ROOT / "data" / "processed" / SET_ID
OUT = REPO_ROOT / "reports" / "external" / SET_ID
IMAGE_ROOT = REPO_ROOT / "data" / "raw" / "images" / "external_v001"   # shared by every external set
S5_SELECTION = REPO_ROOT / "reports" / "s5" / "selected_model.json"
REGIONS = ("orinoquia", "other", "unknown", "no_location")
COLOMBIA = ("inaturalist_co", "mushroom_observer_co")
SOURCE_ES = {"inaturalist_co": "iNaturalist CO", "mushroom_observer_co": "Mushroom Observer CO",
             "mind_funga_v2": "MIND.Funga v2"}


def configure(set_id: str) -> None:
    global SET_ID, SET_VER, SET_PROC, OUT
    SET_ID = set_id
    SET_VER = REPO_ROOT / "data" / "versions" / set_id
    SET_PROC = REPO_ROOT / "data" / "processed" / set_id
    OUT = REPO_ROOT / "reports" / "external" / set_id
    if not (SET_VER / "manifest.json").is_file():
        raise SystemExit(f"external set {set_id} is not published (data/versions/{set_id}/manifest.json)")


# --------------------------------------------------------------------------- external set
def load_set(image_root: Path) -> dict[str, Any]:
    man = json.loads((SET_VER / "manifest.json").read_text(encoding="utf-8"))
    for name in ("observations.csv", "images.csv"):
        expected = next(o["sha256"] for o in man["outputs"] if o["name"] == name)
        if sha256_file(SET_PROC / name) != expected:
            raise SystemExit(f"{SET_PROC / name} does not match {SET_ID}/manifest.json; run build_external_set.py")
    integrity_path = SET_VER / "image_integrity.csv"
    if not integrity_path.is_file():
        raise SystemExit(f"{integrity_path} not found: finish scripts/data/download_external_images.py first")
    obs = pd.read_csv(SET_PROC / "observations.csv", dtype={"source_observation_id": str, "gbif_id": str})
    img = pd.read_csv(SET_PROC / "images.csv", dtype=str)
    img["label_id"] = img["label_id"].astype(int)
    integ = pd.read_csv(integrity_path, dtype=str)
    dl = pd.read_csv(image_root / "download_manifest.csv", dtype=str, keep_default_na=False)
    frame = img.merge(integ[["image_uid", "sha256"]], on="image_uid", how="inner", validate="one_to_one")
    frame = frame.merge(dl[["image_uid", "filename"]], on="image_uid", how="left", validate="one_to_one")
    if frame["filename"].isna().any():
        raise SystemExit("images frozen in image_integrity.csv are missing from download_manifest.csv")
    frame["path"] = frame["filename"].map(lambda f: str(image_root / f))
    print(f"Verificando SHA-256 de {len(frame):,} imágenes externas...", flush=True)
    bad = [p for p, h in zip(frame["path"], frame["sha256"]) if not Path(p).is_file() or sha256_file(Path(p)) != h]
    if bad:
        raise SystemExit(f"{len(bad)} images are missing or differ from image_integrity.csv (first: {bad[0]})")
    frame = frame.sort_values("image_uid").reset_index(drop=True)
    classes = pd.read_csv(SET_VER / "classes.csv").sort_values("label_id")["genus"].tolist()
    return {"manifest": man, "obs": obs, "frame": frame, "classes": classes,
            "integrity_sha256": sha256_file(integrity_path),
            "images_excluded": int(len(img) - len(frame)),
            "observations_without_images": int((~obs["observation_uid"].isin(frame["observation_uid"])).sum())}


# --------------------------------------------------------------------------- metrics
def confusion(y: np.ndarray, pred: np.ndarray, n: int) -> np.ndarray:
    return np.bincount(y * n + pred, minlength=n * n).reshape(n, n)


def macro_f1(cm: np.ndarray, labels: np.ndarray) -> float:
    tp = np.diag(cm).astype(float)
    denom = cm.sum(axis=0) + cm.sum(axis=1)
    f1 = np.divide(2 * tp, denom, out=np.zeros(len(tp)), where=denom > 0)
    return float(f1[labels].mean())


def bootstrap(y: np.ndarray, pred: np.ndarray, n: int, present: np.ndarray, resamples: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=(resamples, len(y)))
    all_labels = np.arange(n)
    s10, sp, acc = [], [], []
    for i in idx:
        cm = confusion(y[i], pred[i], n)
        s10.append(macro_f1(cm, all_labels))
        sp.append(macro_f1(cm, present))
        acc.append(float((y[i] == pred[i]).mean()))
    q = lambda v: [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]  # noqa: E731
    return {"macro_f1_ci95": q(s10), "macro_f1_present_classes_ci95": q(sp), "accuracy_ci95": q(acc),
            "ci_method": f"percentile bootstrap over observations, {resamples} resamples, seed {seed}"}


def group_metrics(frame: pd.DataFrame, probs: np.ndarray, classes: list[str], top_k: list[int],
                  resamples: int, seed: int) -> dict[str, Any]:
    from metrics import aggregate_observations
    from pipeline.evaluation import evaluate
    m = evaluate(frame, probs, classes, top_k)
    yo, po = aggregate_observations(frame["image_uid"], frame["observation_uid"], frame["label_id"].to_numpy(), probs)
    pred = po.argmax(axis=1)
    present = np.unique(yo)
    ol = m["observation_level"]
    ol["classes_present"] = [classes[i] for i in present]
    ol["macro_f1_present_classes"] = macro_f1(confusion(yo, pred, len(classes)), present)
    ol.update(bootstrap(yo, pred, len(classes), present, resamples, seed))
    m["n_observations"], m["n_images"] = int(len(yo)), int(len(frame))
    return m


# --------------------------------------------------------------------------- one checkpoint
def check_s5_selection(run: Path, cfg: dict[str, Any], ckpt_sha: str, allow: bool) -> None:
    if cfg.get("experiment", {}).get("campaign") != "mobilenet_v3_large_v002" or allow:
        return
    if not S5_SELECTION.is_file():
        raise SystemExit("S5 has no frozen selection yet; external evaluation of S5 trials would leak into selection")
    sel = json.loads(S5_SELECTION.read_text(encoding="utf-8"))["selected"]
    if resolve(sel["run"]).resolve() != run.resolve() or sel["checkpoint_sha256"] != ckpt_sha:
        raise SystemExit(f"{run.name} is not the frozen S5 selection ({sel['run']}); use --allow-unselected only for "
                         "a documented exploratory comparison")


def evaluate_run(run: Path, ds: dict[str, Any], args: argparse.Namespace) -> Path:
    import torch
    from torch.utils.data import DataLoader
    from pipeline.artifacts import code_hashes, environment, git_commit, write_json
    from pipeline.datasets import ImageTable
    from pipeline.evaluation import predict, save_predictions
    from pipeline.models import build_model
    from pipeline.transforms import build_transforms

    if not (run / "completed.json").is_file():
        raise SystemExit(f"{run} is not a completed run")
    info = json.loads((run / "run.json").read_text(encoding="utf-8"))
    best = run / "best.pt"
    ckpt_sha = sha256_file(best)
    checkpoint = torch.load(best, map_location="cpu", weights_only=True)
    cfg = checkpoint["config"]
    if checkpoint["classes"] != ds["classes"]:
        raise SystemExit(f"{run.name}: checkpoint classes {checkpoint['classes']} != set classes {ds['classes']}")
    check_s5_selection(run, cfg, ckpt_sha, args.allow_unselected)

    dest = OUT / run.name
    OUT.mkdir(parents=True, exist_ok=True)
    dest.mkdir(exist_ok=False)          # one evaluation per checkpoint and set
    started = datetime.now(timezone.utc).isoformat()
    with (dest / "started.json").open("x", encoding="utf-8") as fh:
        json.dump({"started_utc": started, "run": run.name, "checkpoint_sha256": ckpt_sha}, fh, indent=2)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    amp = bool(cfg["training"].get("amp")) and device.type == "cuda"
    model = build_model(cfg, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model"])
    _, tf = build_transforms(cfg)
    frame = ds["frame"]
    loader = DataLoader(ImageTable(frame, tf), batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda")
    probs = predict(model, loader, device, amp, f"Externa {SET_ID} - {cfg['run_name']}")
    top_k = cfg["evaluation"]["top_k"]

    groups: dict[str, Any] = {"all": group_metrics(frame, probs, ds["classes"], top_k, args.resamples, args.seed)}
    masks = {}
    colombia = frame["source_id"].isin(COLOMBIA).to_numpy()
    if colombia.any() and not colombia.all():
        masks["scope:colombia"] = colombia
    for col, values in (("source_id", sorted(frame["source_id"].unique())), ("region", REGIONS)):
        for v in values:
            masks[f"{col.replace('_id', '')}:{v}"] = (frame[col] == v).to_numpy()
    for name, mask in masks.items():
        if mask.any():
            groups[name] = group_metrics(frame[mask].reset_index(drop=True), probs[mask], ds["classes"], top_k,
                                         args.resamples, args.seed)
    write_json(dest / "metrics_external.json", {"set_id": SET_ID, "run": run.name, "groups": groups})
    save_predictions(dest / "predictions_external.csv", frame, probs, ds["classes"])

    from metrics import aggregate_observations
    yo, po = aggregate_observations(frame["image_uid"], frame["observation_uid"], frame["label_id"].to_numpy(), probs)
    ids = sorted(frame["observation_uid"].unique())
    op = pd.DataFrame({"observation_uid": ids, "pred_label_id": po.argmax(axis=1), "p_pred": po.max(axis=1).round(6)})
    op = ds["obs"][["observation_uid", "source_id", "region", "department", "record_url", "genus", "label_id",
                    "gbif_id"]].merge(op, on="observation_uid", how="inner", validate="one_to_one")
    op["pred_genus"] = op["pred_label_id"].map(dict(enumerate(ds["classes"])))
    op["correct"] = op["pred_label_id"] == op["label_id"]
    op.to_csv(dest / "observation_predictions.csv", index=False, lineterminator="\n")

    here = Path(__file__)
    write_json(dest / "evaluation.json", {
        "set_id": SET_ID, "set_fingerprint": ds["manifest"]["content_fingerprint"],
        "image_integrity_sha256": ds["integrity_sha256"],
        "images_evaluated": int(len(frame)), "observations_evaluated": int(len(ids)),
        "images_excluded_not_downloadable": ds["images_excluded"],
        "observations_without_images": ds["observations_without_images"],
        "run": str(run.relative_to(REPO_ROOT)), "run_name": cfg["run_name"], "checkpoint_sha256": ckpt_sha,
        "run_json_sha256": sha256_file(run / "run.json"),
        "checkpoint_dataset": {"version": info["dataset"]["version"],
                               "content_fingerprint": checkpoint.get("dataset_fingerprint")},
        "dataset_fingerprint_differs_by_design": True,
        "config_sha256": hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
        "used_for_selection": False, "device": str(device), "amp": amp,
        "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
        "code_sha256": {"evaluate_external.py": sha256_file(here), **code_hashes()},
        "git": git_commit(REPO_ROOT), **environment(device)})
    (dest / "completed.json").write_text(json.dumps({"run": run.name, "set_id": SET_ID, "evaluations": 1}, indent=2),
                                         encoding="utf-8")
    o = groups["all"]["observation_level"]
    print(f"{cfg['run_name']}: obs macro-F1 {o['macro_f1']:.4f} {o['macro_f1_ci95']} | "
          f"Orinoquía {groups.get('region:orinoquia', {}).get('observation_level', {}).get('macro_f1_present_classes', float('nan')):.4f}")
    return dest


# --------------------------------------------------------------------------- summary README
def f3(v: float) -> str:
    return f"{v:.3f}".replace(".", ",")


def ci(v: list[float]) -> str:
    return f"[{f3(v[0])}; {f3(v[1])}]"


def in_domain(run: Path) -> str:
    for s in ("test", "val"):
        p = run / f"metrics_{s}.json"
        if p.is_file():
            return f"{f3(json.loads(p.read_text(encoding='utf-8'))['observation_level']['macro_f1'])} ({s})"
    return "—"


def paired_difference(a: pd.DataFrame, b: pd.DataFrame, region_name: str | None, n: int,
                      resamples: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Observation macro-F1 of b minus a, with a paired percentile bootstrap (same resampled observations)."""
    a, b = a.sort_values("observation_uid"), b.sort_values("observation_uid")
    if not (a["observation_uid"].to_numpy() == b["observation_uid"].to_numpy()).all():
        raise SystemExit("predictions of the two models cover different observations")
    mask = np.ones(len(a), bool) if region_name is None else (a["region"] == region_name).to_numpy()
    y = a["label_id"].to_numpy()[mask]
    pa, pb = a["pred_label_id"].to_numpy()[mask], b["pred_label_id"].to_numpy()[mask]
    labels = np.arange(n) if region_name is None else np.unique(y)
    f = lambda yy, pp: macro_f1(confusion(yy, pp, n), labels)  # noqa: E731
    idx = np.random.default_rng(seed).integers(0, len(y), size=(resamples, len(y)))
    d = np.array([f(y[i], pb[i]) - f(y[i], pa[i]) for i in idx])
    return f(y, pb) - f(y, pa), float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


def site_day_note(ori: pd.DataFrame) -> str:
    """Observations of the same site-day (v001 group_id) are not independent: say how many share one."""
    v001 = REPO_ROOT / "data" / "processed" / "v001" / "observations.csv"
    generic = ("* El bootstrap remuestrea observaciones, no sitios; observaciones del mismo sitio y día no son "
               "independientes, así que los IC95 pueden ser optimistas.")
    if not v001.is_file():
        return generic
    g = pd.read_csv(v001, usecols=["observation_uid", "group_id"])
    x = ori.merge(g, on="observation_uid", how="left")
    parts = []
    for src, s in x.groupby("source_id"):
        sz = s.groupby("group_id").size()
        parts.append(f"{SOURCE_ES.get(src, src)}: {len(s)} observaciones en {len(sz)} sitio-días, "
                     f"{int(sz[sz > 1].sum())} comparten sitio-día")
    return ("* En la Orinoquía, " + "; ".join(parts) + " (sitio-día de `v001`: misma fecha y celda de ~1 km). "
            "El bootstrap remuestrea observaciones, no sitio-días, así que los IC95 de ese grupo pueden ser optimistas.")


def summary() -> None:
    done = sorted(p.parent for p in OUT.glob("*/completed.json"))
    if not done:
        return
    man = json.loads((SET_VER / "manifest.json").read_text(encoding="utf-8"))
    res = []
    for d in done:
        ev = json.loads((d / "evaluation.json").read_text(encoding="utf-8"))
        met = json.loads((d / "metrics_external.json").read_text(encoding="utf-8"))["groups"]
        res.append((d, ev, met, pd.read_csv(d / "observation_predictions.csv", dtype={"gbif_id": str})))
    classes = man["classes"]
    ev0 = res[0][1]
    full = "source:mind_funga_v2" in res[0][2]
    mil = lambda n: f"{int(n):,}".replace(",", ".")  # noqa: E731
    title = (f"Evaluación externa completa: `test_external` de `v001` (`{SET_ID}`)" if full
             else f"Evaluación externa en Colombia (`{SET_ID}`)")
    L = [f"# {title}", "",
         "> Generado por `scripts/train/evaluate_external.py`; no editar a mano.", "",
         f"Conjunto: [`data/versions/{SET_ID}/`](../../../data/versions/{SET_ID}/DATASET_CARD.md), huella "
         f"`{man['content_fingerprint'][:16]}…`, integridad de imágenes `{ev0['image_integrity_sha256'][:16]}…`. "
         f"{mil(ev0['observations_evaluated'])} observaciones y {mil(ev0['images_evaluated'])} imágenes evaluadas"
         + (f"; {ev0['images_excluded_not_downloadable']} imágenes no descargables y "
            f"{ev0['observations_without_images']} observaciones sin imagen quedaron fuera."
            if ev0["images_excluded_not_downloadable"] else "; se descargaron todas las imágenes."),
         "",
         "Modelos entrenados solo con FungiTastic (`v002`, Dinamarca). Prueba de cambio de dominio: no se usó para "
         "seleccionar modelos. Unidad estadística: la observación (`mean_prob`). IC95 por bootstrap de observaciones.",
         "", "## Resultado global", "",
         "| Modelo | Macro-F1 obs. (10 clases) | IC95 | Exactitud obs. | Exactitud balanceada obs. | Top-3 obs. | "
         "Macro-F1 img. | Macro-F1 obs. en FungiTastic |", "|---|---:|---|---:|---:|---:|---:|---:|"]
    for d, ev, met, _ in res:
        o, i = met["all"]["observation_level"], met["all"]["image_level"]
        L.append(f"| `{ev['run_name']}` | {f3(o['macro_f1'])} | {ci(o['macro_f1_ci95'])} | {f3(o['accuracy'])} | "
                 f"{f3(o['balanced_accuracy'])} | {f3(o.get('top3_accuracy', float('nan')))} | {f3(i['macro_f1'])} | "
                 f"{in_domain(REPO_ROOT / ev['run'])} |")
    L += ["", "## Por fuente y por región", "",
          "En los grupos donde falta alguna clase, el macro-F1 sobre 10 clases cuenta como 0 las clases ausentes; "
          "la columna «clases presentes» es la comparable.", "",
          "| Modelo | Grupo | Obs. | Clases presentes | Macro-F1 obs. (clases presentes) | IC95 | Exactitud obs. | IC95 |",
          "|---|---|---:|---:|---:|---|---:|---|"]
    names = [("scope:colombia", "**Colombia** (iNaturalist + MO)"),
             ("source:inaturalist_co", "iNaturalist CO"), ("source:mushroom_observer_co", "Mushroom Observer CO"),
             ("source:mind_funga_v2", "**MIND.Funga v2** (sin ubicación)"),
             ("region:orinoquia", "**Orinoquía**"), ("region:other", "Otra región de Colombia"),
             ("region:unknown", "Colombia, sin departamento")]
    for d, ev, met, _ in res:
        for g, label in names:
            if g in met:
                o = met[g]["observation_level"]
                L.append(f"| `{ev['run_name']}` | {label} | {o['n']} | {len(o['classes_present'])} | "
                         f"{f3(o['macro_f1_present_classes'])} | {ci(o['macro_f1_present_classes_ci95'])} | "
                         f"{f3(o['accuracy'])} | {ci(o['accuracy_ci95'])} |")
    if len(res) > 1:
        L += ["", "## Comparación pareada", "",
              f"Diferencia de macro-F1 de observación respecto de `{res[0][1]['run_name']}`, con los mismos "
              "remuestreos de observaciones para los dos modelos (2.000, semilla 0).", "",
              "| Modelo | Grupo | Diferencia | IC95 pareado |", "|---|---|---:|---|"]
        for _, ev, _, op in res[1:]:
            whole = "Conjunto completo (10 clases)" if full else "Colombia (10 clases)"
            for label, region_name in ((whole, None), ("**Orinoquía** (clases presentes)", "orinoquia")):
                d, lo, hi = paired_difference(res[0][3], op, region_name, len(classes))
                L.append(f"| `{ev['run_name']}` | {label} | {d:+.3f} | [{lo:+.3f}; {hi:+.3f}] |".replace(".", ","))
    per_genus = [g for g in ("all", "scope:colombia", "source:mind_funga_v2") if g in res[0][2]]
    if full:
        label = {"all": "Completo", "scope:colombia": "Colombia", "source:mind_funga_v2": "MIND.Funga"}
        L += ["", "## Por género (nivel de observación)", ""]
        for _, ev, met, _ in res:
            L += [f"`{ev['run_name']}`. Soporte y F1 por grupo.", "",
                  "| Género | " + " | ".join(f"Soporte {label[g]} | F1 {label[g]}" for g in per_genus) + " |",
                  "|---|" + "---:|---:|" * len(per_genus)]
            for c in range(len(classes)):
                cells = []
                for g in per_genus:
                    pc = met[g]["observation_level"]["per_class"][c]
                    cells.append(f"{pc['support']} | {f3(pc['f1']) if pc['support'] else '—'}")
                L.append(f"| {classes[c]} | " + " | ".join(cells) + " |")
            L.append("")
    L += ["", "## Orinoquía", "",
          "Observaciones con departamento Meta, Casanare, Arauca o Vichada (mismo criterio que la descarga GBIF "
          "`0007586-260221153910048`). Lista completa con enlace a cada registro y la predicción de cada modelo en "
          "[`orinoquia_predictions.csv`](orinoquia_predictions.csv).", "",
          "### Por género (nivel de observación)", ""]
    head = "| Género | Soporte | " + " | ".join(f"Recall `{ev['run_name']}` | F1 `{ev['run_name']}`" for _, ev, _, _ in res) + " |"
    L += [head, "|---|---:|" + "---:|---:|" * len(res)]
    for c in range(len(classes)):
        sup = res[0][2].get("region:orinoquia", {}).get("observation_level", {}).get("per_class", [{}] * len(classes))[c].get("support", 0)
        if not sup:
            continue
        cells = []
        for _, _, met, _ in res:
            pc = met["region:orinoquia"]["observation_level"]["per_class"][c]
            cells.append(f"{f3(pc['recall'])} | {f3(pc['f1'])}")
        L.append(f"| {classes[c]} | {sup} | " + " | ".join(cells) + " |")
    absent = [c for c in classes if c not in res[0][2].get("region:orinoquia", {}).get("observation_level", {}).get("classes_present", [])]
    if absent:
        L += ["", f"Sin observaciones en la Orinoquía: {', '.join(absent)}."]
    L += ["", "### Por departamento y fuente (exactitud de observación)", "",
          "| Departamento | Fuente | Obs. | " + " | ".join(f"`{ev['run_name']}`" for _, ev, _, _ in res) + " |",
          "|---|---|---:|" + "---:|" * len(res)]
    base = res[0][3]
    ori = base[base["region"] == "orinoquia"]
    for (dep, src), g in ori.groupby(["department", "source_id"]):
        cells = []
        for _, _, _, op in res:
            x = op[(op["region"] == "orinoquia") & (op["department"] == dep) & (op["source_id"] == src)]
            cells.append(f3(x["correct"].mean()))
        L.append(f"| {dep} | {SOURCE_ES.get(src, src)} | {len(g)} | " + " | ".join(cells) + " |")
    # combined Orinoquía table
    comb = ori[["observation_uid", "source_id", "department", "record_url", "gbif_id", "genus"]].copy()
    for _, ev, _, op in res:
        x = op.set_index("observation_uid")
        comb[f"pred_{ev['run_name']}"] = comb["observation_uid"].map(x["pred_genus"])
        comb[f"p_{ev['run_name']}"] = comb["observation_uid"].map(x["p_pred"])
    comb.sort_values(["department", "source_id", "observation_uid"]).to_csv(OUT / "orinoquia_predictions.csv",
                                                                           index=False, lineterminator="\n")
    L += ["", "## Lectura y límites", "",
          "* Las etiquetas externas son identificaciones de comunidad (iNaturalist grado de investigación; consenso de "
          "votos de Mushroom Observer), no curaduría experta como FungiTastic: la caída frente a FungiTastic mezcla "
          "cambio de dominio con ruido de etiqueta.",
          *(["* MIND.Funga v2 no tiene coordenadas, fecha ni país; sus etiquetas vienen del nombre de carpeta y mezcla "
              "imágenes neotropicales de varios orígenes (una imagen por observación). Se reporta aparte de Colombia "
              "y no entra en ningún grupo regional."] if full else []),
          "* La Orinoquía tiene 8 de los 10 géneros y Trametes concentra cerca de la mitad de sus observaciones; las "
          "clases con pocas observaciones tienen intervalos amplios.",
          site_day_note(ori),
          "* Una sola semilla por modelo; las diferencias entre modelos cuyos IC95 se solapan no son concluyentes.", ""]
    (OUT / "README.md").write_text("\n".join(L), encoding="utf-8")
    print(OUT / "README.md")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", default="ext_full_001", help="external set id (ext_full_001 or ext_co_001)")
    ap.add_argument("--run", action="append", default=[], help="completed run directory (repeatable)")
    ap.add_argument("--images", default=str(IMAGE_ROOT), help="root of the downloaded external images")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--resamples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow-unselected", action="store_true", help="evaluate a non-selected S5 trial (exploratory)")
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()
    configure(args.set)
    if not args.summary_only:
        if not args.run:
            ap.error("give at least one --run")
        ds = load_set(Path(args.images))
        for r in args.run:
            evaluate_run(resolve(r), ds, args)
    summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
