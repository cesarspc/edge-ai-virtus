"""Build a versioned processed dataset (Week 3): cleaning, normalisation, environmental metadata and splits.

Usage:
    python scripts/data/build_dataset.py                          # builds the version in configs/dataset_v001.yaml
    python scripts/data/build_dataset.py --config configs/dataset_v001.yaml
    python scripts/data/build_dataset.py --check                  # rebuild in memory and compare with the manifest
    python scripts/data/build_dataset.py --rewrite-manifest       # same content: refresh provenance (e.g. after committing)

Outputs:
    data/processed/<version>/observations.csv, images.csv   (full tables; ignored by Git, reproducible)
    data/versions/<version>/                                  (versioned: manifest, splits, classes, logs, card)

A published version is immutable: if data/versions/<version>/manifest.json exists and the rebuilt content differs,
the script stops and asks for a new version (see data/versions/README.md).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cleaning as K  # noqa: E402
import metadata_schema as MS  # noqa: E402
import splits as SP  # noqa: E402
from common import RAW_DIR, RAW_FILES_CSV, REPO_ROOT, log, rel, setup_logging, sha256_file, utc_now  # noqa: E402
from curation_sources import LOADERS, log_env  # noqa: E402

CODE_FILES = ["scripts/data/build_dataset.py", "scripts/data/curation_sources.py", "scripts/data/cleaning.py",
              "scripts/data/splits.py", "scripts/data/metadata_schema.py", "scripts/data/adapters.py",
              "scripts/data/common.py"]
MO_ID = re.compile(r"(\d+)\s*/?\s*$")


# --------------------------------------------------------------------------- cross-source duplicates
def cross_source_dedup(obs: pd.DataFrame, img: pd.DataFrame, log_: K.CleaningLog) -> tuple[pd.DataFrame, pd.DataFrame]:
    mo = obs["source_id"] == "mushroom_observer_co"
    inat = obs["source_id"] == "inaturalist_co"
    # D4: explicit link iNaturalist -> Mushroom Observer
    linked = set(obs.loc[inat, "mo_url"].dropna().astype(str).str.extract(MO_ID)[0].dropna())
    d4 = mo & obs["source_observation_id"].isin(linked)
    log_.add("D4", "mushroom_observer_co", int(d4.sum()), int(img["observation_uid"].isin(obs.loc[d4, "observation_uid"]).sum()),
             note=f"enlaces 'Mushroom Observer URL' en iNaturalist: {len(linked)}")
    obs = obs[~d4]
    # D5: same genus, same day, coordinates closer than 0.01 degrees
    cols = ["observation_uid", "genus", "event_date", "latitude", "longitude"]
    a = obs.loc[(obs["source_id"] == "mushroom_observer_co") & (obs["event_date_precision"] == "day"), cols]
    b = obs.loc[(obs["source_id"] == "inaturalist_co") & (obs["event_date_precision"] == "day"), cols]
    m = a.merge(b, on=["genus", "event_date"], suffixes=("_mo", "_in"))
    near = ((m["latitude_mo"] - m["latitude_in"]).abs() < 0.01) & ((m["longitude_mo"] - m["longitude_in"]).abs() < 0.01)
    d5 = set(m.loc[near, "observation_uid_mo"])
    log_.add("D5", "mushroom_observer_co", len(d5), int(img["observation_uid"].isin(d5).sum()))
    obs = obs[~obs["observation_uid"].isin(d5)]
    return obs, img[img["observation_uid"].isin(obs["observation_uid"])]


# --------------------------------------------------------------------------- build
def build(cfg: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    clog = K.CleaningLog()
    obs_parts, img_parts = [], []
    for sid in cfg["sources"]:
        log.info("=== loading %s", sid)
        o, i = LOADERS[sid](cfg, clog)
        obs_parts.append(o)
        img_parts.append(i)
    obs, img = pd.concat(obs_parts, ignore_index=True), pd.concat(img_parts, ignore_index=True)
    obs, img = cross_source_dedup(obs, img, clog)
    for sid in cfg["sources"]:   # E-rules are counted on the final observations of each source
        log_env(clog, sid, obs[obs["source_id"] == sid])
    extra = {"vote_cache": obs.set_index("observation_uid")["vote_cache"].dropna()}
    obs = obs.drop(columns=["vote_cache", "mo_url"])
    # labels, groups and splits
    label = {g: i for i, g in enumerate(cfg["classes"])}
    obs["label_id"] = obs["genus"].map(label).astype(int)
    img["label_id"] = img["genus"].map(label).astype(int)
    obs = SP.assign(obs, cfg)
    img["split"] = img["observation_uid"].map(obs.set_index("observation_uid")["split"])
    # C1: minimum observations per class in development
    dev_counts = obs[obs["split"] != "test_external"].groupby("genus").size().reindex(cfg["classes"], fill_value=0)
    minimum = cfg["cleaning"]["min_observations_per_class_dev"]
    low = dev_counts[dev_counts < minimum]
    clog.add("C1", "fungitastic", int(len(low)), None,
             note=f"mínimo {minimum}; clase menor: {dev_counts.idxmin()} ({int(dev_counts.min())} observaciones)")
    if len(low):
        raise SystemExit(f"C1 failed: classes below {minimum} observations in development: {low.to_dict()}")
    for r in ("N1", "N2", "N3", "N4", "N5", "N6"):
        clog.add(r, "todas", note="aplicada a todas las filas conservadas")
    obs = obs[MS.observation_columns(schema)].sort_values("observation_uid", kind="stable").reset_index(drop=True)
    img = img[MS.image_columns(schema)].sort_values("image_uid", kind="stable").reset_index(drop=True)
    errors = MS.validate(obs, img, schema, cfg)
    if errors:
        raise SystemExit("schema validation failed:\n  " + "\n  ".join(errors))
    return {"obs": obs, "img": img, "log": clog.frame(), "leakage": SP.leakage_checks(obs, img), "extra": extra}


# --------------------------------------------------------------------------- summaries
def summaries(res: dict[str, Any], cfg: dict[str, Any], schema: dict[str, Any]) -> dict[str, pd.DataFrame]:
    obs, img = res["obs"], res["img"]
    order = ["train", "val", "test", "test_external"]
    so = obs.groupby(["source_id", "genus", "split"]).size().rename("n_observations")
    si = img.groupby(["source_id", "genus", "split"]).size().rename("n_images")
    split_summary = pd.concat([so, si], axis=1).fillna(0).astype(int).reset_index()
    split_summary["split"] = pd.Categorical(split_summary["split"], order)
    split_summary = split_summary.sort_values(["source_id", "genus", "split"]).reset_index(drop=True)
    cov = []
    for group in schema["environmental_groups"]:
        vc = obs.groupby("source_id")[f"{group}__quality"].value_counts().rename("n_observations").reset_index()
        vc = vc.rename(columns={f"{group}__quality": "quality"}).assign(group=group)
        cov.append(vc)
    env_cov = pd.concat(cov)[["group", "source_id", "quality", "n_observations"]].sort_values(["group", "source_id", "quality"])
    tot = obs.groupby("source_id").size()
    env_cov["pct"] = (100 * env_cov["n_observations"] / env_cov["source_id"].map(tot)).round(2)
    classes = pd.DataFrame({"label_id": range(len(cfg["classes"])), "genus": cfg["classes"]})
    splits_tbl = obs[["observation_uid", "source_id", "genus", "label_id", "group_id", "split", "split_temporal", "n_images"]]
    return {"split_summary": split_summary, "env_coverage": env_cov.reset_index(drop=True), "classes": classes,
            "splits": splits_tbl}


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False, lineterminator="\n")
    return buf.getvalue().encode("utf-8")


def git_info() -> dict[str, Any]:
    def run(*a: str) -> str:
        try:
            return subprocess.run(["git", *a], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()
        except Exception:  # noqa: BLE001
            return ""
    return {"git_commit": run("rev-parse", "HEAD") or None, "git_dirty": bool(run("status", "--porcelain"))}


def input_files(cfg: dict[str, Any]) -> list[Path]:
    files = [RAW_DIR / "fungitastic" / "metadata.zip"]
    files += [RAW_DIR / "mushroom_observer" / f"{n}.csv" for n in
              ("observations", "names", "name_classifications", "images_observations", "images", "locations")]
    files += sorted((RAW_DIR / "inaturalist_api" / "api").glob("v001_co_*.json"))
    files += [RAW_DIR / "mind_funga_v2" / "api" / "mendeley_dataset.json", RAW_DIR / "mind_funga_v2" / "zip_central_directory.csv"]
    return files


def describe_inputs(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    registered = pd.read_csv(RAW_FILES_CSV).set_index("relative_path")["sha256"].to_dict()
    out = []
    for p in input_files(cfg):
        h = sha256_file(p)
        out.append({"path": rel(p), "size_bytes": p.stat().st_size, "sha256": h,
                    "matches_raw_files_csv": registered.get(rel(p)) == h})
    return out


# --------------------------------------------------------------------------- dataset card
def card(version: str, cfg: dict[str, Any], res: dict[str, Any], sm: dict[str, pd.DataFrame], manifest: dict[str, Any]) -> str:
    obs, img = res["obs"], res["img"]
    piv_o = obs.pivot_table(index="genus", columns="split", values="observation_uid", aggfunc="count", fill_value=0)
    piv_i = img.pivot_table(index="genus", columns="split", values="image_uid", aggfunc="count", fill_value=0)
    cols = [c for c in ["train", "val", "test", "test_external"] if c in piv_o.columns]
    lines = [f"# Ficha del dataset `{version}`", "",
             "> Archivo generado por `scripts/data/build_dataset.py`; no editar a mano.", "",
             f"* **Huella de contenido:** `{manifest['content_fingerprint']}`",
             f"* **Configuración:** `{manifest['config']['path']}` (SHA-256 `{manifest['config']['sha256'][:16]}…`)",
             f"* **Esquema:** `{manifest['schema']['path']}`",
             f"* **Clases ({len(cfg['classes'])}, nivel {cfg['label_level']}):** " + ", ".join(f"{i}={g}" for i, g in enumerate(cfg["classes"])),
             f"* **Observaciones:** {len(obs):,} · **Imágenes:** {len(img):,}".replace(",", "."), "",
             "## Observaciones por género y partición", "",
             "| Género | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for g in cfg["classes"]:
        lines.append(f"| {g} | " + " | ".join(f"{int(piv_o.loc[g, c]) if g in piv_o.index else 0:,}".replace(",", ".") for c in cols) + " |")
    lines += ["", "## Imágenes por género y partición", "", "| Género | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for g in cfg["classes"]:
        lines.append(f"| {g} | " + " | ".join(f"{int(piv_i.loc[g, c]) if g in piv_i.index else 0:,}".replace(",", ".") for c in cols) + " |")
    lines += ["", "## Fuentes", "", "| Fuente | Papel | Observaciones | Imágenes |", "|---|---|---|---|"]
    for sid, s in cfg["sources"].items():
        lines.append(f"| `{sid}` | {s['role']} | {int((obs['source_id'] == sid).sum()):,} | {int((img['source_id'] == sid).sum()):,} |".replace(",", "."))
    lines += ["", "Fuentes excluidas: " + "; ".join(f"`{k}` ({v})" for k, v in cfg["excluded_sources"].items()) + ".", "",
              "## Controles de fuga (todos deben ser 0)", ""]
    lines += [f"* `{k}`: {v}" for k, v in res["leakage"].items()]
    lines += ["", "## Licencias de las imágenes", "", "| Licencia | Alcance | Imágenes |", "|---|---|---|"]
    for (lic, scope), n in img.groupby(["license", "license_scope"]).size().items():
        lines.append(f"| {lic} | {scope} | {n:,} |".replace(",", "."))
    lines += ["", "Más detalle: `cleaning_log.csv`, `split_summary.csv`, `env_coverage.csv`, `manifest.json`.", ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/dataset_v001.yaml")
    ap.add_argument("--check", action="store_true", help="rebuild and compare with the stored manifest without writing")
    ap.add_argument("--rewrite-manifest", action="store_true",
                    help="if the content is unchanged, rewrite manifest.json (new git commit / timestamp) instead of keeping it")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    setup_logging(args.verbose)
    cfg_path = REPO_ROOT / args.config
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    schema_path = REPO_ROOT / cfg["schema"]
    schema = MS.load_schema(schema_path)
    version = cfg["version"]
    proc_dir, ver_dir = REPO_ROOT / "data" / "processed" / version, REPO_ROOT / "data" / "versions" / version

    res = build(cfg, schema)
    sm = summaries(res, cfg, schema)
    processed = {"observations.csv": to_csv_bytes(res["obs"]), "images.csv": to_csv_bytes(res["img"])}
    versioned = {"splits.csv": to_csv_bytes(sm["splits"]), "classes.csv": to_csv_bytes(sm["classes"]),
                 "cleaning_log.csv": to_csv_bytes(res["log"]), "split_summary.csv": to_csv_bytes(sm["split_summary"]),
                 "env_coverage.csv": to_csv_bytes(sm["env_coverage"])}
    content = {k: hashlib.sha256(v).hexdigest() for k, v in {**processed, **versioned}.items()}
    fingerprint = hashlib.sha256("".join(f"{k}:{content[k]}\n" for k in sorted(content)).encode()).hexdigest()
    log.info("content fingerprint %s | leakage %s", fingerprint, res["leakage"])

    man_path = ver_dir / "manifest.json"
    old = json.loads(man_path.read_text(encoding="utf-8")) if man_path.exists() else None
    if args.check:
        if old is None:
            log.error("no manifest for %s", version)
            return 1
        same = old["content_fingerprint"] == fingerprint
        for k, h in content.items():
            prev = next((o["sha256"] for o in old["outputs"] if o["name"] == k), None)
            log.info("%-18s %s", k, "OK" if prev == h else f"DIFFERS ({prev} != {h})")
        log.info("reproducibility check: %s", "IDENTICAL" if same else "DIFFERENT")
        return 0 if same else 1
    if old is not None and old["content_fingerprint"] != fingerprint:
        log.error("version %s already published with fingerprint %s; the rebuilt content differs (%s). "
                  "Versions are immutable: create a new config/version.", version, old["content_fingerprint"], fingerprint)
        return 1

    proc_dir.mkdir(parents=True, exist_ok=True)
    ver_dir.mkdir(parents=True, exist_ok=True)
    for k, v in processed.items():
        (proc_dir / k).write_bytes(v)
    for k, v in versioned.items():
        (ver_dir / k).write_bytes(v)
    if old is not None and not args.rewrite_manifest:
        log.info("version %s unchanged (same fingerprint): manifest kept", version)
        return 0
    obs, img = res["obs"], res["img"]
    manifest = {
        "version": version, "created_utc": utc_now(), "content_fingerprint": fingerprint,
        "builder": {"command": "python scripts/data/build_dataset.py --config " + args.config, **git_info(),
                    "python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
                    "pyyaml": yaml.__version__},
        "config": {"path": rel(cfg_path), "sha256": sha256_file(cfg_path)},
        "schema": {"path": rel(schema_path), "sha256": sha256_file(schema_path), "schema_id": schema["schema_id"]},
        "code": [{"path": p, "sha256": sha256_file(REPO_ROOT / p)} for p in CODE_FILES],
        "inputs": describe_inputs(cfg),
        "outputs": [{"name": k, "path": rel(proc_dir / k), "versioned_in_git": False, "sha256": content[k],
                     "rows": len(res["obs"] if k.startswith("obs") else res["img"])} for k in processed]
                   + [{"name": k, "path": rel(ver_dir / k), "versioned_in_git": True, "sha256": content[k]} for k in versioned],
        "counts": {"observations": len(obs), "images": len(img),
                   "observations_by_split": obs["split"].value_counts().sort_index().to_dict(),
                   "images_by_split": img["split"].value_counts().sort_index().to_dict(),
                   "observations_by_source": obs["source_id"].value_counts().sort_index().to_dict(),
                   "images_by_source": img["source_id"].value_counts().sort_index().to_dict(),
                   "groups_development": int(obs.loc[obs["split"] != "test_external", "group_id"].nunique())},
        "leakage_checks": res["leakage"],
        "dataset_level_licenses": {"fungitastic": "Declaraciones inconsistentes (artículo CC BY 4.0; Kaggle CC BY-NC-SA 4.0; "
                                                  "registros GBIF CC BY-NC 4.0). Se registra la más restrictiva: CC-BY-NC-SA.",
                                   "mind_funga_v2": "CC BY-NC 3.0 (Mendeley Data)."},
        "known_limitations": ["No se descargaron imágenes: las reglas de integridad de imagen (decodificación, tamaño mínimo, "
                              "hash perceptual) están definidas en image_policy pero no ejecutadas.",
                              "Clima no disponible (climatic.zip de FungiTastic responde HTTP 403 el 2026-09-26).",
                              "Las consultas de iNaturalist son instantáneas del " + cfg["access_date"] + "."],
    }
    man_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (ver_dir / "DATASET_CARD.md").write_text(card(version, cfg, res, sm, manifest), encoding="utf-8")
    log.info("wrote %s and %s", rel(proc_dir), rel(ver_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
