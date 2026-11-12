"""Build ``ext_full_001``: the complete ``test_external`` of ``v001`` (the three external sources) for evaluation.

``ext_full_001`` = the Colombian set ``ext_co_001`` (iNaturalist CO + Mushroom Observer CO, with department and
region, unchanged) + the ``mind_funga_v2`` observations of ``v001``'s ``test_external``. Nothing is re-derived for the
Colombian part, so its rows are byte-identical to ``ext_co_001`` and its results can be compared directly.

MIND.Funga v2 (Mendeley Data, DOI 10.17632/sfrbdjvxcc.2) has no coordinates, date or country: its observations get
``region = no_location``. Each image keeps the SHA-256 published by the source (``sha256_reported``), which the image
downloader checks.

Outputs:
    data/processed/ext_full_001/observations.csv, images.csv      (ignored by Git, reproducible)
    data/versions/ext_full_001/manifest.json, classes.csv, DATASET_CARD.md

    python scripts/data/build_external_full.py
    python scripts/data/build_external_full.py --check
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO_ROOT, rel, setup_logging, sha256_file, utc_now  # noqa: E402

SET_ID = "ext_full_001"
BASE_VERSION = "v001"
CO_SET = "ext_co_001"
MIND = "mind_funga_v2"
MIND_DOI = "https://doi.org/10.17632/sfrbdjvxcc.2"
VERSIONS = REPO_ROOT / "data" / "versions"
PROCESSED = REPO_ROOT / "data" / "processed"
VER_OUT, PROC_OUT = VERSIONS / SET_ID, PROCESSED / SET_ID


def verified(name: str, files: tuple[str, ...]) -> dict:
    man = json.loads((VERSIONS / name / "manifest.json").read_text(encoding="utf-8"))
    for f in files:
        expected = next(o["sha256"] for o in man["outputs"] if o["name"] == f)
        if sha256_file(PROCESSED / name / f) != expected:
            raise SystemExit(f"data/processed/{name}/{f} does not match data/versions/{name}/manifest.json")
    return man


def csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False, lineterminator="\n")
    return buf.getvalue().encode("utf-8")


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def build() -> dict:
    v001 = verified(BASE_VERSION, ("observations.csv", "images.csv"))
    co_man = verified(CO_SET, ("observations.csv", "images.csv"))
    co_obs = pd.read_csv(PROCESSED / CO_SET / "observations.csv", dtype=str, keep_default_na=False)
    co_img = pd.read_csv(PROCESSED / CO_SET / "images.csv", dtype=str, keep_default_na=False)
    obs = pd.read_csv(PROCESSED / BASE_VERSION / "observations.csv", dtype=str, keep_default_na=False, low_memory=False)
    img = pd.read_csv(PROCESSED / BASE_VERSION / "images.csv", dtype=str, keep_default_na=False, low_memory=False)
    ext = obs[obs["split"] == "test_external"]
    m_obs = ext[ext["source_id"] == MIND].copy()
    m_img = img[img["observation_uid"].isin(m_obs["observation_uid"])].copy()

    m_obs = m_obs.assign(record_url=MIND_DOI, department="", department_raw="", region="no_location",
                         region_basis="not_provided", gbif_id="")[co_obs.columns]
    m_img = m_img.assign(region="no_location")
    co_img = co_img.assign(sha256_reported="")
    img_cols = list(co_img.columns)
    all_obs = pd.concat([co_obs, m_obs], ignore_index=True).sort_values("observation_uid", kind="stable")
    all_img = pd.concat([co_img, m_img[img_cols]], ignore_index=True).sort_values("image_uid", kind="stable")

    # The set must be exactly v001's test_external: same observations and images, no more, no less.
    want_obs, want_img = set(ext["observation_uid"]), set(img.loc[img["split"] == "test_external", "image_uid"])
    if set(all_obs["observation_uid"]) != want_obs or set(all_img["image_uid"]) != want_img:
        raise SystemExit("ext_full_001 does not match v001's test_external")
    if (m_img["sha256_reported"] == "").any():
        raise SystemExit("MIND.Funga images without the SHA-256 published by the source")

    classes = (VERSIONS / BASE_VERSION / "classes.csv").read_bytes()
    tables = {"observations.csv": csv_bytes(all_obs), "images.csv": csv_bytes(all_img), "classes.csv": classes}
    outputs = [{"name": n, "sha256": sha(b), "rows": int(b.count(b"\n") - 1)} for n, b in tables.items()]
    fingerprint = sha("".join(f"{o['name']}:{o['sha256']}\n" for o in sorted(outputs, key=lambda o: o["name"])).encode())
    here = Path(__file__)
    manifest = {
        "set_id": SET_ID, "kind": "external_evaluation_set", "created_utc": utc_now(),
        "content_fingerprint": fingerprint,
        "derived_from": {"version": BASE_VERSION, "content_fingerprint": v001["content_fingerprint"],
                         "split": "test_external", "sources": sorted(ext["source_id"].unique().tolist())},
        "colombian_part": {"set_id": CO_SET, "content_fingerprint": co_man["content_fingerprint"],
                           "note": "rows copied unchanged from ext_co_001"},
        "region_definition": co_man["region_definition"] | {"no_location": f"{MIND}: no coordinates, date or country"},
        "classes": co_man["classes"],
        "code": {rel(here): sha256_file(here)},
        "inputs": [{"path": rel(p), "sha256": sha256_file(p)} for p in
                   (PROCESSED / BASE_VERSION / "observations.csv", PROCESSED / BASE_VERSION / "images.csv",
                    PROCESSED / CO_SET / "observations.csv", PROCESSED / CO_SET / "images.csv",
                    VERSIONS / BASE_VERSION / "manifest.json", VERSIONS / CO_SET / "manifest.json")],
        "outputs": outputs,
        "counts": {"observations": len(all_obs), "images": len(all_img),
                   "observations_by_source": all_obs["source_id"].value_counts().sort_index().to_dict(),
                   "images_by_source": all_img["source_id"].value_counts().sort_index().to_dict(),
                   "observations_by_region": all_obs["region"].value_counts().sort_index().to_dict()},
    }
    return {"obs": all_obs, "img": all_img, "tables": tables, "manifest": manifest}


def card(res: dict) -> str:
    obs, man = res["obs"], res["manifest"]
    mil = lambda n: f"{int(n):,}".replace(",", ".")  # noqa: E731
    classes = man["classes"]
    piv = obs.pivot_table(index="genus", columns="source_id", values="observation_uid", aggfunc="count", fill_value=0)
    srcs = ["inaturalist_co", "mushroom_observer_co", MIND]
    piv = piv.reindex(index=classes, columns=srcs, fill_value=0)
    c = man["counts"]
    L = [f"# Conjunto de evaluación externa `{SET_ID}` (test_external completo de `{BASE_VERSION}`)", "",
         "> Generado por `scripts/data/build_external_full.py`; no editar a mano.", "",
         f"* **Huella de contenido:** `{man['content_fingerprint']}`",
         f"* **Derivado de:** `{BASE_VERSION}` (`test_external` completo, huella "
         f"`{man['derived_from']['content_fingerprint'][:16]}…`), sin cambiar etiquetas, imágenes ni filtros.",
         f"* **Parte colombiana:** copiada sin cambios de [`{CO_SET}`](../{CO_SET}/DATASET_CARD.md) "
         "(departamento y región, incluida la Orinoquía).",
         f"* **Observaciones:** {mil(c['observations'])} · **Imágenes:** {mil(c['images'])}", "",
         "| Fuente | Observaciones | Imágenes | Etiqueta | Región |", "|---|---:|---:|---|---|",
         f"| `inaturalist_co` | {mil(c['observations_by_source']['inaturalist_co'])} | "
         f"{mil(c['images_by_source']['inaturalist_co'])} | comunidad, grado de investigación | departamento |",
         f"| `mushroom_observer_co` | {mil(c['observations_by_source']['mushroom_observer_co'])} | "
         f"{mil(c['images_by_source']['mushroom_observer_co'])} | consenso de votos ≥ 1 | departamento |",
         f"| `{MIND}` | {mil(c['observations_by_source'][MIND])} | {mil(c['images_by_source'][MIND])} | "
         "nombre de carpeta | `no_location` (sin coordenadas, fecha ni país) |", "",
         f"MIND.Funga v2 ({MIND_DOI}): imágenes de macrohongos neotropicales de origen mixto; una imagen por "
         "observación. Cada imagen conserva el SHA-256 publicado por la fuente (`sha256_reported`).", "",
         "## Observaciones por género y fuente", "",
         "| Género | iNaturalist CO | Mushroom Observer CO | MIND.Funga | Total |", "|---|---:|---:|---:|---:|"]
    L += [f"| {g} | " + " | ".join(mil(r[s]) for s in srcs) + f" | {mil(r.sum())} |" for g, r in piv.iterrows()]
    L += ["", "## Uso", "", "Solo evaluación de cambio de dominio. Imágenes: "
          f"`python scripts/data/download_external_images.py --set {SET_ID}`. Evaluación: "
          f"`python scripts/train/evaluate_external.py --set {SET_ID} --run runs/<run_id>`.", ""]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    setup_logging()
    res = build()
    man = res["manifest"]
    published = VER_OUT / "manifest.json"
    if published.exists():
        old = json.loads(published.read_text(encoding="utf-8"))
        same = old["content_fingerprint"] == man["content_fingerprint"]
        if args.check:
            print("OK: identical to the published set" if same else "DIFFERENT from the published set")
            return 0 if same else 1
        if not same:
            raise SystemExit(f"{SET_ID} is published with another content; create a new set version")
        man["created_utc"] = old["created_utc"]
    elif args.check:
        raise SystemExit(f"{SET_ID} is not published yet")
    PROC_OUT.mkdir(parents=True, exist_ok=True)
    VER_OUT.mkdir(parents=True, exist_ok=True)
    for n in ("observations.csv", "images.csv"):
        (PROC_OUT / n).write_bytes(res["tables"][n])
    (VER_OUT / "classes.csv").write_bytes(res["tables"]["classes.csv"])
    published.write_text(json.dumps(man, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (VER_OUT / "DATASET_CARD.md").write_text(card(res), encoding="utf-8")
    print(json.dumps(man["counts"], indent=2, ensure_ascii=False))
    print(f"fingerprint {man['content_fingerprint']}\n{rel(VER_OUT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
