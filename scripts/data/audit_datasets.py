"""Week 2 dataset audit: inventory, missing values, duplicates, class imbalance.

Reads ONLY local raw files / stored API snapshots (see ``acquire_datasets.py``) and writes small, versionable
results to ``reports/audit/``, ``reports/figures/`` and ``data/manifests/dataset_inventory.csv``.
It does not modify, clean, deduplicate or split any data (those are Week 3 tasks).

Usage:
    python scripts/data/audit_datasets.py                    # audit every source (~2 min)
    python scripts/data/audit_datasets.py df20 gbif          # only some sources
    python scripts/data/audit_datasets.py --list             # show source names

When only some sources are audited the outputs of the others are kept (rows are replaced per dataset_id).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adapters as A  # noqa: E402
import audit_core as C  # noqa: E402
from common import AUDIT_DIR, MANIFEST_DIR, REPO_ROOT, SOURCES_DIR, log, setup_logging, utc_now  # noqa: E402
from inventory import build_inventory, build_metadata_matrix, write_source_manifests  # noqa: E402

FIGURES_DIR = REPO_ROOT / "reports" / "figures"

ADAPTERS: dict[str, Callable[[], A.AdapterResult]] = {
    "fungitastic": A.audit_fungitastic,
    "df20": A.audit_df20,
    "mind_funga_v2": A.audit_mind_funga,
    "inaturalist_open_data": A.audit_inaturalist_open_data,
    "inaturalist_api": A.audit_inaturalist_api,
    "sib_macrohongosmitu": lambda: A.audit_sib_resource("sib_macrohongosmitu"),
    "sib_macrohongos_jbb_2023_1": lambda: A.audit_sib_resource("sib_macrohongos_jbb_2023_1"),
    "sib_hongos_col2023": lambda: A.audit_sib_resource("sib_hongos_col2023"),
    "gbif": A.audit_gbif,
    "mushroom_observer": A.audit_mushroom_observer,
}
OUTPUTS = {
    "size": "size_summary.csv", "missing": "missing_values.csv", "duplicates": "duplicates.csv",
    "imbalance": "imbalance_summary.csv", "classes": "class_distribution.csv", "overlaps": "overlaps.csv",
    "remote": "remote_counts.csv", "facts": "audit_facts.json",
}


def _merge_write(name: str, new: pd.DataFrame, datasets: list[str]) -> pd.DataFrame:
    """Replace the rows of ``datasets`` in the existing CSV so partial re-runs keep the other results."""
    path = AUDIT_DIR / name
    if path.exists() and datasets:
        old = pd.read_csv(path)
        if "dataset_id" in old.columns:
            old = old[~old["dataset_id"].isin(datasets)]
            new = pd.concat([old, new], ignore_index=True)
    new.to_csv(path, index=False, lineterminator="\n")
    return new


def classes_for(t: C.AuditTable) -> list[pd.DataFrame]:
    out = []
    levels = ["species", "genus", "family"]
    for lv in levels:
        d = C.counts_by_taxon(t, lv)
        if d is not None and len(d):
            out.append(d)
    if t.dataset_id == "mind_funga_v2":     # folder = class in this dataset
        d = C.counts_by_taxon(C.AuditTable(t.dataset_id, t.table, t.df, {**t.roles, "species": t.roles["taxon_label"]}, t.grain), "species")
        if d is not None:
            d["level"] = "label_folder"
            out.append(d)
    return out


def imbalance_rows(cd: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for (ds, tb, lv), g in cd.groupby(["dataset_id", "table", "level"], sort=False):
        for unit in ("n_images", "n_observations"):
            if g[unit].notna().any():
                rows.append({"dataset_id": ds, "table": tb, "level": lv, "unit": unit.replace("n_", ""), **C.imbalance_stats(g[unit])})
    return rows


def flatten_counts(res: A.AdapterResult) -> list[dict[str, Any]]:
    """Scalar counts reported by remote APIs (SOURCE_REPORTED at query time) -> long table."""
    rows = []
    if res.dataset_id not in ("gbif", "inaturalist_api"):
        return rows
    for k, v in res.facts.items():
        if isinstance(v, (int, np.integer)) and not isinstance(v, bool):
            rows.append({"dataset_id": res.dataset_id, "metric": k, "value": int(v), "evidence": "SOURCE_REPORTED (API count at query date)"})
        elif isinstance(v, dict) and v and all(isinstance(x, (int, np.integer)) for x in v.values()) and k.endswith(("counts", "_research", "stillimage")):
            for kk, vv in v.items():
                rows.append({"dataset_id": res.dataset_id, "metric": f"{k}.{kk}", "value": int(vv), "evidence": "SOURCE_REPORTED (API count at query date)"})
    if res.dataset_id == "gbif":
        for grp in ("top_datasets_colombia_stillimage", "top_datasets_published_in_colombia_stillimage"):
            for d in res.facts.get(grp, []):
                rows.append({"dataset_id": "gbif", "metric": f"{grp}.{d['title'] or d['datasetKey']}", "value": d["records"],
                             "evidence": "SOURCE_REPORTED (API facet at query date)"})
    return rows


def cross_overlaps(results: dict[str, A.AdapterResult]) -> list[dict[str, Any]]:
    """DF20 vs FungiTastic overlap by shared observation id (GBIF occurrence key). Read-only set intersections."""
    rows = []
    ft, df = results.get("fungitastic"), results.get("df20")
    if ft and df and ft.id_sets and df.id_sets:
        for a in ("df20:train", "df20:public_test", "df20:union"):
            for b in ("fungitastic:Train", "fungitastic:ClosedSet-Val", "fungitastic:ClosedSet-Test", "fungitastic:DNA-Test", "fungitastic:full_union"):
                sa, sb = df.id_sets[a], ft.id_sets[b]
                rows.append({"scope": "between datasets (DF20 vs FungiTastic)", "set_a": a, "set_b": b, "key": "gbifID == observationID",
                             "n_a": len(sa), "n_b": len(sb), "n_intersection": len(sa & sb),
                             "note": "identificadores de ocurrencia de GBIF (verificado con 1 ID de cada CSV)"})
    g = results.get("gbif")
    if g:
        f = g.facts
        for title, n in [(d["title"], d["records"]) for d in f.get("top_datasets_colombia_stillimage", [])
                         if d["title"] and ("iNaturalist" in d["title"] or "Mushroom Observer" in d["title"])]:
            rows.append({"scope": "aggregator overlap (GBIF)", "set_a": "gbif: Fungi+StillImage+country=CO", "set_b": f"gbif dataset '{title}'",
                         "key": "GBIF datasetKey facet", "n_a": f.get("fungi_colombia_stillimage"), "n_b": n, "n_intersection": n,
                         "note": "SOURCE_REPORTED: registros de GBIF que provienen de otra plataforma auditada aparte (riesgo de doble conteo)"})
    return rows


def save_figure(cd: pd.DataFrame) -> None:
    """Rank-abundance (long-tail) curves per dataset at species level, in images (or observations if no image count)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sel = [("fungitastic", "full_union"), ("df20", "train+public_test"), ("mind_funga_v2", "files_v2"),
           ("mushroom_observer", "fungi_all_images"), ("mushroom_observer", "fungi_colombia_images")]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for ds, tb in sel:
        lv = "label_folder" if ds == "mind_funga_v2" else "species"
        g = cd[(cd.dataset_id == ds) & (cd.table == tb) & (cd.level == lv)]
        if g.empty:
            continue
        v = np.sort(g["n_images"].dropna().values)[::-1]
        ax.plot(np.arange(1, len(v) + 1), v, label=f"{ds}/{tb} ({len(v):,} clases)")
    for ds, tb, unit in (("gbif", "colombia_stillimage_records", "n_observations"), ("inaturalist_api", "species_counts_research_grade", "n_observations")):
        g = cd[(cd.dataset_id == ds) & (cd.table == tb) & (cd.level == "species")]
        if not g.empty:
            v = np.sort(g[unit].dropna().values)[::-1]
            ax.plot(np.arange(1, len(v) + 1), v, ls="--", label=f"{ds} Colombia, registros ({len(v):,} especies)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("rango de la especie (log)"); ax.set_ylabel("imágenes por especie (log)")
    ax.set_title("Cola larga: imágenes (o registros) por especie")
    ax.grid(True, which="both", alpha=0.25); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "long_tail_species.png", dpi=130); plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sources", nargs="*", choices=list(ADAPTERS), help="subset of sources (default: all)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    if args.list:
        print("\n".join(ADAPTERS)); return 0
    setup_logging(args.verbose)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    selected = args.sources or list(ADAPTERS)

    results: dict[str, A.AdapterResult] = {}
    failed = []
    for name in selected:
        log.info("=== auditing %s", name)
        try:
            results[name] = ADAPTERS[name]()
            for p in results[name].problems:
                log.warning("%s: %s", name, p)
        except Exception as exc:  # noqa: BLE001
            log.exception("audit of %s FAILED: %s", name, exc)
            failed.append(name)

    size, missing, dups, classes, overlaps, remote = [], [], [], [], [], []
    for name, res in results.items():
        for t in res.tables:
            size.append(C.size_summary(t))
            if not t.main:
                continue
            missing.append(C.missing_profile(t))
            dups.append(C.duplicate_profile(t))
            classes += classes_for(t)
        classes += res.class_counts
        overlaps += res.overlaps
        remote += flatten_counts(res)
    overlaps += cross_overlaps(results)

    ds_run = list(results)
    size_df = _merge_write(OUTPUTS["size"], pd.DataFrame(size), ds_run)
    if missing:
        _merge_write(OUTPUTS["missing"], pd.concat(missing, ignore_index=True), ds_run)
    if dups:
        dup_df = pd.DataFrame(dups)
        # keep dataset-specific keys as JSON so the CSV has a stable schema
        core = ["dataset_id", "table", "n_rows", "is_sample", "n_exact_duplicate_rows", "n_rows_with_repeated_image_id",
                "n_distinct_repeated_image_id", "n_rows_with_repeated_image_file", "n_distinct_repeated_image_file",
                "n_rows_with_repeated_url", "n_distinct_repeated_url", "n_observation_ids", "n_observation_ids_with_multiple_rows",
                "n_observations_with_conflicting_species", "sha256_duplicates"]
        for c in core:
            if c not in dup_df.columns:
                dup_df[c] = None
        dup_df["dataset_specific_checks"] = dup_df.drop(columns=core).apply(
            lambda r: json.dumps({k: (None if pd.isna(v) else (v.item() if hasattr(v, "item") else v)) for k, v in r.items() if not (isinstance(v, float) and np.isnan(v))}, ensure_ascii=False), axis=1)
        _merge_write(OUTPUTS["duplicates"], dup_df[core + ["dataset_specific_checks"]], ds_run)
    if classes:
        new_cd = pd.concat(classes, ignore_index=True)
        _merge_write(OUTPUTS["imbalance"], pd.DataFrame(imbalance_rows(new_cd)), ds_run)   # includes the family level
        # family-level counts and the (redundant) all-fungi observation table are only summarised in imbalance_summary.csv
        slim = new_cd[(new_cd["level"] != "family") & (new_cd["table"] != "fungi_all_observations")].copy()
        for c in ("n_images", "n_observations"):
            slim[c] = slim[c].round().astype("Int64")
        cd = _merge_write(OUTPUTS["classes"], slim, ds_run)
        save_figure(cd)
    ov_path = AUDIT_DIR / OUTPUTS["overlaps"]
    if overlaps or ov_path.exists():
        new_ov = pd.DataFrame(overlaps, columns=["scope", "set_a", "set_b", "key", "n_a", "n_b", "n_intersection", "note"])
        new_ov.insert(0, "dataset_id", new_ov["set_a"].str.split(":").str[0])
        if ov_path.exists():   # partial re-runs keep the rows of the sources that were not audited again
            old = pd.read_csv(ov_path)
            if "dataset_id" not in old.columns:
                old.insert(0, "dataset_id", old["set_a"].str.split(":").str[0])
            keep = old[~old["dataset_id"].isin(ds_run)]
            between_old = old[old["scope"].str.startswith("between")]
            keep = keep[~keep["scope"].str.startswith("between")]
            if not ("fungitastic" in results and "df20" in results):
                keep = pd.concat([keep, between_old]).drop_duplicates()
            new_ov = pd.concat([keep, new_ov], ignore_index=True)
        new_ov.to_csv(ov_path, index=False, lineterminator="\n")
    if remote:
        _merge_write(OUTPUTS["remote"], pd.DataFrame(remote), ds_run)

    facts_path = AUDIT_DIR / OUTPUTS["facts"]
    facts = json.loads(facts_path.read_text(encoding="utf-8")) if facts_path.exists() else {}
    facts["_generated_utc"] = utc_now()
    for name, res in results.items():
        facts[name] = {"facts": res.facts, "problems": res.problems}
    facts_path.write_text(json.dumps(facts, ensure_ascii=False, indent=1, default=str), encoding="utf-8")

    # inventory + metadata matrix (need every source: rebuilt from the stored outputs/facts)
    all_facts = {k: v["facts"] for k, v in facts.items() if not k.startswith("_")}
    inv = build_inventory(all_facts, pd.read_csv(AUDIT_DIR / OUTPUTS["size"]),
                          pd.read_csv(AUDIT_DIR / OUTPUTS["missing"]), pd.read_csv(AUDIT_DIR / OUTPUTS["duplicates"]))
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    inv.to_csv(MANIFEST_DIR / "dataset_inventory.csv", index=False, lineterminator="\n")
    write_source_manifests(inv, SOURCES_DIR)
    build_metadata_matrix(inv, pd.read_csv(AUDIT_DIR / OUTPUTS["missing"])).to_csv(AUDIT_DIR / "metadata_matrix.csv", index=False, lineterminator="\n")
    log.info("wrote %d inventory rows; failed sources: %s", len(inv), failed or "none")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
