"""Dataset-agnostic audit metrics (size, missing values, duplicates, class imbalance).

Nothing here modifies data: every function takes a DataFrame that was just loaded from raw files and returns
summary numbers. Each dataset adapter (``adapters.py``) maps its own column names onto the common *roles* below.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

# Roles a table may provide (role -> column name in that table)
ROLES = [
    "observation_id", "image_id", "image_file", "url", "taxon_label", "species", "genus", "family", "order",
    "date", "latitude", "longitude", "country", "region", "substrate", "habitat", "climate", "license",
    "author", "provider",
]
TAXON_ROLES = ["species", "genus", "family", "order"]

# Strings that are used as "missing" markers by some sources (compared after strip + lower-case)
SPECIAL_MISSING = {"", "na", "n/a", "nan", "null", "none", "unknown", "unspecified", "?", "-", "--", "not recorded"}

EVIDENCE = ("SOURCE_REPORTED", "LOCAL_VERIFIED", "DERIVED", "NOT_VERIFIED")


@dataclass
class AuditTable:
    """One auditable table: a DataFrame + the mapping of common roles to its columns."""
    dataset_id: str
    table: str
    df: pd.DataFrame
    roles: dict[str, str]
    grain: str = "image"                # what one row is: image | observation | record | file
    evidence: str = "LOCAL_VERIFIED"    # LOCAL_VERIFIED for full files, DERIVED/NOT_VERIFIED otherwise
    is_sample: bool = False             # True if the table is a sample/fragment (never the full source)
    main: bool = True                   # False: only reported in the size summary (per-file detail tables)
    note: str = ""
    extra_checks: dict[str, Any] = field(default_factory=dict)   # dataset-specific findings

    def col(self, role: str) -> pd.Series | None:
        c = self.roles.get(role)
        return self.df[c] if c and c in self.df.columns else None


# --------------------------------------------------------------------------- helpers
def missing_mask(s: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return boolean masks (is_null, is_empty_string, is_special_token) for a column."""
    is_null = s.isna()
    if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        z = pd.Series(False, index=s.index)
        return is_null, z, z
    st = s.astype("string").str.strip()
    is_empty = st.eq("").fillna(False)
    is_special = st.str.lower().isin(SPECIAL_MISSING).fillna(False) & ~is_empty
    return is_null, is_empty, is_special


def _present(s: pd.Series) -> pd.Series:
    """Values that are neither null, empty nor special missing markers."""
    n, e, sp = missing_mask(s)
    return s[~(n | e | sp)]


def nunique_present(s: pd.Series | None) -> int | None:
    return None if s is None else int(_present(s).nunique())


ISO_DATE = r"^\d{4}(?:-\d{2}(?:-\d{2})?)?(?:[T ].*)?(?:/.*)?$"   # ISO 8601 (also intervals 'a/b') as Darwin Core expects


def suspicious_values(role: str, s: pd.Series, table: AuditTable) -> tuple[int, int, str]:
    """Rule-based counts on NON-missing values: (implausible values, non-standard format values, rule text)."""
    s = _present(s)
    if role == "latitude":
        v = pd.to_numeric(s, errors="coerce")
        lon = table.col("longitude")
        lon = pd.to_numeric(lon.loc[s.index], errors="coerce") if lon is not None else pd.Series(np.nan, index=s.index)
        both0 = int((v.eq(0) & lon.eq(0)).sum())
        unparsable = int(v.isna().sum())
        return int(((v < -90) | (v > 90)).sum() + both0 + unparsable), 0, "lat outside [-90,90], unparsable, or (lat,lon)==(0,0)"
    if role == "longitude":
        v = pd.to_numeric(s, errors="coerce")
        return int(((v < -180) | (v > 180)).sum() + v.isna().sum()), 0, "lon outside [-180,180] or unparsable"
    if role == "date":
        st = s.astype("string").str.strip()
        iso = st.str.match(ISO_DATE).fillna(False)
        yr_iso = pd.to_datetime(st[iso], errors="coerce", utc=True, format="mixed").dt.year
        yr_other = pd.to_datetime(st[~iso], errors="coerce", utc=True, dayfirst=True, format="mixed").dt.year
        now = pd.Timestamp.now().year
        bad = int(yr_iso.isna().sum() + ((yr_iso < 1800) | (yr_iso > now)).sum() + yr_other.isna().sum() + ((yr_other < 1800) | (yr_other > now)).sum())
        return bad, int((~iso).sum()), "date unparsable or year outside [1800, current]; non-standard = not ISO 8601"
    return 0, 0, ""


# --------------------------------------------------------------------------- size
def size_summary(t: AuditTable) -> dict[str, Any]:
    df = t.df
    out: dict[str, Any] = {"dataset_id": t.dataset_id, "table": t.table, "grain": t.grain,
                           "is_sample": t.is_sample, "evidence": t.evidence, "n_rows": len(df)}
    obs, img = t.col("observation_id"), t.col("image_id")
    if img is None:
        img = t.col("image_file")
    out["n_images"] = int(_present(img).nunique()) if img is not None else (len(df) if t.grain == "image" else None)
    out["n_observations"] = (int(_present(obs).nunique()) if obs is not None
                             else (len(df) if t.grain in ("observation", "record") else None))
    for role, key in (("species", "n_species"), ("genus", "n_genera"), ("family", "n_families"), ("order", "n_orders")):
        out[key] = nunique_present(t.col(role))
    out["observation_id_available"] = obs is not None
    n_img_present = int(_present(img).nunique()) if img is not None else 0
    if obs is not None and img is not None and n_img_present > 0:
        pm = pd.Series(~(missing_mask(img)[0] | missing_mask(img)[1] | missing_mask(img)[2]).values, index=df.index)
        per = df.assign(_o=obs, _i=img)[pm].dropna(subset=["_o", "_i"]).groupby("_o")["_i"].nunique()
        out["images_per_observation_mean"] = round(float(per.mean()), 3)
        out["images_per_observation_median"] = float(per.median())
        out["images_per_observation_max"] = int(per.max())
        out["observations_with_1_image"] = int((per == 1).sum())
        out["observations_with_gt1_images"] = int((per > 1).sum())
    else:
        for k in ("images_per_observation_mean", "images_per_observation_median", "images_per_observation_max",
                  "observations_with_1_image", "observations_with_gt1_images"):
            out[k] = None
    out["note"] = t.note
    return out


# --------------------------------------------------------------------------- missing values
def missing_profile(t: AuditTable) -> pd.DataFrame:
    rows = []
    n = len(t.df)
    for role in ROLES:
        c = t.roles.get(role)
        if not c or c not in t.df.columns:
            rows.append({"dataset_id": t.dataset_id, "table": t.table, "role": role, "column": None,
                         "column_present": False, "n_rows": n, "n_null": None, "n_empty_string": None,
                         "n_special_token": None, "n_missing_total": None, "pct_missing": None,
                         "n_suspicious": None, "n_nonstandard_format": None, "suspicious_rule": ""})
            continue
        s = t.df[c]
        is_null, is_empty, is_special = missing_mask(s)
        total = int((is_null | is_empty | is_special).sum())
        sus, nonstd, rule = suspicious_values(role, s, t)
        rows.append({"dataset_id": t.dataset_id, "table": t.table, "role": role, "column": c, "column_present": True,
                     "n_rows": n, "n_null": int(is_null.sum()), "n_empty_string": int(is_empty.sum()),
                     "n_special_token": int(is_special.sum()), "n_missing_total": total,
                     "pct_missing": round(100 * total / n, 3) if n else None, "n_suspicious": sus,
                     "n_nonstandard_format": nonstd, "suspicious_rule": rule})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- duplicates
def duplicate_profile(t: AuditTable) -> dict[str, Any]:
    """Counts of duplicated keys. Never removes anything.

    ``n_exact_duplicate_rows`` is computed on the loaded columns only (see table note).
    """
    df = t.df
    out: dict[str, Any] = {"dataset_id": t.dataset_id, "table": t.table, "n_rows": len(df), "is_sample": t.is_sample}
    def _hashable(c: str) -> bool:
        first = df[c].dropna().head(1)
        return first.empty or not isinstance(first.iloc[0], (list, dict, set))

    hashable = df[[c for c in df.columns if _hashable(c)]]
    out["n_exact_duplicate_rows"] = int(hashable.duplicated().sum())
    for role, key in (("image_id", "image_id"), ("image_file", "image_file"), ("url", "url")):
        s = t.col(role)
        if s is None:
            out[f"n_rows_with_repeated_{key}"] = None
            out[f"n_distinct_repeated_{key}"] = None
            continue
        p = _present(s)
        vc = p.value_counts()
        out[f"n_rows_with_repeated_{key}"] = int(vc[vc > 1].sum())          # every row whose key appears >1 times
        out[f"n_distinct_repeated_{key}"] = int((vc > 1).sum())
    obs = t.col("observation_id")
    if obs is not None:
        vc = _present(obs).value_counts()
        out["n_observation_ids"] = int(len(vc))
        out["n_observation_ids_with_multiple_rows"] = int((vc > 1).sum())  # expected when grain == image
        sp = t.col("species")
        if sp is not None:
            g = df.assign(_o=obs, _s=sp).dropna(subset=["_o", "_s"]).groupby("_o")["_s"].nunique()
            out["n_observations_with_conflicting_species"] = int((g > 1).sum())
        else:
            out["n_observations_with_conflicting_species"] = None
    else:
        out["n_observation_ids"] = None
        out["n_observation_ids_with_multiple_rows"] = None
        out["n_observations_with_conflicting_species"] = None
    out["sha256_duplicates"] = "NOT_VERIFIED (image files not downloaded)"
    out.update(t.extra_checks)
    return out


# --------------------------------------------------------------------------- class imbalance
def counts_by_taxon(t: AuditTable, level: str) -> pd.DataFrame | None:
    """n_images / n_observations per taxon at ``level`` (species | genus | family). None if the level is unavailable.

    Rows with a missing taxon are excluded from the counts (they are reported in the missing-value table).
    """
    tx = t.col(level)
    if tx is None:
        return None
    obs, img = t.col("observation_id"), t.col("image_id")
    if img is None:
        img = t.col("image_file")
    d = pd.DataFrame({"taxon": tx.values})
    if obs is not None:
        d["obs"] = obs.values
    if t.grain == "image":
        d["img"] = img.values if img is not None else np.arange(len(d))
    n, e, sp = missing_mask(d["taxon"])
    d = d[~(n | e | sp)]
    g = d.groupby("taxon")
    out = pd.DataFrame(index=g.size().index)
    out["n_images"] = g["img"].nunique() if "img" in d else np.nan
    if "obs" in d:
        out["n_observations"] = g["obs"].nunique()
    else:
        out["n_observations"] = g.size() if t.grain in ("observation", "record") else np.nan
    out = out.reset_index()
    out.insert(0, "level", level)
    out.insert(0, "table", t.table)
    out.insert(0, "dataset_id", t.dataset_id)
    return out.sort_values(["n_images", "n_observations"], ascending=False, na_position="last").reset_index(drop=True)


def imbalance_stats(counts: pd.Series) -> dict[str, Any]:
    """Long-tail descriptors of a vector of per-class counts."""
    c = counts.dropna().astype(float)
    c = c[c > 0]
    if c.empty:
        return {}
    s = np.sort(c.values)[::-1]
    cum = np.cumsum(s) / s.sum()
    top10 = max(1, int(np.ceil(0.10 * len(s))))
    n = len(s)
    gini = float((2 * np.arange(1, n + 1) - n - 1).dot(np.sort(s)) / (n * s.sum())) if n > 1 else 0.0
    return {
        "n_classes": n, "total": int(s.sum()), "min": int(s.min()), "max": int(s.max()),
        "mean": round(float(s.mean()), 2), "median": float(np.median(s)),
        "p10": float(np.percentile(s, 10)), "p25": float(np.percentile(s, 25)),
        "p75": float(np.percentile(s, 75)), "p90": float(np.percentile(s, 90)),
        "p95": float(np.percentile(s, 95)), "p99": float(np.percentile(s, 99)),
        "ratio_max_min": round(float(s.max() / s.min()), 1),
        "n_classes_eq1": int((s == 1).sum()), "n_classes_lt5": int((s < 5).sum()),
        "n_classes_lt10": int((s < 10).sum()), "n_classes_lt20": int((s < 20).sum()),
        "pct_classes_lt10": round(100 * float((s < 10).mean()), 2),
        "pct_samples_in_top10pct_classes": round(100 * float(cum[top10 - 1]), 2),
        "n_classes_covering_50pct_samples": int(np.searchsorted(cum, 0.5) + 1),
        "gini": round(gini, 4),
    }
