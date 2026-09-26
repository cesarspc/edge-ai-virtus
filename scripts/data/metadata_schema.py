"""Load ``data/schemas/metadata_schema_v1.yaml`` and validate the processed tables against it (Week 3).

The schema is the single source of truth for column names, vocabularies and the provenance triple
(``<group>__source``, ``<group>__method``, ``<group>__quality``) that every environmental group carries.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

PROV = ("source", "method", "quality")
# qualities meaning "there is no value" / "there is a value"
EMPTY_QUALITIES = {"MISSING", "NOT_PROVIDED", "INVALID_REMOVED"}
VALUE_QUALITIES = {"OK", "LOW_PRECISION", "SUSPECT", "CORRECTED"}


def load_schema(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def prov_col(group: str, part: str) -> str:
    return f"{group}__{part}"


def env_columns(schema: dict[str, Any]) -> list[str]:
    cols: list[str] = []
    for group, spec in schema["environmental_groups"].items():
        cols += list(spec["variables"])
        cols += [prov_col(group, p) for p in PROV]
    return cols


def observation_columns(schema: dict[str, Any]) -> list[str]:
    return list(schema["tables"]["observations"]["fields"]) + env_columns(schema)


def image_columns(schema: dict[str, Any]) -> list[str]:
    return list(schema["tables"]["images"]["fields"])


def _check_field(errors: list[str], table: str, df: pd.DataFrame, name: str, spec: dict[str, Any],
                 schema: dict[str, Any], enums: dict[str, list[str]]) -> None:
    s = df[name]
    present = s.notna() & (s.astype("string") != "")
    if spec.get("required") and not present.all():
        errors.append(f"{table}.{name}: {int((~present).sum())} valores vacíos en un campo obligatorio")
    allowed = None
    if "enum" in spec:
        allowed = set(spec["enum"])
    elif "enum_from" in spec:
        allowed = set(enums[spec["enum_from"]])
    elif "vocabulary" in spec:
        allowed = set(schema["vocabularies"][spec["vocabulary"]])
    if allowed is not None:
        bad = present & ~s.astype("string").isin(allowed)
        if bad.any():
            errors.append(f"{table}.{name}: {int(bad.sum())} valores fuera del vocabulario "
                          f"(p. ej. {s[bad].astype(str).unique()[:3].tolist()})")
    if "range" in spec:
        v = pd.to_numeric(s, errors="coerce")
        lo, hi = spec["range"]
        bad = v.notna() & ((v < lo) | (v > hi))
        if bad.any():
            errors.append(f"{table}.{name}: {int(bad.sum())} valores fuera de [{lo}, {hi}]")


def validate(obs: pd.DataFrame, img: pd.DataFrame, schema: dict[str, Any], cfg: dict[str, Any]) -> list[str]:
    """Return the list of schema violations (empty list = valid)."""
    errors: list[str] = []
    enums = {"classes": cfg["classes"], "sources": list(cfg["sources"])}
    for table, df in (("observations", obs), ("images", img)):
        spec = schema["tables"][table]
        expected = observation_columns(schema) if table == "observations" else image_columns(schema)
        missing = [c for c in expected if c not in df.columns]
        extra = [c for c in df.columns if c not in expected]
        if missing or extra:
            errors.append(f"{table}: columnas faltantes {missing}, columnas no declaradas {extra}")
            continue
        pk = spec["primary_key"]
        if df[pk].duplicated().any():
            errors.append(f"{table}.{pk}: {int(df[pk].duplicated().sum())} claves repetidas")
        for name, fspec in spec["fields"].items():
            _check_field(errors, table, df, name, fspec, schema, enums)
    if errors:
        return errors
    # foreign key and label consistency
    orphans = ~img["observation_uid"].isin(obs["observation_uid"])
    if orphans.any():
        errors.append(f"images.observation_uid: {int(orphans.sum())} imágenes sin observación")
    for table, df in (("observations", obs), ("images", img)):
        expected_label = df["genus"].map({g: i for i, g in enumerate(cfg["classes"])})
        if (expected_label != df["label_id"]).any():
            errors.append(f"{table}.label_id no corresponde al orden de classes en la configuración")
    n_img = img.groupby("observation_uid").size().reindex(obs["observation_uid"]).fillna(0).astype(int).to_numpy()
    if (obs["n_images"].to_numpy() != n_img).any() or (n_img == 0).any():
        errors.append("observations.n_images no coincide con la tabla images (o hay observaciones sin imágenes)")
    split_obs = img["observation_uid"].map(obs.set_index("observation_uid")["split"])
    if (split_obs != img["split"]).any():
        errors.append("images.split difiere del split de su observación")
    # environmental groups: variables + provenance triple
    prov = schema["provenance"]
    src_re = re.compile(prov["source"]["pattern"])
    for group, gspec in schema["environmental_groups"].items():
        s_col, m_col, q_col = (prov_col(group, p) for p in PROV)
        for name, fspec in gspec["variables"].items():
            _check_field(errors, "observations", obs, name, fspec, schema, enums)
        bad_src = ~obs[s_col].astype(str).map(lambda v: bool(src_re.match(v)))
        if bad_src.any():
            errors.append(f"{s_col}: {int(bad_src.sum())} valores con formato inválido")
        for col, allowed in ((m_col, prov["method"]), (q_col, prov["quality"])):
            bad = ~obs[col].isin(list(allowed))
            if bad.any():
                errors.append(f"{col}: {int(bad.sum())} valores fuera del vocabulario")
        primary = next(iter(gspec["variables"]))
        has_value = obs[primary].notna()
        q = obs[q_col]
        if (has_value & q.isin(EMPTY_QUALITIES)).any():
            errors.append(f"{group}: hay valores con calidad de ausencia ({sorted(EMPTY_QUALITIES)})")
        if (~has_value & q.isin(VALUE_QUALITIES)).any():
            errors.append(f"{group}: hay calidad de valor presente sin valor en '{primary}'")
        np_rows = q == "NOT_PROVIDED"
        if ((obs.loc[np_rows, s_col] != "none") | (obs.loc[np_rows, m_col] != "NOT_AVAILABLE")).any():
            errors.append(f"{group}: NOT_PROVIDED exige source=none y method=NOT_AVAILABLE")
    return errors
