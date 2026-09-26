"""Cleaning and normalisation rules of the processed dataset (Week 3).

Every rule has a stable id (``RULES``). Loaders apply the rules and record how many observations/images each
one affected in a :class:`CleaningLog`, which is written to ``data/versions/<version>/cleaning_log.csv``.
Raw data is never modified: rules act on in-memory copies.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

# id -> (category, action, description). Descriptions are in Spanish because they go verbatim into the reports.
RULES: dict[str, tuple[str, str, str]] = {
    # --- alcance y taxonomía
    "T1": ("taxonomía", "excluir", "El género (normalizado) no está entre las 10 clases del alcance."),
    "T2": ("taxonomía", "corregir", "Sinónimo de género reasignado al género aceptado (Coriolus→Trametes, Hirneola→Auricularia; en MO vía synonym_id)."),
    "T3": ("taxonomía", "excluir", "Identificación sin resolver a género (rango superior a género o calificador sobre el género: 'cf. Amanita', '?')."),
    "T4": ("taxonomía", "excluir", "Observación cuyas imágenes tienen géneros distintos (etiqueta en conflicto)."),
    "T5": ("taxonomía", "excluir", "Mushroom Observer: consenso de la comunidad (vote_cache) inferior al umbral configurado."),
    "T6": ("taxonomía", "excluir", "Reino distinto de Fungi o nombre sin clasificación."),
    # --- identidad y duplicados
    "D1": ("duplicados", "excluir", "Fila de imagen repetida dentro de la fuente (mismo archivo/ID); se conserva una."),
    "D2": ("duplicados", "excluir", "Imagen enlazada a más de una observación (Mushroom Observer)."),
    "D3": ("duplicados", "excluir", "Archivos con el mismo SHA-256 publicado (MIND.Funga): se conserva uno; si el grupo cruza géneros se excluyen todos."),
    "D4": ("duplicados", "excluir", "Observación de MO publicada también en iNaturalist (enlace explícito 'Mushroom Observer URL'): se conserva la de iNaturalist."),
    "D5": ("duplicados", "excluir", "Posible duplicado entre fuentes: MO e iNaturalist con mismo género, misma fecha y coordenadas a < 0,01°: se conserva la de iNaturalist."),
    "D6": ("duplicados", "excluir", "MIND.Funga: variantes con fondo sustituido ('branco'/'verde' en el nombre) de una misma fotografía."),
    "D7": ("duplicados", "excluir", "Observación sin imágenes tras aplicar las demás reglas."),
    # --- licencias
    "L1": ("licencias", "excluir", "Licencia de la imagen fuera de la lista permitida (todos los derechos reservados o variantes ND)."),
    "L2": ("licencias", "excluir", "Mushroom Observer: imagen marcada ok_for_export = 0."),
    # --- metadatos ambientales (no excluyen: anulan o marcan)
    "E1": ("ambiental", "anular", "Coordenadas no numéricas, fuera de rango o (0, 0): se vacían (calidad INVALID_REMOVED)."),
    "E2": ("ambiental", "marcar", "Geolocalización imprecisa: incertidumbre > umbral, punto oscurecido o centroide de localidad (LOW_PRECISION)."),
    "E3": ("ambiental", "anular", "Fecha no interpretable, anterior al año mínimo o posterior a la fecha de acceso (INVALID_REMOVED)."),
    "E4": ("ambiental", "marcar", "Fecha con precisión de mes o año (LOW_PRECISION)."),
    "E5": ("ambiental", "marcar", "Elevación exactamente 0 m (SUSPECT) o fuera de rango (INVALID_REMOVED)."),
    "E6": ("ambiental", "marcar", "Sustrato solo con categoría gruesa (metaSubstrate) cuando falta el detallado (LOW_PRECISION)."),
    "E7": ("ambiental", "marcar", "Mushroom Observer: la localidad no es el lugar de recolección (is_collection_location = 0) (SUSPECT)."),
    "E8": ("ambiental", "anular", "Clima no disponible: climatic.zip de FungiTastic responde HTTP 403; ninguna otra fuente trae clima (MISSING/NOT_PROVIDED)."),
    # --- normalización (transforman, no excluyen)
    "N1": ("normalización", "transformar", "Género: espacios, mayúsculas y sinónimos normalizados; label_id = posición en la lista de clases."),
    "N2": ("normalización", "transformar", "Fechas a ISO 8601 (AAAA-MM-DD) con precisión explícita (day/month/year)."),
    "N3": ("normalización", "transformar", "Coordenadas a grados decimales WGS84 con 6 decimales; incertidumbre en metros."),
    "N4": ("normalización", "transformar", "Licencias a códigos CC normalizados (CC0, CC-BY, CC-BY-NC, …)."),
    "N5": ("normalización", "transformar", "Sustrato y hábitat a vocabularios controlados, conservando el texto original (*_raw)."),
    "N6": ("normalización", "transformar", "Identificadores con espacio de nombres '<fuente>:<id>' y país en ISO 3166-1 alfa-2."),
    # --- control de clases
    "C1": ("clases", "verificar", "Cada clase debe tener al menos el mínimo configurado de observaciones en desarrollo."),
}


@dataclass
class CleaningLog:
    rows: list[dict[str, Any]] = field(default_factory=list)

    def add(self, rule: str, source_id: str, n_obs: int | None = None, n_img: int | None = None, note: str = "") -> None:
        assert rule in RULES, rule
        cat, action, desc = RULES[rule]
        self.rows.append({"rule_id": rule, "category": cat, "action": action, "source_id": source_id,
                          "n_observations": n_obs if n_obs is not None else "", "n_images": n_img if n_img is not None else "",
                          "note": note, "description": desc})

    def frame(self) -> pd.DataFrame:
        order = {r: i for i, r in enumerate(RULES)}
        df = pd.DataFrame(self.rows)
        return df.assign(_o=df["rule_id"].map(order)).sort_values(["_o", "source_id"], kind="stable").drop(columns="_o")


# --------------------------------------------------------------------------- taxonomy
UNCERTAIN_GENUS = re.compile(r"^\s*(?:cf\.?|aff\.?|\?)\s|\?$", re.IGNORECASE)


def normalise_genus(names: pd.Series, synonyms: dict[str, str]) -> tuple[pd.Series, pd.Series, pd.Series]:
    """First token of a taxon name -> canonical genus. Returns (genus, was_synonym, uncertain_genus)."""
    s = names.astype("string").str.strip()
    uncertain = s.str.contains(UNCERTAIN_GENUS, regex=True).fillna(False)
    tok = s.str.replace(r"^[\"'?]+", "", regex=True).str.split(r"\s+", n=1, regex=True).str[0].str.capitalize()
    was_syn = tok.isin(list(synonyms)).fillna(False)
    return tok.replace(synonyms), was_syn, uncertain


# --------------------------------------------------------------------------- licenses
LICENSE_MAP = {
    # iNaturalist codes
    "cc0": "CC0", "cc-by": "CC-BY", "cc-by-sa": "CC-BY-SA", "cc-by-nc": "CC-BY-NC", "cc-by-nc-sa": "CC-BY-NC-SA",
    "cc-by-nd": "CC-BY-ND", "cc-by-nc-nd": "CC-BY-NC-ND",
    # Mushroom Observer labels (the site links each label to the corresponding CC deed)
    "Creative Commons Wikipedia Compatible v3.0": "CC-BY-SA",
    "Creative Commons Non-commercial v3.0": "CC-BY-NC-SA",
    "Creative Commons Non-commercial v2.5": "CC-BY-NC-SA",
    "Creative Commons Attribution Non-commercial v4.0": "CC-BY-NC",
    "Creative Commons Attribution v4.0 (Wikipedia compatible)": "CC-BY",
    "Creative Commons Attribution ShareAlike v4.0 (Wikipedia compatible)": "CC-BY-SA",
    "Public Domain (Wikipedia compatible)": "PDM",
    "Creative Commons Attribution Non-commercial ShareAlike v4.0": "CC-BY-NC-SA",
    "Creative Commons Attribution Non-commercial NoDerivs v.4.0": "CC-BY-NC-ND",
    "Creative Commons Attribution NoDerivs v4.0": "CC-BY-ND",
}


def normalise_license(s: pd.Series) -> pd.Series:
    return s.map(lambda v: "ARR" if v is None or (isinstance(v, float) and math.isnan(v)) else LICENSE_MAP.get(str(v), f"UNKNOWN:{v}"))


# --------------------------------------------------------------------------- dates
DATE_RE = re.compile(r"^\s*(\d{4})(?:-(\d{1,2}))?(?:-(\d{1,2}))?")


def parse_dates(s: pd.Series, min_year: int, max_date: str) -> tuple[pd.Series, pd.Series, pd.Series]:
    """-> (ISO date string or NA, precision day|month|year or NA, invalid flag). Month/day 00 lowers precision."""
    out, prec, invalid = [], [], []
    for v in s.astype("string").fillna(""):
        m = DATE_RE.match(v)
        if not m:
            out.append(pd.NA); prec.append(pd.NA); invalid.append(bool(v.strip())); continue
        y, mo, d = int(m.group(1)), int(m.group(2) or 0), int(m.group(3) or 0)
        p = "day" if mo and d else ("month" if mo else "year")
        try:
            iso = pd.Timestamp(year=y, month=mo or 1, day=d or 1).strftime("%Y-%m-%d")
        except ValueError:
            out.append(pd.NA); prec.append(pd.NA); invalid.append(True); continue
        if y < min_year or iso > max_date:
            out.append(pd.NA); prec.append(pd.NA); invalid.append(True); continue
        out.append(iso if p == "day" else iso[: 7 if p == "month" else 4]); prec.append(p); invalid.append(False)
    idx = s.index
    return (pd.Series(out, index=idx, dtype="string"), pd.Series(prec, index=idx, dtype="string"),
            pd.Series(invalid, index=idx, dtype=bool))


def date_group(date: pd.Series, precision: pd.Series, source: str | pd.Series, method: str, invalid: pd.Series,
               field_exists: bool = True) -> dict[str, pd.Series]:
    q = np.select([invalid, date.isna(), precision != "day"], ["INVALID_REMOVED", "MISSING", "LOW_PRECISION"], "OK")
    return {"event_date": date, "event_date_precision": precision.where(date.notna()),
            "event_date__source": _src(source, date.index), "event_date__method": pd.Series(method, index=date.index).where(date.notna(), "NOT_AVAILABLE"),
            "event_date__quality": pd.Series(q, index=date.index)}


# --------------------------------------------------------------------------- coordinates
def valid_coords(lat: pd.Series, lon: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """-> (lat, lon) as floats rounded to 6 decimals with invalid values set to NaN, and the invalid flag."""
    la, lo = pd.to_numeric(lat, errors="coerce"), pd.to_numeric(lon, errors="coerce")
    raw_present = lat.notna() | lon.notna()
    bad = raw_present & (la.isna() | lo.isna() | (la.abs() > 90) | (lo.abs() > 180) | ((la == 0) & (lo == 0)))
    return la.where(~bad).round(6), lo.where(~bad).round(6), bad


def box_centroid(north, south, east, west) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Centroid of a lat/lon bounding box and half-diagonal in metres (haversine)."""
    lat, lon = (north + south) / 2, (east + west) / 2
    r = 6_371_000.0
    p1, p2 = np.radians(south), np.radians(north)
    dphi, dlmb = p2 - p1, np.radians(east - west)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return lat, lon, (r * 2 * np.arcsin(np.sqrt(a.clip(0, 1)))) / 2


def _src(source: str | pd.Series, index) -> pd.Series:
    return source if isinstance(source, pd.Series) else pd.Series(source, index=index)


def not_provided(group: str, variables: list[str], index) -> dict[str, pd.Series]:
    d = {v: pd.Series(pd.NA, index=index, dtype="object") for v in variables}
    d.update({f"{group}__source": pd.Series("none", index=index), f"{group}__method": pd.Series("NOT_AVAILABLE", index=index),
              f"{group}__quality": pd.Series("NOT_PROVIDED", index=index)})
    return d


def missing_group(group: str, variables: list[str], index, source: str = "none") -> dict[str, pd.Series]:
    d = not_provided(group, variables, index)
    d[f"{group}__source"] = pd.Series(source, index=index)
    d[f"{group}__quality"] = pd.Series("MISSING", index=index)
    return d


# --------------------------------------------------------------------------- vocabularies (N5)
SUBSTRATE_MAP = {
    "dead wood (including bark)": "wood_dead", "wood chips or mulch": "wood_dead",
    "wood and roots of living trees": "wood_living", "bark of living trees": "wood_living",
    "soil": "soil",
    "leaf or needle litter": "litter", "fruits": "litter", "cones": "litter", "catkins": "litter",
    "dead stems of herbs, grass etc": "litter", "stems of herbs, grass etc": "litter",
    "living stems of herbs, grass etc": "plant_living", "living leaves": "plant_living",
    "mosses": "moss", "peat mosses": "moss", "liverworts": "moss",
    "faeces": "dung", "fungi": "fungi", "lichens": "fungi",
    "other substrate": "other", "fire spot": "other", "stone": "other", "building stone (e.g. bricks)": "other",
}
# FungiTastic 'metaSubstrate' (coarse; partly in Danish). Used only when 'substrate' is empty (rule E6).
META_SUBSTRATE_MAP = {
    "wood": "wood_unspecified", "jord": "soil", "plant material": "litter", "mosses": "moss", "stone": "other",
    "fungi and mycetozoans": "fungi", "other substrate": "other", "animals": "other", "fire spot": "other",
}
META_SUBSTRATE_TRANSLATED = {"jord"}   # Danish term -> quality CORRECTED
HABITAT_MAP = {
    "Deciduous woodland": "forest_deciduous", "Unmanaged deciduous woodland": "forest_deciduous",
    "Acidic oak woodland": "forest_deciduous",
    "coniferous woodland/plantation": "forest_coniferous", "Unmanaged coniferous woodland": "forest_coniferous",
    "Mixed woodland (with coniferous and deciduous trees)": "forest_mixed",
    "Forest bog": "forest_other", "Bog woodland": "forest_other", "wooded meadow, grazing forest": "forest_other",
    "Thorny scrubland": "scrub", "Willow scrubland": "scrub", "hedgerow": "scrub",
    "natural grassland": "open", "dune": "open", "bog": "open", "heath": "open", "salt meadow": "open", "meadow": "open",
    "park/churchyard": "anthropic", "garden": "anthropic", "lawn": "anthropic", "roadside": "anthropic",
    "gravel or clay pit": "anthropic", "fallow field": "anthropic", "ditch": "anthropic", "roof": "anthropic",
    "masonry": "anthropic", "fertilized field in rotation": "anthropic", "improved grassland": "anthropic",
    "other habitat": "other", "rock": "other",
}

COUNTRY_ISO2 = {"Colombia": "CO"}
