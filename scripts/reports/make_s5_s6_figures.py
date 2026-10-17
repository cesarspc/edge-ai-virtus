"""Figures for the Week 5-6 report, built only from saved results (runs/, reports/s5). No images, no PyTorch.

Usage:
    python scripts/reports/make_s5_s6_figures.py      # writes reports/figures/s5_s6_*.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.ticker  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
FIG = REPO / "reports" / "figures"
RESNET = REPO / "runs" / "20261004T133123181686Z_ma_resnet18_v002"
MOBILE = REPO / "runs" / "20261012T161041257797Z_s5_mobilenet_v3_large_v002_a"
C_RES, C_MOB = "#9aa5b1", "#1f6fb2"
LEVEL_ES = {"genus": "Género", "family": "Familia", "order": "Orden", "class": "Clase", "phylum": "Filo"}

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.formatter.use_locale": False})


def comma(value: float, decimals: int) -> str:
    """Spanish decimal comma, as in the text of the report."""
    return f"{value:.{decimals}f}".replace(".", ",")


def comma_axis(axis, decimals: int) -> None:
    axis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: comma(v, decimals)))


def save(fig, name: str) -> None:
    fig.savefig(FIG / name, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("written", FIG / name)


def size_speed() -> None:
    prof = json.loads((REPO / "reports/s5/architecture_profile.json").read_text(encoding="utf-8"))
    names = {"resnet18": "ResNet-18", "mobilenet_v3_large": "MobileNetV3-Large"}
    rows = [(names[m["architecture"]], m["parameters"] / 1e6, m["serialized_state_mb"], m["latency_median_ms"])
            for m in prof["models"]]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), layout="constrained")
    for ax, (i, title, unit) in zip(axes, [(1, "Parámetros", "millones"), (2, "Tamaño del archivo", "MB"),
                                           (3, "Tiempo por foto (PC)", "ms")]):
        vals = [r[i] for r in rows]
        bars = ax.bar([r[0] for r in rows], vals, color=[C_RES, C_MOB])
        ax.bar_label(bars, [comma(v, 1) for v in vals], padding=3)
        ax.set_title(f"{title} ({unit})")
        ax.set_ylim(0, max(vals) * 1.2)
        ax.set_yticks([])
    fig.suptitle("Menos es mejor en las tres", fontsize=11, color="#555")
    save(fig, "s5_s6_size_speed.png")


def training_curves() -> None:
    res, mob = pd.read_csv(RESNET / "history.csv"), pd.read_csv(MOBILE / "history.csv")
    fig, ax = plt.subplots(figsize=(9, 4.2), layout="constrained")
    ax.plot(res.epoch, res.val_obs_macro_f1, "o-", color=C_RES, label="ResNet-18")
    ax.plot(mob.epoch, mob.val_obs_macro_f1, "o-", color=C_MOB, label="MobileNetV3-Large (ensayo A)")
    for h, c, dy in ((res, C_RES, -30), (mob, C_MOB, 14)):
        b = h.loc[h.val_obs_macro_f1.idxmax()]
        ax.annotate(f"mejor: {comma(b.val_obs_macro_f1, 3)}\n(vuelta {int(b.epoch)})", (b.epoch, b.val_obs_macro_f1),
                    xytext=(28 if dy < 0 else 0, dy), textcoords="offset points", ha="center", color=c, fontsize=9)
    ax.set_xlabel("Vuelta de entrenamiento")
    ax.set_ylabel("F1 promedio (validación, por observación)")
    ax.set_ylim(0.84, 0.99)
    comma_axis(ax.yaxis, 2)
    ax.legend(loc="lower right", frameon=False)
    ax.grid(axis="y", alpha=0.3)
    save(fig, "s5_s6_training_curves.png")












def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    for build in (size_speed, training_curves):
        build()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
