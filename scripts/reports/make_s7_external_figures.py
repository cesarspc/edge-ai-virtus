"""Compact figures of the external validation for the week 7-8 report (reports/week_07_08.tex).

    python scripts/reports/make_s7_external_figures.py

Reuses the loading and hash checks of ``make_external_gallery.py`` (no inference) and writes to ``reports/images/``:

    externa_colombia.png     Colombia (iNaturalist CO + Mushroom Observer CO): 5 hits + 5 errors
    externa_mind_funga.png   MIND.Funga v2 (no location): 2 hits + 2 errors

Examples are drawn at random with a fixed seed (hits spread across genera, errors across the most frequent
confusions of the group); none is chosen by hand. The photograph shown is the first image of each observation.
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_external_gallery as g  # noqa: E402

SEED = 0
OUT = g.REPO_ROOT / "reports" / "images"


def caption(r: dict, tag: str) -> str:
    src = {"inaturalist_co": "iNaturalist", "mushroom_observer_co": "Mushroom Obs.", "mind_funga_v2": "MIND.Funga"}[r["source_id"]]
    place = f" · {r['department']}" if r["department"] else ""
    ok = "acierto" if r["correct"] else "error"
    return (f"{tag} · {src}{place}\nReal: {r['genus']}\nPredicho: {r['pred_genus']} ({g.f2(r['p_pred'])})\n{ok}")


def figure(rows: list[tuple[str, list[dict]]], path: Path, title: str, cols: int, letters: str = "AB") -> None:
    n = len(rows)
    fig, axes = plt.subplots(n, cols, figsize=(2.3 * cols, 3.5 * n), squeeze=False)
    for i, (label, items) in enumerate(rows):
        for j in range(cols):
            ax = axes[i][j]
            ax.axis("off")
            if j >= len(items):
                continue
            r = items[j]
            ax.imshow(g.thumb(r, 360))
            color = g.GREEN if r["correct"] else g.RED
            ax.set_title(caption(r, f"{letters[i]}{j + 1}"), fontsize=7, pad=3, linespacing=1.25, color="black")
            ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False, ec=color, lw=3.5, clip_on=False))
            ax.axis("on")
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
        axes[i][0].set_ylabel(textwrap.fill(label, 14), fontsize=8.5, weight="bold", rotation=90, labelpad=8)
    fig.suptitle(title, fontsize=10, weight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=3.0)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main() -> int:
    d = g.load("ext_full_001", (g.REPO_ROOT / g.DEFAULT_RUN).resolve())
    op = d["op"]
    OUT.mkdir(parents=True, exist_ok=True)
    rng = lambda k: np.random.default_rng([SEED, k])  # noqa: E731

    col = op[op["source_id"] != "mind_funga_v2"].copy()
    hits = g.stratified(col[col["correct"]], 5, rng(1), "genus")
    wrong = col[~col["correct"]].copy()
    wrong["pair"] = wrong["genus"] + ">" + wrong["pred_genus"]
    top = wrong["pair"].value_counts().head(5).index
    errs = g.stratified(wrong[wrong["pair"].isin(top)], 5, rng(2), "pair")
    figure([("Aciertos", hits), ("Errores", errs)], OUT / "externa_colombia.png",
           "Colombia (iNaturalist CO y Mushroom Observer CO): ejemplos al azar", 5)

    mind = op[op["source_id"] == "mind_funga_v2"].copy()
    mh = g.stratified(mind[mind["correct"]], 2, rng(3), "genus")
    mw = mind[~mind["correct"]].copy()
    mw["pair"] = mw["genus"] + ">" + mw["pred_genus"]
    me = g.stratified(mw[mw["pair"].isin(mw["pair"].value_counts().head(5).index)], 2, rng(4), "pair")
    figure([("Aciertos y errores", mh + me)], OUT / "externa_mind_funga.png",
           "MIND.Funga v2 (sin ubicación): 4 ejemplos al azar", 4, letters="M")
    for p in sorted(OUT.glob("externa_*.png")):
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
