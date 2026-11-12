"""Visual report of an external evaluation: examples the model got right and wrong, for presentation.

    python scripts/reports/make_external_gallery.py
    python scripts/reports/make_external_gallery.py --set ext_full_001 --run runs/<run_id> --seed 0

Reads the saved predictions of ``scripts/train/evaluate_external.py`` (no inference) and writes, under
``reports/external/<set>/galeria/`` (ignored by Git: it contains third-party photographs):

    galeria_validacion_externa.pdf   16:9 pages: summary, confusion matrices, correct examples per genus, most
                                     frequent errors, most confident errors, Orinoquía, photo credits
    pagina_NN.png                    the same pages as images, for slides
    creditos.csv                     ID of every photo shown -> source, record, author, license

Rules (stated on the first page):
  * Unit = observation, as in the metrics: the prediction is the mean over the observation's images; the photo shown
    is its first image (by ``image_uid``) and the caption says how many images the observation has.
  * Examples are drawn at random with a fixed seed inside each category, spread across sources (or genera), never
    picked by hand. Only the "most confident errors" page is chosen by a rule (highest predicted probability).
  * Every photo is checked against ``data/versions/<set>/image_integrity.csv`` before it is drawn.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import textwrap
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from PIL import Image  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = "runs/20261024T031650621510Z_s5_mobilenet_v3_large_v002_c"
IMAGE_ROOT = REPO_ROOT / "data" / "raw" / "images" / "external_v001"
SOURCE_ES = {"inaturalist_co": "iNaturalist CO", "mushroom_observer_co": "Mushroom Observer CO",
             "mind_funga_v2": "MIND.Funga"}
MIND_CREDIT = "MIND.Funga v2, Mendeley Data, DOI 10.17632/sfrbdjvxcc.2"
PAGE = (13.333, 7.5)
GREEN, RED, GREY = "#2e7d32", "#c62828", "#555555"
COLS, ROWS_PER_PAGE = 5, 3


def f2(v: float) -> str:
    return f"{v:.2f}".replace(".", ",")


def f3(v: float) -> str:
    return f"{v:.3f}".replace(".", ",")


def mil(n: int) -> str:
    return f"{int(n):,}".replace(",", ".")


# --------------------------------------------------------------------------- data
def load(set_id: str, run: Path) -> dict:
    out = REPO_ROOT / "reports" / "external" / set_id / run.name
    if not (out / "completed.json").is_file():
        raise SystemExit(f"{out} has no completed external evaluation; run scripts/train/evaluate_external.py first")
    classes = json.loads((REPO_ROOT / "data" / "versions" / set_id / "manifest.json").read_text(encoding="utf-8"))["classes"]
    op = pd.read_csv(out / "observation_predictions.csv", dtype={"gbif_id": str, "department": str})
    op["department"] = op["department"].fillna("")
    ip = pd.read_csv(out / "predictions_external.csv")
    pcols = [f"p_{c}" for c in classes]
    agg = ip.groupby("observation_uid")[pcols].mean()
    op = op.merge(agg, left_on="observation_uid", right_index=True, how="left", validate="one_to_one")
    P = op[pcols].to_numpy()
    op["p_true"] = P[np.arange(len(op)), op["label_id"].to_numpy()]
    proc = REPO_ROOT / "data" / "processed" / set_id
    obs = pd.read_csv(proc / "observations.csv", dtype=str, keep_default_na=False)
    op = op.merge(obs[["observation_uid", "taxon_name_raw", "n_images"]], on="observation_uid", how="left",
                  validate="one_to_one")
    img = pd.read_csv(proc / "images.csv", dtype=str, keep_default_na=False)
    dl = pd.read_csv(IMAGE_ROOT / "download_manifest.csv", dtype=str, keep_default_na=False)[["image_uid", "filename"]]
    integ = pd.read_csv(REPO_ROOT / "data" / "versions" / set_id / "image_integrity.csv", dtype=str)[["image_uid", "sha256"]]
    img = img.merge(dl, on="image_uid", how="inner").merge(integ, on="image_uid", how="inner")
    first = img.sort_values("image_uid").groupby("observation_uid", as_index=False).first()
    op = op.merge(first[["observation_uid", "image_uid", "filename", "sha256", "license", "rights_holder", "locator"]],
                  on="observation_uid", how="left", validate="one_to_one")
    if op["filename"].isna().any():
        raise SystemExit("some observations have no downloaded image")
    return {"op": op, "classes": classes, "out": out,
            "metrics": json.loads((out / "metrics_external.json").read_text(encoding="utf-8"))["groups"],
            "evaluation": json.loads((out / "evaluation.json").read_text(encoding="utf-8")),
            "val": json.loads((run / "metrics_val.json").read_text(encoding="utf-8"))["observation_level"]}


def stratified(df: pd.DataFrame, k: int, rng: np.random.Generator, by: str) -> list[dict]:
    """Up to k random rows, taken round-robin across the values of ``by`` (so no group dominates)."""
    pools = {g: list(rng.permutation(len(x))) for g, x in df.groupby(by)}
    frames = {g: x.reset_index(drop=True) for g, x in df.groupby(by)}
    keys = list(rng.permutation(sorted(pools)))
    picked = []
    while len(picked) < k and any(pools[g] for g in keys):
        for g in keys:
            if pools[g] and len(picked) < k:
                picked.append(frames[g].iloc[pools[g].pop()].to_dict())
    return picked


# --------------------------------------------------------------------------- drawing
class Report:
    def __init__(self, out_dir: Path, meta: dict):
        self.out_dir = out_dir
        self.pdf = PdfPages(out_dir / "galeria_validacion_externa.pdf", metadata=meta)
        self.n = 0
        self.credits: list[dict] = []
        self.counter: dict[str, int] = {}

    def save(self, fig) -> None:
        self.n += 1
        fig.text(0.97, 0.015, f"{self.n}", ha="right", fontsize=8, color=GREY)
        self.pdf.savefig(fig)
        fig.savefig(self.out_dir / f"pagina_{self.n:02d}.png", dpi=150)
        plt.close(fig)

    def close(self) -> None:
        self.pdf.close()

    def tag(self, r: dict, prefix: str, section: str) -> str:
        self.counter[prefix] = self.counter.get(prefix, 0) + 1
        fid = f"{prefix}{self.counter[prefix]:02d}"
        mind = r["source_id"] == "mind_funga_v2"
        self.credits.append({
            "id": fid, "seccion": section, "fuente": SOURCE_ES[r["source_id"]], "observacion": r["observation_uid"],
            "registro": r["record_url"], "imagen": r["locator"],
            "autor": MIND_CREDIT if mind else r["rights_holder"], "licencia": r["license"],
            "genero_real": r["genus"], "genero_predicho": r["pred_genus"], "p_predicho": round(r["p_pred"], 4),
            "acierto": bool(r["correct"])})
        return fid


def header(fig, title: str, subtitle: str = "") -> None:
    fig.text(0.03, 0.945, title, fontsize=20, weight="bold")
    if subtitle:
        fig.text(0.03, 0.905, textwrap.fill(subtitle, 165), fontsize=10, color=GREY, va="top")


def thumb(r: dict, size: int = 420) -> np.ndarray:
    path = IMAGE_ROOT / r["filename"]
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != r["sha256"]:
        raise SystemExit(f"{path} does not match image_integrity.csv")
    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((size, size))
        canvas = Image.new("RGB", (size, size), "white")
        canvas.paste(im, ((size - im.width) // 2, (size - im.height) // 2))
    return np.asarray(canvas)


NEAR_DUP_BITS = 8   # difference hash 16x16: <= 8 differing bits = the same photo re-saved


def dhash(r: dict) -> np.ndarray:
    with Image.open(IMAGE_ROOT / r["filename"]) as im:
        a = np.asarray(im.convert("L").resize((17, 16)), dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).ravel()


def where(r: dict) -> str:
    if r["department"]:
        return r["department"]
    return "sin ubicación" if r["source_id"] == "mind_funga_v2" else "Colombia, sin depto."


def caption(r: dict, fid: str) -> str:
    n = int(r["n_images"]) if str(r["n_images"]).isdigit() else 1
    imgs = f" · {n} fotos" if n > 1 else ""
    taxon = textwrap.shorten(str(r["taxon_name_raw"]), 30, placeholder="…")
    if r["correct"]:
        verdict = f"predicho {r['pred_genus']} ({f2(r['p_pred'])})"
    else:
        verdict = f"predicho {r['pred_genus']} ({f2(r['p_pred'])})\nprob. del real {f2(r['p_true'])}"
    return f"{fid} · {SOURCE_ES[r['source_id']]}\n{where(r)}{imgs}\n{taxon}\n{verdict}"


def grid_pages(rep: Report, title: str, subtitle: str, rows: list[tuple[str, list[dict]]], prefix: str) -> None:
    top, bottom, left, label_w = 0.84, 0.04, 0.03, 0.15
    slot = (top - bottom) / ROWS_PER_PAGE
    w = (0.97 - left - label_w) / COLS
    for start in range(0, len(rows), ROWS_PER_PAGE):
        fig = plt.figure(figsize=PAGE)
        header(fig, title, subtitle)
        for i, (label, items) in enumerate(rows[start:start + ROWS_PER_PAGE]):
            y0 = top - (i + 1) * slot
            fig.text(left, y0 + slot * 0.6, label, fontsize=11, weight="bold", va="center")
            if not items:
                fig.text(left + label_w, y0 + slot * 0.6, "sin ejemplos en esta categoría", fontsize=10,
                         color=GREY, va="center")
            for j, r in enumerate(items):
                fid = rep.tag(r, prefix, title)
                x0 = left + label_w + j * w
                cap = 0.36 * slot
                ax = fig.add_axes([x0 + 0.004, y0 + cap, w - 0.008, slot - cap - 0.010])
                ax.imshow(thumb(r))
                ax.set_xticks([])
                ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_edgecolor(GREEN if r["correct"] else RED)
                    s.set_linewidth(3)
                fig.text(x0 + w / 2, y0 + cap - 0.004, caption(r, fid), ha="center", va="top", fontsize=6.8,
                         linespacing=1.2)
        rep.save(fig)


# --------------------------------------------------------------------------- pages
GROUPS = [("all", "Conjunto externo completo"), ("scope:colombia", "Colombia (iNaturalist + MO)"),
          ("source:inaturalist_co", "   iNaturalist CO"), ("source:mushroom_observer_co", "   Mushroom Observer CO"),
          ("source:mind_funga_v2", "MIND.Funga (sin ubicación)"), ("region:orinoquia", "Orinoquía"),
          ("region:other", "Colombia fuera de la Orinoquía")]


def summary_page(rep: Report, d: dict, set_id: str, seed: int) -> None:
    m, ev, op = d["metrics"], d["evaluation"], d["op"]
    fig = plt.figure(figsize=PAGE)
    header(fig, "Validación externa: aciertos y errores con ejemplos",
           f"Modelo {ev['run_name']} (MobileNetV3-Large, seleccionado en S5), entrenado solo con FungiTastic "
           f"(Dinamarca). Conjunto {set_id}: {mil(ev['observations_evaluated'])} observaciones y "
           f"{mil(ev['images_evaluated'])} imágenes que el modelo nunca vio. Generado el {date.today():%d-%m-%Y}.")
    rows = [["FungiTastic val (referencia, en dominio)", mil(d["val"]["n"]), "10", f3(d["val"]["macro_f1"]), "",
             f3(d["val"]["accuracy"])]]
    for key, name in GROUPS:
        if key in m:
            o = m[key]["observation_level"]
            lo, hi = o["macro_f1_present_classes_ci95"]
            rows.append([name, mil(o["n"]), str(len(o["classes_present"])), f3(o["macro_f1_present_classes"]),
                         f"[{f3(lo)}; {f3(hi)}]", f3(o["accuracy"])])
    ax = fig.add_axes([0.03, 0.47, 0.58, 0.38])
    ax.axis("off")
    t = ax.table(cellText=rows, colLabels=["Grupo", "Obs.", "Clases", "Macro-F1", "IC95", "Exactitud"],
                 colLoc="center", cellLoc="center", loc="upper left", colWidths=[0.40, 0.10, 0.09, 0.12, 0.18, 0.12])
    t.auto_set_font_size(False)
    t.set_fontsize(10)
    t.scale(1, 1.6)
    for (i, j), c in t.get_celld().items():
        if i == 0:
            c.set_text_props(weight="bold")
            c.set_facecolor("#e8eef5")
        if j == 0 and i > 0:
            c.set_text_props(ha="left")
            c._loc = "left"
        if i in (2, 7):
            c.set_text_props(weight="bold")
    wrong = op[~op["correct"]]
    top = wrong.groupby(["genus", "pred_genus"]).size().sort_values(ascending=False)
    (g1, p1), n1 = top.index[0], int(top.iloc[0])
    allo, col = m["all"]["observation_level"], m.get("region:orinoquia", {}).get("observation_level")
    find = [f"• La exactitud baja de {f3(d['val']['accuracy'])} en FungiTastic a {f3(allo['accuracy'])} con fotos externas.",
            f"• Error más frecuente: {g1} predicho como {p1} ({n1} observaciones).",
            "• Las etiquetas externas son de comunidad (iNaturalist, Mushroom Observer) o de carpeta (MIND.Funga):",
            "   algunos «errores» pueden ser etiquetas dudosas; cada foto tiene su enlace en Créditos."]
    if col:
        find.insert(1, f"• Orinoquía: exactitud {f3(col['accuracy'])} frente a "
                       f"{f3(m['region:other']['observation_level']['accuracy'])} en el resto de Colombia.")
    fig.text(0.03, 0.41, "Hallazgos", fontsize=13, weight="bold")
    fig.text(0.03, 0.38, "\n".join(find), fontsize=10.5, va="top", linespacing=1.6)
    how = ["Cómo leer las páginas",
           "",
           "Borde verde = acierto, borde rojo = error.",
           "",
           "La unidad es la observación, como en las",
           "métricas: la predicción promedia todas sus",
           "fotos; se muestra la primera.",
           "",
           "Pie de cada foto:",
           "ID · fuente · departamento · n.º de fotos",
           "nombre que trae la fuente",
           "género predicho (probabilidad) · prob. del real",
           "",
           f"Los ejemplos se eligen al azar (semilla {seed})",
           "dentro de cada categoría, repartidos entre",
           "fuentes o géneros; ninguno se eligió a mano.",
           "Solo «errores con más confianza» sigue una",
           "regla: los de mayor probabilidad predicha.",
           "",
           "Macro-F1 sobre las clases presentes en cada",
           "grupo; IC95 por bootstrap de observaciones."]
    fig.text(0.66, 0.84, "\n".join(how), fontsize=10, va="top", linespacing=1.35,
             bbox=dict(boxstyle="round,pad=0.8", facecolor="#f5f5f5", edgecolor="#cccccc"))
    rep.save(fig)


def confusion_page(rep: Report, d: dict) -> None:
    op, classes = d["op"], d["classes"]
    parts = [("Conjunto externo completo", op)]
    if (op["region"] == "orinoquia").any():
        parts.append(("Orinoquía", op[op["region"] == "orinoquia"]))
    fig = plt.figure(figsize=PAGE)
    header(fig, "Matrices de confusión (nivel de observación)",
           "Cada fila es el género real y suma 100 %; el número de cada celda es el número de observaciones. "
           "La diagonal son los aciertos. Entre paréntesis, observaciones de cada género real.")
    n = len(classes)
    for k, (name, df) in enumerate(parts):
        cm = np.zeros((n, n), int)
        np.add.at(cm, (df["label_id"].to_numpy(), df["pred_label_id"].to_numpy()), 1)
        present = [i for i in range(n) if cm[i].sum()]
        mat = cm[present]
        norm = mat / mat.sum(axis=1, keepdims=True)
        ax = fig.add_axes([0.13 + k * 0.48, 0.17, 0.34, 0.66])
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for a in range(len(present)):
            for b in range(n):
                if mat[a, b]:
                    ax.text(b, a, mat[a, b], ha="center", va="center", fontsize=8,
                            color="white" if norm[a, b] > 0.55 else "black")
        ax.set_xticks(range(n), classes, rotation=45, ha="right", fontsize=8.5)
        ax.set_yticks(range(len(present)), [f"{classes[i]} ({mat[r].sum()})" for r, i in enumerate(present)],
                      fontsize=8.5)
        ax.set_xlabel("Predicho")
        ax.set_ylabel("Real")
        ax.set_title(f"{name} · {mil(len(df))} obs. · exactitud {f3((df['correct']).mean())}", fontsize=11)
    rep.save(fig)


def credits_pages(rep: Report) -> None:
    df = pd.DataFrame(rep.credits)
    df.to_csv(rep.out_dir / "creditos.csv", index=False, lineterminator="\n", encoding="utf-8")
    per = 26
    for start in range(0, len(df), per):
        fig = plt.figure(figsize=PAGE)
        header(fig, "Créditos de las fotografías",
               "Fotos de terceros, reducidas para este informe, con su licencia original. Uso académico no comercial; "
               "no publicar este documento sin revisar las condiciones NC y SA. Lista completa en creditos.csv.")
        ax = fig.add_axes([0.03, 0.04, 0.94, 0.83])
        ax.axis("off")
        rows = [[r.id, r.fuente, textwrap.shorten(str(r.registro), 58, placeholder="…"),
                 textwrap.shorten(str(r.autor), 62, placeholder="…"), r.licencia]
                for r in df.iloc[start:start + per].itertuples()]
        t = ax.table(cellText=rows, colLabels=["ID", "Fuente", "Registro", "Autor / titular", "Licencia"],
                     loc="upper left", cellLoc="left", colLoc="left", colWidths=[0.05, 0.13, 0.36, 0.38, 0.08])
        t.auto_set_font_size(False)
        t.set_fontsize(7.5)
        t.scale(1, 1.25)
        for (i, _), c in t.get_celld().items():
            if i == 0:
                c.set_text_props(weight="bold")
                c.set_facecolor("#e8eef5")
        rep.save(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", default="ext_full_001")
    ap.add_argument("--run", default=DEFAULT_RUN)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per-row", type=int, default=COLS)
    args = ap.parse_args()
    run = (REPO_ROOT / args.run).resolve() if not Path(args.run).is_absolute() else Path(args.run)
    d = load(args.set, run)
    op, classes, m = d["op"], d["classes"], d["metrics"]
    out_dir = REPO_ROOT / "reports" / "external" / args.set / "galeria"
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("pagina_*.png"):
        old.unlink()
    rep = Report(out_dir, {"Title": "Validación externa: aciertos y errores", "Subject": f"{args.set} · {run.name}"})
    rng = lambda k: np.random.default_rng([args.seed, k])  # noqa: E731
    k = args.per_row

    summary_page(rep, d, args.set, args.seed)
    confusion_page(rep, d)

    per_class = m["all"]["observation_level"]["per_class"]
    rows = []
    for c, g in enumerate(classes):
        ok = op[(op["label_id"] == c) & op["correct"]]
        pc = per_class[c]
        label = f"{g}\nF1 {f3(pc['f1'])}\n{mil(pc['support'])} obs.\n{mil(len(ok))} aciertos"
        rows.append((label, stratified(ok, k, rng(c), "source_id") if len(ok) else []))
    grid_pages(rep, "Aciertos por género",
               "Observaciones bien clasificadas, al azar y repartidas entre fuentes. "
               "A la izquierda, el F1 del género en el conjunto externo completo.", rows, "A")

    wrong = op[~op["correct"]]
    pairs = wrong.groupby(["genus", "pred_genus"]).size().sort_values(ascending=False).head(6)
    rows = [(f"Real: {g}\nPredicho: {p}\n{n} obs.",
             stratified(wrong[(wrong["genus"] == g) & (wrong["pred_genus"] == p)], k, rng(100 + i), "source_id"))
            for i, ((g, p), n) in enumerate(pairs.items())]
    grid_pages(rep, "Errores más frecuentes",
               f"Las 6 confusiones más comunes ({mil(int(pairs.sum()))} de {mil(len(wrong))} errores). "
               "Ejemplos al azar, repartidos entre fuentes. «prob. del real» es la probabilidad que el modelo dio al "
               "género correcto.",
               rows, "E")

    conf, seen, names = [], [], set()
    for r in wrong.sort_values("p_pred", ascending=False).to_dict("records"):
        name = (r["source_id"], r["taxon_name_raw"])
        h = dhash(r)
        if name in names or any(int(np.count_nonzero(h != s)) <= NEAR_DUP_BITS for s in seen):
            continue
        conf.append(r)
        seen.append(h)
        names.add(name)
        if len(conf) == 2 * k:
            break
    grid_pages(rep, "Errores con más confianza",
               "Los errores en que el modelo estaba más seguro, como máximo uno por nombre de la fuente (varias fotos "
               "del mismo ejemplar repetirían el mismo caso). Suelen ser fotos ambiguas o etiquetas dudosas de la "
               "fuente: vale la pena revisarlos en el registro original (ver Créditos).",
               [(f"Más seguros\n{1 + i * k}–{(i + 1) * k}", conf[i * k:(i + 1) * k]) for i in range(2)], "C")

    ori = op[op["region"] == "orinoquia"]
    if len(ori):
        o = m["region:orinoquia"]["observation_level"]
        ok, bad = ori[ori["correct"]], ori[~ori["correct"]]
        sel_ok = stratified(ok, 2 * k, rng(200), "genus")
        sel_bad = stratified(bad, 2 * k, rng(201), "genus")
        grid_pages(rep, "Orinoquía (Meta, Casanare, Arauca, Vichada)",
                   f"{mil(len(ori))} observaciones; exactitud {f3(o['accuracy'])}, macro-F1 "
                   f"{f3(o['macro_f1_present_classes'])} en {len(o['classes_present'])} géneros presentes. "
                   "Ejemplos al azar, repartidos entre géneros.",
                   [(f"Aciertos\n{mil(len(ok))} obs.", sel_ok[:k]), ("", sel_ok[k:]),
                    (f"Errores\n{mil(len(bad))} obs.", sel_bad[:k])], "O")
        if len(sel_bad) > k:
            grid_pages(rep, "Orinoquía: más errores",
                       "Continuación. Ejemplos al azar, repartidos entre géneros.", [("Errores", sel_bad[k:])], "O")

    credits_pages(rep)
    rep.close()
    print(f"{rep.n} páginas, {len(rep.credits)} fotos\n{out_dir / 'galeria_validacion_externa.pdf'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
