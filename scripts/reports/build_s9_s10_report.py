"""Generate a standalone LaTeX S9/S10 report from saved results and compile it.

python scripts/reports/build_s9_s10_report.py
python scripts/reports/build_s9_s10_report.py --source-only
Matches the document structure and typography of reports/week_07_08/main.tex.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "reports/s9_s10/frozen_context"
REPORT = ROOT / "reports/week_09_10"
PDF_ALIAS = ROOT / "output/pdf/reporte_s9_s10_virtus.pdf"


def number(value, digits=6, signed=False):
    spec = f"+.{digits}f" if signed else f".{digits}f"
    return format(float(value), spec).replace(".", ",")


def integer(value):
    return f"{int(value):,}".replace(",", r"\,")


def read_csv(name):
    with (RESULTS / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def latex_table(caption, label, columns, headers, rows, flexible=False, note=None):
    environment = "tabularx" if flexible else "tabular"
    width = r"{\linewidth}" if flexible else ""
    lines = [r"\begin{table}[H]", r"\centering", r"\caption{" + caption + "}", r"\label{" + label + "}",
             r"\begin{" + environment + "}" + width + "{" + columns + "}", r"\toprule",
             " & ".join(headers) + r" \\", r"\midrule"]
    lines += [" & ".join(row) + r" \\" for row in rows]
    lines += [r"\bottomrule", r"\end{" + environment + "}"]
    if note:
        lines += [r"\par\smallskip{\footnotesize " + note + "}"]
    lines += [r"\end{table}"]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-only", action="store_true")
    args = parser.parse_args()
    completed = json.loads((RESULTS / "completed.json").read_text(encoding="utf-8"))
    if not completed["external_evaluated"]:
        raise ValueError("the report requires completed validation and external evaluation")
    val = {(r["model"], r["condition"]): r for r in read_csv("validation_summary.csv")}
    external = {(r["model"], r["group"]): r for r in read_csv("external_summary.csv")}
    pairs = json.loads((RESULTS / "paired_validation.json").read_text(encoding="utf-8"))
    names = {"A": "Imagen", "B": "Imagen + GPS", "C": "Imagen + GPS + clima",
             "D": "Imagen + GPS + clima + sustrato"}
    rows = []
    md_rows = []
    for model in names:
        row = val[model, "normal"]
        n = int(row["n"])
        correct = round(n * float(row["accuracy"]))
        rows.append([model, names[model], number(row["macro_f1"]), number(100 * float(row["accuracy"]), 2),
                     integer(correct), integer(n - correct)])
        md_rows.append([model, names[model], number(row["macro_f1"]), number(100 * float(row["accuracy"]), 2),
                        str(correct), str(n - correct)])
    tables = {"@@VALIDATION_TABLE@@": latex_table(
        "Resultados por observación en validación; misma referencia visual y cohorte en todas las variantes.",
        "tab:validation", "lYrrrr", ["Modelo", "Modalidades", "F1 macro", r"Exactitud (\%)", "Aciertos", "Errores"], rows, True)}
    delta_rows = []
    for row in pairs:
        interval = row["ci95_delta_macro_f1"]
        delta_rows.append([row["comparison"].replace("-", "--"), number(row["delta_macro_f1"], signed=True),
                          "[" + number(interval[0], signed=True) + "; " + number(interval[1], signed=True) + "]",
                          str(row["fixed_errors"]), str(row["introduced_errors"])])
    tables["@@DELTA_TABLE@@"] = latex_table("Diferencias incrementales de F1 macro e intercambio de errores en validación.",
        "tab:deltas", "lrlrr", ["Comparación", r"$\Delta$ F1", "IC95 de la diferencia", "Corregidos", "Introducidos"], delta_rows,
        note="F1 expresado en escala 0--1; intervalos pareados por observación. Todos incluyen cero.")
    conditions = [("normal", "Entradas originales"), ("missing_all", "Todo el contexto ausente"),
        ("missing_geo", "GPS ausente"), ("missing_climate", "Clima ausente"), ("missing_substrate", "Sustrato ausente"),
        ("shuffled_geo", "GPS permutado"), ("shuffled_climate", "Clima permutado"),
        ("shuffled_substrate", "Sustrato permutado"), ("imprecise_geo", "GPS redondeado a un grado")]
    sensitivity_rows = []
    for condition, title in conditions:
        sensitivity_rows.append([title] + [number(val[m, condition]["macro_f1"]) if (m, condition) in val else "---" for m in ["C", "D"]])
    tables["@@SENSITIVITY_TABLE@@"] = latex_table("F1 macro por observación ante perturbaciones del contexto en validación.",
        "tab:sensitivity", "lrr", ["Condición", "C", "D"], sensitivity_rows)
    accuracy_rows, macro_rows = [], []
    for group, title in [("full", "Completo"), ("colombia", "Colombia"), ("orinoquia", "Orinoquía")]:
        base = external["A", group]
        keys = ["A", "B_raw", "C_raw", "D_raw"]
        accuracy_rows.append([title, integer(base["n"])] + [number(100 * float(external[k, group]["accuracy"]), 2) for k in keys])
        macro_rows.append([title, base["classes_present"]] + [number(external[k, group]["macro_f1_present_classes"], 4) for k in keys])
    tables["@@EXTERNAL_TABLE@@"] = latex_table("Exactitud externa por observación, con transferencia directa del contexto disponible.",
        "tab:external", "lrrrrr", ["Grupo", "Observaciones", r"A (\%)", r"B (\%)", r"C (\%)", r"D (\%)"], accuracy_rows) + "\n\n" + latex_table(
        "F1 macro externo sobre los géneros presentes en cada grupo.", "tab:external-f1", "lrrrrr",
        ["Grupo", "Géneros", "A", "B", "C", "D"], macro_rows,
        note="Los grupos tienen distinta composición taxonómica. El F1 de ocho géneros presentes no equivale al promedio sobre diez clases.")
    template = (REPORT / "template.tex").read_text(encoding="utf-8")
    for key, value in tables.items():
        if template.count(key) != 1:
            raise ValueError(f"missing or repeated template placeholder: {key}")
        template = template.replace(key, value)
    if "@@" in template:
        raise ValueError("unresolved report placeholder")
    (REPORT / "main.tex").write_text(template, encoding="utf-8", newline="\n")
    md = ["# Resultados de las semanas 9 y 10: integración de clima y sustrato", "",
        "El informe principal se entrega en [LaTeX](../week_09_10/main.tex) y [PDF](../week_09_10/main.pdf), "
        "siguiendo la estructura y presentación del informe de las semanas 7 y 8.", "",
        "## Alcance experimental", "",
        "Se ajustaron tres clasificadores de contexto en CPU con 17.799 observaciones de train y se conservaron "
        "las predicciones visuales de MobileNetV3-Large S5-C. La fusión usa un peso fijo de 0,1 para el contexto. "
        "No se realizó ajuste conjunto de la red visual ni una nueva evaluación en test interno.", "",
        "## Validación", "", "| Modelo | Modalidades | F1 macro | Exactitud (%) | Aciertos | Errores |",
        "|---|---|---:|---:|---:|---:|"]
    md += ["| " + " | ".join(row) + " |" for row in md_rows]
    md += ["", "La incorporación de clima corrigió dos errores e introdujo uno. El sustrato corrigió dos e "
        "introdujo dos respecto a C. Los intervalos de las diferencias incrementales de F1 incluyen cero; "
        "no se establece una mejora concluyente con este diseño.", "", "## Evaluación externa", "",
        "Se verificaron nuevamente 9.649 fotografías y se evaluaron 4.895 observaciones. Colombia incluye 3.348 "
        "observaciones y contiene las 309 de Orinoquía. Son fuentes ya evaluadas, no un conjunto nuevo e intacto. "
        "Sin clima ni sustrato compatibles, la evaluación mide transferencia geográfica y modalidades faltantes, "
        "no el aporte externo de esas modalidades completas.", "",
        "En Colombia, A obtuvo 88,32 % de exactitud; B/C directos, 87,54 %; D directo, 88,23 %. "
        "El filtro geográfico previo desactiva el contexto en todo el externo y coincide con A por construcción.", "",
        "## Reproducción", "", "```powershell", "python scripts/reports/build_s9_s10_report.py", "```", "",
        "Las tablas provienen de `frozen_context/validation_summary.csv`, `paired_validation.json` y `external_summary.csv`. "
        "La fuente `main.tex` es autónoma; también puede compilarse directamente con dos pasadas de pdfLaTeX.", ""]
    (ROOT / "reports/s9_s10/REPORTE.md").write_text("\n".join(md), encoding="utf-8", newline="\n")
    print(REPORT / "main.tex", flush=True)
    if args.source_only:
        return
    engine = shutil.which("pdflatex")
    if not engine:
        raise SystemExit("pdfLaTeX no está disponible; la fuente LaTeX se generó correctamente.")
    build = ROOT / "tmp/latex_week_09_10"
    build.mkdir(parents=True, exist_ok=True)
    for _ in range(2):
        result = subprocess.run([engine, "--disable-installer", "-interaction=nonstopmode", "-halt-on-error",
                                 f"-output-directory={build}", "main.tex"], cwd=REPORT, capture_output=True, text=True,
                                errors="replace")
        if result.returncode:
            print(result.stdout[-7000:])
            print(result.stderr[-2500:])
            raise SystemExit(result.returncode)
    shutil.copyfile(build / "main.pdf", REPORT / "main.pdf")
    PDF_ALIAS.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(REPORT / "main.pdf", PDF_ALIAS)
    print(REPORT / "main.pdf")


if __name__ == "__main__":
    main()
