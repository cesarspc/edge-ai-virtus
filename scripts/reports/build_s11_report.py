"""Generate and compile the S11 LaTeX report from saved evaluation results."""
import argparse
import csv
import json
import shutil
import subprocess

from build_s9_s10_report import ROOT, integer, latex_table, number

REPORT = ROOT / "reports/week_11"
PDF_ALIAS = ROOT / "output/pdf/reporte_s11_virtus.pdf"
MODALITIES = {
    "A": "Imagen", "B": "Imagen + GPS", "C": "Imagen + GPS + clima",
    "D": "Imagen + GPS + clima + sustrato", "E": "Imagen + clima",
    "F": "Imagen + GPS + sustrato",
}
CONDITIONS = {
    "missing_all": "Todo el contexto ausente", "missing_geo": "GPS ausente",
    "imprecise_geo": "GPS redondeado a 1 grado", "shuffled_geo": "GPS permutado",
    "missing_climate": "Clima ausente", "shuffled_climate": "Clima permutado",
    "missing_substrate": "Sustrato ausente", "shuffled_substrate": "Sustrato permutado",
}


def generate_source():
    evidence = json.loads((REPORT / "evidencia.json").read_text(encoding="utf-8"))
    val = evidence["validation_by_model"]
    ext = evidence["external_by_model"]
    with (REPORT / "validacion.csv").open(encoding="utf-8", newline="") as stream:
        scores = {(r["model"], r["condition"]): float(r["macro_f1"]) for r in csv.DictReader(stream)}
    validation = []
    for model, modalities in MODALITIES.items():
        row = val[model]
        correct = round(row["n"] * row["accuracy"])
        validation.append([model, modalities, number(row["macro_f1"]), number(100 * row["accuracy"], 2),
                           integer(correct), integer(row["n"] - correct)])
    sensitivity = [[title] + [number(100 * (scores[model, condition] - val[model]["macro_f1"]), 3, signed=True)
                              if (model, condition) in scores else "---" for model in "BCDEF"]
                   for condition, title in CONDITIONS.items()]
    external = [[model, number(groups["total"]["macro_f1"]), number(100 * groups["total"]["accuracy"], 2),
                 number(groups["colombia"]["macro_f1"]), number(groups["orinoquia"]["macro_f1"])]
                for model, groups in ext.items()]
    genera = [[r["genus"], number(100 * r["precision"], 2), number(100 * r["recall"], 2),
               number(r["f1"], 5), integer(r["support"])] for r in ext["A"]["total"]["per_class"]]
    comparison = next(r for r in evidence["paired_vs_A"] if r["model"] == "C")
    lower, upper = (number(value * 100, 3, signed=True) for value in comparison["ci95_delta_macro_f1"])
    replacements = {
        "VALIDATION": latex_table("Resultados por observación en validación.", "tab:val", "lYrrrr",
                                  ["Modelo", "Modalidades", "F1 macro", r"Exactitud (\%)", "Aciertos", "Errores"],
                                  validation, flexible=True),
        "DELTA": number(100 * comparison["delta_macro_f1"], 3, signed=True),
        "INTERVAL": f"{lower} a {upper}",
        "SENSITIVITY": latex_table("Cambio de F1 macro en puntos sobre escala 0--100 respecto a la condición original.",
                                   "tab:sens", "lrrrrr", ["Condición", *"BCDEF"], sensitivity,
                                   note="El guion indica que la modalidad queda fuera de la variante."),
        "EXTERNAL": latex_table("Transferencia externa por observación.", "tab:ext", "lrrrr",
                                ["Modelo", "F1 total", r"Exactitud total (\%)", "F1 Colombia", "F1 Orinoquía"],
                                external, note="El F1 macro promedia los diez géneros en todos los grupos."),
        "GENERA": latex_table("Precisión, sensibilidad y F1 de A en el conjunto externo completo.",
                              "tab:genera", "lrrrr", ["Género", r"Precisión (\%)", r"Sensibilidad (\%)", "F1", "Soporte"],
                              genera),
    }
    source = (REPORT / "template.tex").read_text(encoding="utf-8-sig")
    for name, value in replacements.items():
        source = source.replace(f"@@{name}@@", value)
    (REPORT / "main.tex").write_text(source, encoding="utf-8", newline="\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-only", action="store_true")
    args = parser.parse_args()
    generate_source()
    if args.source_only:
        return
    engine = shutil.which("pdflatex")
    if not engine:
        raise SystemExit("pdfLaTeX no está disponible; se generó main.tex.")
    build = ROOT / "tmp/latex_week_11"
    build.mkdir(parents=True, exist_ok=True)
    for _ in range(2):
        result = subprocess.run([engine, "-interaction=nonstopmode", "-halt-on-error",
                                 f"-output-directory={build}", "main.tex"],
                                cwd=REPORT, capture_output=True, text=True, errors="replace")
        if result.returncode:
            raise SystemExit(result.stdout[-6500:] + result.stderr[-2500:])
    shutil.copyfile(build / "main.pdf", REPORT / "main.pdf")
    PDF_ALIAS.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(REPORT / "main.pdf", PDF_ALIAS)


if __name__ == "__main__":
    main()
