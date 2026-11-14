"""Build a plain-Spanish S9/S10 report from completed measured results.

Requires reportlab (or the local tmp/report_dependencies installation).
"""
from pathlib import Path
import json
import sys
from html import escape

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tmp/report_dependencies"))
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether

RESULTS = ROOT / "reports/s9_s10/frozen_context"
OUT = ROOT / "output/pdf/reporte_s9_s10_virtus.pdf"
LABELS = {"A": "Solo foto", "B": "Foto + GPS", "C": "Foto + GPS + clima", "D": "Foto + GPS + clima + sustrato"}


def percent(value, digits=2):
    return f"{100 * float(value):.{digits}f}".replace(".", ",")


def table_md(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"] +
                     ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def main():
    completion = json.loads((RESULTS / "completed.json").read_text(encoding="utf-8"))
    if not completion["external_evaluated"]:
        raise ValueError("Complete the external audit before building the final report")
    val = pd.read_csv(RESULTS / "validation_summary.csv")
    normal = val[val.condition.eq("normal")].set_index("model")
    pairs = {r["comparison"]: r for r in json.loads((RESULTS / "paired_validation.json").read_text())}
    ext = pd.read_csv(RESULTS / "external_summary.csv")
    provenance = json.loads((RESULTS / "external/provenance.json").read_text())
    validation_rows = []
    for model, row in normal.iterrows():
        correct = round(row.accuracy * row.n)
        validation_rows.append([model, LABELS[model], f"{correct:,}".replace(",", "."),
                                str(int(row.n) - correct), percent(row.accuracy), percent(row.macro_f1)])
    external_rows = []
    for group, name in [("full", "Completo"), ("colombia", "Colombia"), ("orinoquia", "Orinoquía")]:
        selected = ext[ext.group.eq(group)].set_index("model")
        external_rows.append([name, str(int(selected.loc["A", "n"])), percent(selected.loc["A", "accuracy"]),
            percent(selected.loc["C_raw", "accuracy"]), percent(selected.loc["D_raw", "accuracy"]),
            percent(selected.loc["D_guarded", "accuracy"])])
    climate = pairs["C-B"]
    substrate = pairs["D-C"]
    result_text = (f"El clima corrigió {climate['fixed_errors']} errores e introdujo {climate['introduced_errors']}: "
                   f"el saldo fue {climate['fixed_errors'] - climate['introduced_errors']} acierto adicional. "
                   f"Al sumar sustrato se corrigieron {substrate['fixed_errors']} errores y aparecieron "
                   f"{substrate['introduced_errors']}; el total de aciertos no aumentó.")
    interval_c = climate["ci95_delta_macro_f1"]
    interval_d = substrate["ci95_delta_macro_f1"]
    conclusions = "Las diferencias son demasiado pequeñas para demostrar una mejora estable en este ensayo. No se cambió el peso del contexto para buscar un resultado favorable."
    methodology = (
        "Usamos las respuestas guardadas de MobileNetV3-Large S5-C, la misma versión visual en validación y en la evaluación externa. "
        "Entrenamos tres clasificadores de contexto en CPU con 17.799 observaciones de entrenamiento. "
        "Cada uno aprende únicamente de GPS; GPS y clima; o GPS, clima y sustrato. "
        "Combinamos 90 % de la respuesta fotográfica con 10 % de la respuesta del contexto. "
        "Ese peso, la regularización y el resto de las reglas quedaron fijos antes de calcular los resultados. "
        "Si no hay contexto utilizable, la respuesta es exactamente la de la foto. "
        "No volvimos a entrenar la red de imágenes: es una prueba de combinación final con la parte visual congelada.")
    external_explanation = (
        f"Verificamos nuevamente las huellas digitales de las {provenance['images_freshly_hash_verified']:,} fotografías externas. "
        "El conjunto completo tiene 4.895 observaciones: incluye 3.348 de Colombia y 1.547 de MIND.Funga. "
        "Dentro de las colombianas, 309 corresponden a la Orinoquía. Son grupos que se contienen entre sí; no se suman. "
        "No se usaron para entrenar, elegir pesos ni seleccionar una variante. Ya habían sido evaluados en una fase anterior: "
        "esta es una revisión de robustez sobre un conjunto conocido, no un test externo nuevo e intacto.")
    absent_explanation = (
        "La tabla externa no trae las 19 variables CHELSA ni el sustrato necesario para estos modelos. "
        "Se dejaron ausentes. Por eso la evaluación externa mide el comportamiento con GPS y datos faltantes; "
        "no demuestra que el clima o el sustrato mejoren la identificación fuera de FungiTastic. "
        "La ubicación aprendida principalmente en Dinamarca puede ser una pista equivocada en Colombia. "
        "La columna sin protección aplica directamente el contexto disponible. La protección geográfica desactiva "
        "el contexto fuera de los límites de ubicación aprendidos solo de train (percentiles 0,5 y 99,5). "
        "Es una regla previa, no un ajuste a partir de las etiquetas externas; no es un detector de dominio validado. "
        "En esta muestra las ubicaciones externas no pasan esa regla; la ruta protegida usa la foto.")
    s10_text = (
        "Sustrato significa sobre qué crece el hongo: madera, suelo, hojarasca, etc. "
        "No es lo mismo que hábitat, que describe el entorno. En train había 1.811 registros sin detalle, frente a uno en val: "
        "todos decían solamente madera. Unificamos madera viva, muerta y sin detalle bajo la misma categoría amplia. "
        "La calidad o el año no entran como pistas para predecir el género. Las categorías desconocidas tienen una marca propia.")
    s9_text = (
        "S9 añade 19 variables de clima CHELSA: describen condiciones típicas de 1981 a 2010, no el tiempo del día de la foto. "
        "Hay clima completo en 17.782 de 17.799 observaciones de train y en 3.442 de 3.452 de val. "
        "Los datos faltantes se conservan con máscaras; no eliminamos esos hongos de la comparación. "
        "Este producto sustituye explícitamente el ERA5-Land propuesto para este ensayo. No incluye humedad relativa.")
    limits = [
        "Se evalúan diez géneros, no una identificación definitiva de especie.",
        "Las observaciones de validación ya se usaron para seleccionar el modelo visual. No son una prueba nueva e independiente.",
        "Las normales climáticas pueden incluir años posteriores a 745 registros antiguos de train; el análisis es retrospectivo.",
        "Sustrato y hábitat son anotaciones de la fuente; no suponemos que una persona pueda reconocerlos al tomar una foto.",
        "La microscopía, las fotos con poco detalle y las vistas heterogéneas necesitan revisión específica; se conserva la revisión preliminar de seis casos de S8.",
        "No se ejecutó un ajuste conjunto en GPU ni una prueba de latencia o funcionamiento real en Android.",
    ]
    future = [
        "Conservar la foto como referencia: este ensayo no justifica dar por ganada la incorporación de clima o sustrato.",
        "En el equipo con GPU, repetir A/B/C/D con el mismo protocolo y varias semillas; no mezclar B solo GPS con B foto + GPS.",
        "Conseguir clima externo compatible y sustrato fiable antes de afirmar una mejora multimodal en Colombia.",
        "En Android, foto y fecha se capturan localmente; GPS depende de permiso y señal. Clima remoto llega al sincronizar. Sustrato debe permitir 'no sé'.",
    ]
    md = ["# S9 y S10 de Virtus, explicado sin enredos", "", "Fecha del reporte: 13 de noviembre de 2026.", "",
        "## Qué hicimos", "", methodology, "", "## Resultado en 3.452 observaciones de validación", "",
        table_md(["Modelo", "Qué recibe", "Aciertos", "Errores", "Acierto %", "F1 por género /100"], validation_rows),
        "", "F1 por género da el mismo peso a cada género, aunque uno tenga muchas más observaciones. No es el porcentaje de aciertos.",
        "", result_text, "", conclusions, "", "## S9: clima", "", s9_text, "",
        f"C menos B: {100*climate['delta_macro_f1']:+.4f} puntos de F1; intervalo del 95 % "
        f"[{100*interval_c[0]:+.4f}, {100*interval_c[1]:+.4f}] puntos. Incluye cero.",
        "", "## S10: sustrato", "", s10_text, "",
        f"D menos C: {100*substrate['delta_macro_f1']:+.4f} puntos de F1; intervalo del 95 % "
        f"[{100*interval_d[0]:+.4f}, {100*interval_d[1]:+.4f}] puntos. Incluye cero.",
        "", "Los intervalos se calcularon con 2.000 remuestreos pareados por observación, semilla 42. Describen esta comparación; no corrigen el uso previo de val.",
        "", "## Evaluación externa", "", external_explanation, "",
        table_md(["Grupo", "Observaciones", "Foto %", "C sin protección %", "D sin protección %", "D protegido %"], external_rows),
        "", absent_explanation, "", "## Límites", ""] + [f"- {s}" for s in limits] + ["", "## Qué sigue", ""] + [f"- {s}" for s in future] + [
        "", "## Reproducir y revisar los datos", "", "```powershell", "python scripts/data/prepare_s8_s9_context.py",
        "python scripts/train/run_s9_s10_frozen.py --external", "python scripts/reports/build_s9_s10_report.py", "```", "",
        "Resultados completos: `reports/s9_s10/frozen_context/`. Configuración fija: `configs/experiments/s9_s10_frozen_context.yaml`.",
        "Modelos de contexto: `models/B.json`, `C.json`, `D.json` (categorías, escalas y coeficientes).",
        "Sensibilidades de ausencia/permutación: `validation_summary.csv`. Intervalos: `paired_validation.json` y `paired_external.json`.",
        "Las probabilidades visuales se reutilizaron; no se recalcularon. Los archivos conservan hashes y procedencia. No se evaluó test interno.",
        "", "Fuentes: [datos climáticos FungiTastic](https://arxiv.org/html/2408.13632v2), "
        "[contrato Android](../../docs/android_metadatos.md), [revisión visual](../s9/visual_review/REVIEW.md).", ""]
    report_path = ROOT / "reports/s9_s10/REPORTE_FAMILIAR.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(md), encoding="utf-8")

    pdfmetrics.registerFont(TTFont("Arial", "C:/Windows/Fonts/arial.ttf"))
    pdfmetrics.registerFont(TTFont("ArialBold", "C:/Windows/Fonts/arialbd.ttf"))
    pdfmetrics.registerFontFamily("Arial", normal="Arial", bold="ArialBold", italic="Arial", boldItalic="ArialBold")
    ink = colors.HexColor("#153B3B")
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="BodyES", fontName="Arial", fontSize=10.5, leading=15.2, spaceAfter=10, textColor=colors.HexColor("#263838")))
    styles.add(ParagraphStyle(name="TitleES", fontName="ArialBold", fontSize=28, leading=33, spaceAfter=15, textColor=ink))
    styles.add(ParagraphStyle(name="HeadingES", fontName="ArialBold", fontSize=17, leading=22, spaceBefore=12, spaceAfter=10, textColor=ink))
    styles.add(ParagraphStyle(name="SmallES", fontName="Arial", fontSize=8.5, leading=12, spaceAfter=8, textColor=colors.HexColor("#556565")))
    styles.add(ParagraphStyle(name="CellES", fontName="Arial", fontSize=8.3, leading=11))
    def p(text, style="BodyES"):
        return Paragraph(escape(text), styles[style])
    def table(headers, rows, widths):
        data = [[p(str(c), "CellES") for c in headers]] + [[p(str(c), "CellES") for c in row] for row in rows]
        t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E1EEEE")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F4F7F7")]),
            ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 7),
            ("RIGHTPADDING", (0, 0), (-1, -1), 7), ("TOPPADDING", (0, 0), (-1, -1), 9),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 9), ("LINEBELOW", (0, 0), (-1, 0), .8, ink)]))
        return t
    story = [p("VIRTUS  /  SEMANAS 9 Y 10", "SmallES"), p("¿Ayudan el clima y el lugar donde crece un hongo?", "TitleES"),
        p("Un reporte para entender el avance sin tener que saber de inteligencia artificial."),
        p("13 de noviembre de 2026", "SmallES"), p("La respuesta que tenemos hoy", "HeadingES"),
        p(result_text), p(conclusions), p("Cómo hicimos una comparación justa", "HeadingES"), p(methodology),
        p("Lo que pasó en validación", "HeadingES"),
        table(["Caso", "Información", "Aciertos", "Errores", "Acierto"],
              [[r[0], r[1], r[2], r[3], r[4] + " %"] for r in validation_rows], [1.4*cm, 7.1*cm, 2.4*cm, 2*cm, 2.5*cm]),
        Spacer(1, 10), p("Son las mismas 3.452 observaciones en las cuatro variantes. Una observación puede tener varias fotos; cuenta una sola vez.", "SmallES"),
        PageBreak(), p("S9: darle al modelo una pista del clima", "TitleES"), p(s9_text),
        p("¿Mejoró?", "HeadingES"), p(result_text.split("Al sumar")[0]),
        p(f"La diferencia de F1 por género entre C y B fue {100*climate['delta_macro_f1']:+.3f} puntos sobre 100. "
          f"El intervalo estimado va de {100*interval_c[0]:+.3f} a {100*interval_c[1]:+.3f}; incluye tanto una pequeña pérdida como una pequeña mejora."),
        p("F1 por género es una nota que da el mismo peso a cada género. Sirve para que los géneros numerosos no escondan los errores de los pequeños.", "SmallES"),
        p("S10: sumar sobre qué crece", "HeadingES"), p(s10_text),
        p(f"D tuvo {substrate['fixed_errors']} correcciones y {substrate['introduced_errors']} errores nuevos respecto a C. "
          "No aumentó el total de aciertos. La pequeña diferencia en la nota por género tampoco demuestra una mejora estable."),
        p("Qué significa para una persona en campo", "HeadingES"),
        p("No deberíamos exigir saber el sustrato para usar la aplicación. Debe existir la opción 'no sé'. "
          "La foto sigue siendo la entrada indispensable; el contexto se usa solo cuando existe y es apropiado."),
        PageBreak(), p("La prueba fuera de FungiTastic", "TitleES"), p(external_explanation),
        table(["Grupo", "Casos", "Foto", "C directo", "D directo", "D con filtro"],
              [[r[0], r[1]] + [v + " %" for v in r[2:]] for r in external_rows], [2.5*cm, 1.6*cm, 2.4*cm, 3*cm, 3*cm, 2.9*cm]),
        Spacer(1, 12), p("Qué se pudo comprobar y qué no", "HeadingES"), p(absent_explanation),
        p("La Orinoquía es parte del grupo de Colombia. No son dos pruebas independientes. Los grupos tienen distinta composición de géneros.", "SmallES"),
        PageBreak(), p("Lo que aprendimos y lo que falta", "TitleES"),
        p("No necesitamos que todos los experimentos mejoren", "HeadingES"),
        p("Este ensayo no aporta evidencia suficiente para decir que sumar clima o sustrato ya haga mejor el sistema. "
          "Encontrar ese límite también es un resultado: evita complicar la aplicación sin un beneficio demostrado."),
        p("Límites que acompañan los números", "HeadingES")]
    for item in limits:
        story.append(p("• " + item))
    story.append(p("Los próximos pasos", "HeadingES"))
    for item in future:
        story.append(p("• " + item))
    story += [Spacer(1, 8), p("Dónde revisar el detalle", "HeadingES"),
        p("El reporte de texto y las tablas están en reports/s9_s10/. La configuración, los modelos de contexto, "
          "las sensibilidades y los intervalos quedan guardados para reproducir el experimento. "
          "La revisión preliminar de fotografías y el contrato de metadatos Android siguen disponibles.", "SmallES"),
        p("Fuente del clima: Picek y colaboradores, FungiTastic, sección 2.1 (arxiv.org/html/2408.13632v2).", "SmallES")]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    def footer(canvas, doc):
        canvas.setStrokeColor(colors.HexColor("#C5D5D5"))
        canvas.line(2*cm, 1.5*cm, 19.4*cm, 1.5*cm)
        canvas.setFont("Arial", 8)
        canvas.setFillColor(ink)
        canvas.drawString(2*cm, 1.05*cm, "VIRTUS · S9 y S10 · Ensayo con parte visual congelada")
        canvas.drawRightString(19.4*cm, 1.05*cm, str(doc.page))
    doc = SimpleDocTemplate(str(OUT), pagesize=(21.4*cm, 29.7*cm), rightMargin=2*cm, leftMargin=2*cm,
                            topMargin=1.8*cm, bottomMargin=2*cm,
                            title="Virtus - S9 y S10 para toda la familia", author="Proyecto Virtus")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(OUT)
    print(report_path)


if __name__ == "__main__":
    main()
