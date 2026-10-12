"""Refresh the existing standalone LaTeX report from the consolidated saved results.

Plots in LaTeX contain their data inline; no external figures/project files are
needed by the editor. Exported PNG figures remain separate shareable artifacts.
"""
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs/20261004T133123181686Z_ma_resnet18_v002"
REPORT = ROOT / "reports/dataset_v002_training_report.tex"


def number(value, decimals=4):
    return f"{value:.{decimals}f}".replace(".", ",")


def count(value):
    return f"{value:,}".replace(",", r"\,")


def coords(history, column):
    return " ".join(f"({int(r.epoch)},{getattr(r,column):.8f})" for r in history.itertuples())


def main():
    history = pd.read_csv(RUN / "history.csv")
    metrics = {s:json.loads((RUN / f"metrics_{s}.json").read_text(encoding="utf-8")) for s in ("val","test")}
    summary = json.loads((ROOT / "reports/baseline_v002/summary.json").read_text(encoding="utf-8"))
    classes = [c["genus"] for c in metrics["val"]["observation_level"]["per_class"]]
    lines = [r"\section{Resultados y discusión}", "", r"\subsection{Finalización y selección de la arquitectura de referencia}", "",
             r"La ejecución consolidada es \archivo{20261004T133123181686Z_ma_resnet18_v002},",
             r"que heredó el historial y los estados de entrenamiento de las reanudaciones previas.",
             r"Se completaron \textbf{16 épocas}; la parada temprana ocurrió después de cinco",
             r"épocas consecutivas (12--16) sin superar la macro-F1 de observación en val",
             r"de la \textbf{época 11}. Este checkpoint fue recuperado para la evaluación final.",
             r"El registro de finalización confirma \textbf{una evaluación de test} y la",
             r"existencia de historial, predicciones y métricas finales.", "",
             r"La configuración efectiva corresponde a Python 3.12.10, PyTorch 2.14.1+cpu",
             r"y torchvision 0.29.1+cpu, con ejecución en CPU, AMP desactivado y cero",
             r"imágenes faltantes. La consolidación utiliza exclusivamente los archivos",
             r"guardados; no ejecuta otra inferencia ni vuelve a evaluar test.", "",
             r"\subsection{Evolución del entrenamiento}", "",
             r"F1 se expresa en la escala de 0 a 1. La exactitud se presenta como porcentaje",
             r"con símbolo explícito; la pérdida es entropía cruzada y el tiempo está en minutos.",
             r"\begin{center}\small", r"\begin{tabular}{rrrrrr}", r"\toprule",
             r"\textbf{Época} & \textbf{Pérdida train} & \textbf{F1 imagen val} & \textbf{F1 obs. val} & \textbf{Exactitud imagen} & \textbf{Minutos}\\",r"\midrule"]
    for r in history.itertuples():
        f1 = number(r.val_obs_macro_f1)
        if r.epoch == 11:
            f1 = r"\textbf{"+f1+"}"
        lines.append(f"{int(r.epoch)} & {number(r.train_loss)} & {number(r.val_img_macro_f1)} & {f1} & {number(r.val_img_accuracy*100,2)}"+r"\,\% & "+number(r.seconds/60,1)+r"\\")
    lines += [r"\bottomrule\end{tabular}\end{center}", "",
              r"La pérdida disminuye de 1,0928 a 0,5483. La macro-F1 de observación pasa",
              r"de 0,8454 en la primera época a 0,9545 en la seleccionada (mejora de",
              r"10,92 puntos porcentuales en la escala de F1). La época 16 alcanza",
              r"0,9121 en F1 de imagen, pero 0,9494 en F1 de observación; por ello no",
              r"sustituye al checkpoint de la época 11. La elección respeta el criterio",
              r"definido previamente y no utiliza el rendimiento de test.", "",
              f"La suma de los tiempos registrados para las 16 épocas es de {number(summary['total_recorded_epoch_seconds']/3600,2)} horas;",
              f"la media es {number(summary['mean_recorded_epoch_seconds']/60,1)} minutos por época.",
              r"Estos tiempos incluyen validación dentro de cada época, pueden reflejar",
              r"suspensiones y no equivalen al tiempo continuo entre el inicio y el fin",
              r"de una ejecución con pausas y reanudaciones.", ""]
    curve = r"""
\begin{figure}[htbp]
\centering
\begin{tikzpicture}
\begin{axis}[width=.94\textwidth,height=5.5cm,xlabel={Época},ylabel={Entropía cruzada},
xmin=1,xmax=16,xtick={1,4,8,11,16},grid=major,
tick label style={font=\small},label style={font=\small}]
\addplot[blue,mark=*,mark size=1.5pt] coordinates {@LOSS@};
\addplot[red,dashed] coordinates {(11,0.54)(11,1.1)};
\end{axis}
\end{tikzpicture}
\par\smallskip
\begin{tikzpicture}
\begin{axis}[width=.94\textwidth,height=6cm,xlabel={Época},ylabel={Macro-F1 (0--1)},
xmin=1,xmax=16,ymin=.77,ymax=.97,xtick={1,4,8,11,16},grid=major,
legend pos=south east,legend style={font=\small},tick label style={font=\small},label style={font=\small}]
\addplot[teal,mark=*,mark size=1.5pt] coordinates {@OBS@};
\addlegendentry{Val por observación}
\addplot[blue,mark=square*,mark size=1.5pt] coordinates {@IMG@};
\addlegendentry{Val por imagen}
\addplot[red,dashed,forget plot] coordinates {(11,.77)(11,.97)};
\end{axis}
\end{tikzpicture}
\caption{Curvas construidas a partir del historial guardado. La línea roja señala
la época 11; las cinco épocas siguientes no mejoran el criterio de selección.
El eje de F1 está restringido a 0,77--0,97 para mostrar su evolución.}
\end{figure}
""".replace("@LOSS@",coords(history,"train_loss")).replace("@OBS@",coords(history,"val_obs_macro_f1")).replace("@IMG@",coords(history,"val_img_macro_f1"))
    lines += [curve,r"\subsection{Evaluación final del checkpoint seleccionado}","",
              r"\begin{center}\small\begin{tabular}{llrrrr}",r"\toprule",
              r"\textbf{Conjunto} & \textbf{Unidad} & \textbf{N} & \textbf{Macro-F1} & \textbf{Exactitud} & \textbf{Top-3}\\",r"\midrule"]
    for split in ("val","test"):
        for level,label in (("image_level","Imagen"),("observation_level","Observación")):
            m=metrics[split][level]
            lines.append(f"{split.upper()} & {label} & {count(m['n'])} & {number(m['macro_f1'])} & {number(m['accuracy']*100,2)}"+r"\,\% & "+number(m['top3_accuracy']*100,2)+r"\,\%\\")
    lines += [r"\bottomrule\end{tabular}\end{center}","",
              r"La macro-F1 de observación es 0,9545 en val y 0,9463 en test; la",
              r"diferencia es de 0,82 puntos porcentuales en la escala de F1. La exactitud",
              r"por observación en test es 95,79\,\%. F1 combina precisión y sensibilidad",
              r"y no representa directamente el porcentaje de aciertos.","",
              r"En val se registran 499 errores de imagen y 140 de observación; en test,",
              r"638 y 170, respectivamente. El promedio de probabilidades de las fotografías",
              r"de un hallazgo permite una predicción conjunta con mayor rendimiento",
              r"observado que la evaluación de fotografías individuales. Las unidades y",
              r"sus denominadores son diferentes; estas diferencias no constituyen una",
              r"estimación causal del aporte de cada fotografía.","",
              r"\subsection{Rendimiento por género y soporte}","",
              r"\begin{center}\small\begin{tabular}{lrrrrrr}\toprule",
              r"& \multicolumn{3}{c}{\textbf{Val: observación}} & \multicolumn{3}{c}{\textbf{Test: observación}}\\",
              r"\textbf{Género} & \textbf{N} & \textbf{F1} & \textbf{Recall} & \textbf{N} & \textbf{F1} & \textbf{Recall}\\",r"\midrule"]
    for v,t in zip(metrics["val"]["observation_level"]["per_class"],metrics["test"]["observation_level"]["per_class"]):
        lines.append(f"{v['genus']} & {count(v['support'])} & {number(v['f1'])} & {number(v['recall']*100,2)}"+r"\,\% & "+f"{count(t['support'])} & {number(t['f1'])} & {number(t['recall']*100,2)}"+r"\,\%\\")
    lines += [r"\bottomrule\end{tabular}\end{center}","",
              r"El archivo de métricas por género conserva también precisión, F1 y",
              r"sensibilidad a nivel de imagen, junto con verdaderos positivos, falsos",
              r"positivos y falsos negativos para los dos conjuntos.","",
              r"\subsection{Matrices de confusión}","",
              r"Las filas representan el género real y las columnas el predicho; todos",
              r"los valores de las siguientes matrices son \textbf{conteos enteros de observaciones}.",
              r"Las columnas se abrevian en el mismo orden de las clases: Aman, Auri,",
              r"Gano, Geas, Heri, Lyco, Morc, Pleu, Tram y Xyla."]
    abbreviations=[c[:4] for c in classes]
    for split in ("val","test"):
        lines += [r"\begin{center}\scriptsize",r"\setlength{\tabcolsep}{3pt}",r"\begin{tabular}{lrrrrrrrrrr}\toprule",
                  r"\multicolumn{11}{c}{\textbf{"+split.upper()+r": matriz por observación}}\\",
                  "Real / Predicho & "+" & ".join(abbreviations)+r"\\\midrule"]
        for name,row in zip(classes,metrics[split]["observation_level"]["confusion_matrix"]):
            lines.append(name+" & "+" & ".join(str(n) for n in row)+r"\\")
        lines += [r"\bottomrule\end{tabular}\end{center}"]
    lines += ["",r"Se exportaron además matrices gráficas de imagen y observación, tanto",
              r"con conteos como normalizadas por fila. En estas últimas cada celda",
              r"indica explícitamente un porcentaje del soporte del género real.","",
              r"\subsection{Análisis de errores de validación}","",
              r"Los errores de val se reconstruyeron a partir de las predicciones guardadas",
              r"sin utilizar el modelo. Las probabilidades exportadas tienen seis decimales;",
              r"la agregación reprodujo exactamente la matriz de observación guardada.","",
              r"\begin{center}\small\begin{tabular}{llrr}\toprule",
              r"\textbf{Género real} & \textbf{Predicción} & \textbf{Errores obs.} & \textbf{Soporte real}\\\midrule",
              r"Amanita & Lycoperdon & 22 & 904\\",
              r"Ganoderma & Trametes & 14 & 335\\",
              r"Trametes & Ganoderma & 10 & 756\\",
              r"Trametes & Lycoperdon & 9 & 756\\",
              r"Hericium & Trametes & 5 & 66\\",
              r"\bottomrule\end{tabular}\end{center}","",
              r"Hericium registra siete observaciones incorrectas de 66; cinco se",
              r"confunden con Trametes. Geastrum presenta tres errores de 79. Morchella",
              r"acierta las once observaciones de val, pero tiene cuatro imágenes",
              r"incorrectas de 21: el acierto por hallazgo no implica acierto en cada fotografía.","",
              r"El recall de Morchella de 100\,\% tiene un intervalo Wilson aproximado",
              r"al 95\,\% de 74,12\,\%--100\,\%; Hericium, 79,69\,\%--94,77\,\%;",
              r"y Geastrum, 89,42\,\%--98,70\,\%. Estos intervalos describen la incertidumbre",
              r"del recall debida al soporte reducido, bajo un supuesto binomial; no son",
              r"intervalos de F1 ni una corrección de posibles dependencias entre observaciones.","",
              r"La inspección visual de ejemplos de val muestra tomas parciales de",
              r"superficies, ejemplares deteriorados y una imagen de microscopía. Algunos",
              r"errores reciben probabilidades elevadas. Estos ejemplos orientan una",
              r"revisión taxonómica y de condiciones de captura, sin demostrar que la",
              r"etiqueta sea incorrecta ni modificarla automáticamente.","",
              r"Se conservan listas completas de errores por imagen y observación, con",
              r"identificadores y probabilidades, y una galería local de ejemplos. El",
              r"análisis detallado para orientar los siguientes experimentos se limita",
              r"a val; las cifras de test se reportan como evaluación final descriptiva.","",
              r"\subsection{Conservación de la línea base}","",
              r"El checkpoint de la época 11, la configuración efectiva, el manifiesto",
              r"del dataset, el historial, las métricas y las predicciones finales se",
              r"copiaron a un archivo de conservación independiente de la carpeta de corrida.",
              r"La igualdad de los hashes SHA-256 entre origen y copia fue verificada,",
              r"y los archivos de la corrida original permanecen sin cambios.","",
              r"La huella SHA-256 del checkpoint conservado es:",
              r"\begin{center}\small\ttfamily",summary["checkpoint_sha256"],r"\end{center}","",
              r"La configuración efectiva es la registrada durante el entrenamiento.",
              r"La conservación incluye hashes de cada archivo y permite utilizar esta",
              r"línea base como referencia para S5 y S6.",""]
    section="\n".join(lines)
    text=REPORT.read_text(encoding="utf-8")
    text=re.sub(r"\\section\{Resultados(?: parciales)? y discusión\}.*?(?=\\section\{Limitaciones del experimento\})",lambda _:section,text,flags=re.S)
    if r"\usepackage{pgfplots}" not in text:
        text=text.replace(r"\usepackage{booktabs,tabularx,array,enumitem,amsmath,hyperref}",r"\usepackage{booktabs,tabularx,array,enumitem,amsmath,hyperref}"+"\n"+r"\usepackage{pgfplots}"+"\n"+r"\pgfplotsset{compat=1.18}")
    REPORT.write_text(text,encoding="utf-8")
    print(f"Updated results section in {REPORT}")


if __name__=="__main__":
    main()
