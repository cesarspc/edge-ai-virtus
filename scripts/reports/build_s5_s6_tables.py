"""Render LaTeX tables from analyze_s5_s6.py outputs; no model inference or test execution."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from tqdm import tqdm

ROOT=Path(__file__).resolve().parents[2]
S5=ROOT/"reports/s5/comparison"
S6=ROOT/"reports/s6/comparison"
MODELS=["R","A","B","C","D"]
NAMES={"R":"ResNet-18",**{x:f"MobileNet {x}" for x in "ABCD"}}
LEVELS={"genus":"Género","family":"Familia","order":"Orden","class":"Clase","phylum":"Filo"}


def number(value,d=3):
    if pd.isna(value):return "--"
    return f"{value:.{d}f}".replace(".",",")


def integer(value):
    return f"{int(value):,}".replace(",",r"\,")


def interval(value,low,high,d=3):
    return f"{number(value,d)} [{number(low,d)}; {number(high,d)}]"


def table(headers,rows,align,caption,source,label,font="small"):
    return "\n".join([r"\begin{table}[H]",r"\centering",f"\\{font}",f"\\caption{{{caption}}}",f"\\label{{{label}}}",
        f"\\begin{{tabularx}}{{\\linewidth}}{{{align}}}",r"\toprule"," & ".join(f"\\textbf{{{x}}}" for x in headers)+r"\\",r"\midrule",
        *[" & ".join(map(str,row))+r"\\" for row in rows],r"\bottomrule",r"\end{tabularx}",f"\\source{{{source}}}",r"\end{table}",""])


def main():
    results=pd.read_csv(S5/"validation_results.csv").set_index("model")
    pc=pd.read_csv(S5/"validation_per_genus.csv")
    dist=pd.read_csv(S5/"class_distribution.csv")
    paired=pd.read_csv(S5/"bootstrap_paired.csv")
    taxa=pd.read_csv(S6/"taxonomic_summary.csv")
    taxpc=pd.read_csv(S6/"taxonomic_per_class.csv")
    winner=json.loads((S5/"validation_recommendation.json").read_text())["winner_point_estimate"]
    files={}
    files[S5/"results_metrics_table.tex"]=table(["Modelo",r"F1 obs. [IC 95\,\%]","F1 foto","Acierto obs."],
       [[NAMES[m],interval(results.loc[m,"f1_observation"],results.loc[m,"ci95_low"],results.loc[m,"ci95_high"]),number(results.loc[m,"f1_image"]),number(results.loc[m,"accuracy_observation"]*100,1)+r"\,\%"] for m in MODELS],
       "Y r r r","Resultados de validación; IC percentil por observación.","reports/s5/comparison/validation_results.csv","tab:results")
    files[S5/"training_table.tex"]=table(["Modelo","Mejor vuelta","Vueltas","Horas","Dispositivo / precisión"],
       [[NAMES[m],int(results.loc[m,"best_epoch"]),int(results.loc[m,"epochs"]),number(results.loc[m,"hours"],2),"CPU / FP32" if m in ["R","A"] else "GTX 1650 SUPER / AMP"] for m in MODELS],
       "Y r r r l","Duración y selección del checkpoint. Todos los entrenamientos finalizaron por parada temprana; las horas incluyen entrenamiento y validación por época.","reports/s5/comparison/validation_results.csv","tab:training")
    configs={m:json.loads((ROOT/results.loc[m,"run"]/"run.json").read_text())["config"] for m in "ABCD"}
    config_rows=[]
    getters=[("Aprendizaje inicial",lambda c:number(c["training"]["lr"],4)),
      ("Pesos por clase",lambda c:"Ninguno" if c["training"]["class_weighting"]=="none" else r"$1/\sqrt{n_c}$"),
      ("Fotos por paso",lambda c:str(c["training"]["batch_size"])),
      ("Lote de validación",lambda c:str(c["evaluation"].get("batch_size",c["training"]["batch_size"]*2))),
      ("Entrada (píxeles)",lambda c:f"{c['input']['size']} $\\times$ {c['input']['size']}"),
      ("Máximo de vueltas",lambda c:str(c["training"]["epochs"])),
      ("Paciencia",lambda c:str(c["training"]["early_stopping_patience"])),
      ("Optimizador",lambda c:c["training"]["optimizer"].replace("adamw","AdamW")),
      ("Decaimiento de pesos",lambda c:number(c["training"]["weight_decay"],2)),
      ("Suavizado de etiquetas",lambda c:number(c["training"]["label_smoothing"],1)),
      ("Programa de aprendizaje",lambda c:"Coseno"),
      ("Calentamiento (vueltas)",lambda c:str(c["training"]["warmup_epochs"])),
      ("Procesos de carga",lambda c:str(c["training"]["num_workers"])),
      ("Semilla",lambda c:str(c["training"]["seed"])),
      ("Determinismo exigido",lambda c:"Sí" if c["training"]["deterministic"] else "No")]
    for name,getter in getters:
        values=[getter(configs[m]) for m in "ABCD"]
        config_rows.append([name,*[f"\\textbf{{{v}}}" if v!=values[0] else v for v in values]])
    config_rows += [["Precisión efectiva","FP32",r"\textbf{AMP}",r"\textbf{AMP}",r"\textbf{AMP}"],
                    ["Capas congeladas","Ninguna","Ninguna","Ninguna","Ninguna"]]
    files[S5/"configuration_table.tex"]=table(["Ajuste","A","B","C","D"],config_rows,"Y l l l l",
      "Configuración de los ensayos. Negrita: ajustes que cambian respecto de A. Se indica el lote de validación utilizado en cada entorno.",
      "reports/s5/comparison/configuration.csv","tab:configuration")
    rows=[]
    for genus,g in pc.groupby("genus",sort=False):
        values=g.set_index("model")
        rows.append([f"\\textit{{{genus}}}",int(g.iloc[0].support),*[number(values.loc[m,"f1"]) for m in MODELS]])
    files[S5/"f1_genus_table.tex"]=table(["Género","Obs.","ResNet","A","B","C","D"],rows,"Y r r r r r r",
       "F1 por género en validación, a partir de las observaciones.","reports/s5/comparison/validation_per_genus.csv","tab:genus")
    rows=[]
    for genus,g in pc.groupby("genus",sort=False):
        values=g.set_index("model")
        rows.append([f"\\textit{{{genus}}}",*[f"{number(values.loc[m,'precision'])} / {number(values.loc[m,'recall'])}" for m in "ABCD"]])
    files[S5/"precision_recall_table.tex"]=table(["Género","A: P / R","B: P / R","C: P / R","D: P / R"],rows,"Y r r r r",
       "Precisión (P) y recuperación o recall (R), por observación. P baja cuando se asigna el género a hongos de otro género; R baja cuando se omiten hongos del género verdadero.",
       "reports/s5/comparison/validation_per_genus.csv","tab:precision-recall","footnotesize")
    rows=[]
    for genus in pc.genus.unique():
        g=dist[dist.genus.eq(genus)].set_index("split")
        rows.append([f"\\textit{{{genus}}}",*[f"{integer(g.loc[s,key])} ({number(g.loc[s,pct],1)}\\,\\%)" for s in ["train","val","test"] for key,pct in [("observations","observation_pct"),("photos","photo_pct")]]])
    files[S5/"distribution_table.tex"]=table(["Género","Train obs.","Train fotos","Val obs.","Val fotos","Test obs.","Test fotos"],rows,"Y r r r r r r",
       "Distribución por partición: cantidad entera y porcentaje dentro de la partición y unidad correspondientes.",
       "reports/s5/comparison/class_distribution.csv","tab:distribution","scriptsize")
    ratios=pd.read_csv(S5/"imbalance_ratios.csv")
    files[S5/"imbalance_ratios_table.tex"]=table(["Partición","Observaciones","Fotos","Mayor/menor obs.","Mayor/menor fotos"],
       [[r.split,integer(r.observations),integer(r.photos),number(r.observation_max_min,2),number(r.photo_max_min,2)] for r in ratios.itertuples()],
       "Y r r r r","Totales y razón entre el género más abundante y el menos abundante.","reports/s5/comparison/imbalance_ratios.csv","tab:ratios")
    files[S5/"paired_table.tex"]=table(["Comparación","Diferencia F1",r"IC 95\,\% pareado","Incluye cero"],
       [[f"{r.model} menos {r.reference}",number(r.delta_f1,4),f"[{number(r.ci95_low,4)}; {number(r.ci95_high,4)}]","Sí" if r.contains_zero else "No"] for r in paired.itertuples()],
       "Y r r l","Diferencias por observación. R denota ResNet-18. Intervalos exploratorios sin ajuste por comparaciones múltiples.",
       "reports/s5/comparison/bootstrap_paired.csv","tab:paired")
    tax=pd.read_csv(ROOT/"data/taxonomy/v002/genus_taxonomy.csv")
    files[S6/"taxonomy_map_table.tex"]=table(["Género","Familia","Orden","Clase","Filo","Especies"],
       [[f"\\textit{{{r['genus']}}}",r["family"],r["order"],r["class"],r["phylum"],r["n_species"]] for r in tax.to_dict("records")],
       "l Y Y Y Y r","Taxonomía de FungiTastic y número de especies presentes en el dataset.","data/taxonomy/v002/genus_taxonomy.csv","tab:taxmap","scriptsize")
    for split,models,filename in [("val",["R",winner],"taxonomy_validation_table.tex"),("test",["R"],"taxonomy_historical_test_table.tex")]:
        rows=[]
        for model in models:
            for level in LEVELS:
                r=taxa[(taxa.model.eq(model))&(taxa.split.eq(split))&(taxa.level.eq(level))].iloc[0]
                rows.append([NAMES[model],LEVELS[level],interval(r.macro_f1,r.ci95_low,r.ci95_high),interval(r.accuracy*100,r.accuracy_ci95_low*100,r.accuracy_ci95_high*100,1)])
        cap="Validación por observación: F1 macro y exactitud porcentual con IC 95\\,\\% en la propia tabla." if split=="val" else "Test histórico de ResNet-18: resultados ya guardados, sin nueva inferencia. No intervienen en la selección de S5."
        files[S6/filename]=table(["Modelo","Nivel",r"F1 [IC 95\,\%]",r"Exactitud, \% [IC 95\,\%]"],rows,"l l Y Y",cap,
          "reports/s6/comparison/taxonomic_summary.csv",f"tab:tax-{split}","footnotesize")
    diagnostic=taxpc[((taxpc.model=="A")&(taxpc.split=="val")&taxpc.level.isin(["family","order"]))|((taxpc.model=="R")&(taxpc.split=="val")&taxpc.level.isin(["class","phylum"]))]
    files[S6/"taxonomy_diagnostics_table.tex"]=table(["Modelo","Nivel","Taxón","Obs.","F1"],
       [[r.model,LEVELS[r.level],r.taxon,integer(r.support),number(r.f1)] for r in diagnostic.itertuples()],"l l Y r r",
       "F1 por taxón que explica las variaciones del promedio al cambiar de nivel (validación).",
       "reports/s6/comparison/taxonomic_per_class.csv","tab:tax-diagnostics","footnotesize")
    conf=pd.read_csv(S6/"error_hierarchy.csv")
    files[S6/"error_hierarchy_table.tex"]=table(["Modelo / partición","Errores de género","Misma familia","Mismo orden"],
       [[f"{r.model} / {r.split}",r.errors,r.same_family_errors,r.same_order_errors] for r in conf.itertuples()],"Y r r r",
       "Errores de género que se convierten en aciertos al evaluar familia u orden. Las columnas no se suman: misma familia está incluida en mismo orden.",
       "reports/s6/comparison/error_hierarchy.csv","tab:error-hierarchy")
    pairs=pd.read_csv(S6/"confusion_pairs_top5.csv")
    files[S6/"confusion_pairs_table.tex"]=table(["Modelo / partición","Género real","Género predicho","Conteo"],
       [[f"{r.model} / {r.split}",f"\\textit{{{r.true_genus}}}",f"\\textit{{{r.predicted_genus}}}",r.count] for r in pairs.itertuples()],"l Y Y r",
       "Las cinco confusiones dirigidas más frecuentes de cada evaluación; los empates se ordenan por identificador de clase.",
       "reports/s6/comparison/confusion_pairs_top5.csv","tab:confusions","footnotesize")
    sp=pd.read_csv(S6/"species_difficult.csv")
    rows=[]
    for r in sp.to_dict("records"):
        cells=[f"\\textit{{{r['species']}}}",integer(r['obs_val']),integer(r['obs_test'])]
        for m,s in [(m,"val") for m in MODELS]+[("R","test")]:
            x=r[f"accuracy_{m}_{s}"]
            cells.append("--" if pd.isna(x) else number(x*100,1)+(r"$^{*}$" if r[f"obs_{s}"]<5 else ""))
        rows.append(cells)
    files[S6/"difficult_species_table.tex"]=table(["Especie","Obs. val","Obs. test","R val","A val","B val","C val","D val","R test previo"],
       rows,"Y r r r r r r r r","Acierto de género por especie (\\%). Se incluyen todas las especies con al menos cinco observaciones y menos de 90\\,\\% en al menos una evaluación disponible. Un asterisco indica soporte menor de cinco en esa celda; -- indica ausencia de observaciones.",
       "reports/s6/comparison/species_difficult.csv","tab:species","footnotesize")
    profile=json.loads((ROOT/"reports/s5/architecture_profile.json").read_text())
    files[S5/"profile_table.tex"]=table(["Arquitectura","Parámetros","Pesos (MB)","Mediana (ms/foto)"],
       [[r["architecture"].replace("resnet18","ResNet-18").replace("mobilenet_v3_large","MobileNetV3-Large"),integer(r["parameters"]),number(r["serialized_state_mb"]),number(r["latency_median_ms"])] for r in profile["models"]],
       "Y r r r","Perfil histórico en CPU del PC, FP32, lote unitario y entrada sintética. No incluye preparación de la foto ni ejecución en Android.",
       "reports/s5/architecture_profile.json","tab:profile")
    files[S5/"provenance_table.tex"]=table(["Clave","Carpeta de la corrida usada"],
       [[m,r"\path{"+str(results.loc[m,"run"])+"}"] for m in MODELS],"l Y","Correspondencia entre las claves del informe y las corridas completas.",
       "reports/s5/comparison/validation_results.csv","tab:provenance","footnotesize")
    inv=pd.read_csv(S5/"run_inventory.csv")
    bad=inv[inv.complete & inv.valid_probabilities.eq(False)]
    files[S5/"excluded_table.tex"]=table(["Ensayo","Corrida excluida"],
       [[r.trial.upper(),r"\path{"+r.run+"}"] for r in bad.itertuples()],"l Y","Corridas completas excluidas: cada una contiene probabilidades no finitas en 6\\,272 de 6\\,290 fotos de validación.",
       "reports/s5/comparison/run_inventory.csv","tab:excluded","footnotesize")
    for path,text in tqdm(files.items(),desc="Tablas LaTeX",unit="tabla"):
        path.write_text(text,encoding="utf-8")
    lines=["# Auditoría y reproducción del informe de S5 y S6", "",
       "## Fase 0: hallazgos antes de la reescritura", "",
       "Se inspeccionaron las corridas, sus reanudaciones y los informes históricos. El árbol y los archivos por carpeta se guardan en `artifact_tree.json` y `run_inventory.csv` (reports/s5/comparison/).", "",
       "| Ensayo | Tasa inicial | Lote train / val | Ponderación | Entorno efectivo | Fuente de los ajustes |",
       "|---|---:|---:|---|---|---|"]
    for m in "ABCD":
        c=configs[m]
        lines.append(f"| {m} | {number(c['training']['lr'],4)} | {c['training']['batch_size']} / {c['evaluation'].get('batch_size',c['training']['batch_size']*2)} | {c['training']['class_weighting']} | {'CPU FP32' if m=='A' else 'GPU AMP'} | ({results.loc[m,'run']}/run.json) |")
    lines += ["", "El lote de A se reconstruyó de la versión histórica del código cuyo SHA-256 con CRLF coincide con run.json; se conserva la comprobación (reports/s5/comparison/historical_code_audit.json).", "",
       "El resto de la receta se conserva entre ensayos: arquitectura, entrada, aumentos, optimizador, scheduler, suavizado, semilla y ausencia de capas congeladas; la tabla LaTeX y el CSV de configuración contienen cada valor y su fuente (reports/s5/comparison/configuration.csv).", "",
       "Las corridas válidas comparten huella v002, orden de géneros y todos los identificadores de validación. Se verificaron los bytes de metadatos contra el manifiesto y se reprodujeron las matrices de confusión a partir de los CSV (reports/s5/comparison/analysis_manifest.json).", "",
       "Las predicciones por observación no existían como archivos separados: se reconstruyeron promediando probabilidades por observación, antes de argmax; se entregan cinco CSV derivados, uno por modelo (scripts/train/metrics.py; reports/s5/comparison/predictions_observation_*_val.csv).", "",
       "Ningún MobileNet tiene test guardado; ResNet-18 tiene una evaluación histórica y su exposición se declara. No se ejecutó inferencia nueva sobre test (reports/s5/comparison/run_inventory.csv; reports/s5/comparison/analysis_manifest.json).", "",
       "### Corridas usadas y resultados de validación", "",
       "| Modelo | F1 observación [IC 95 %] | F1 foto | Acierto observación, % | Mejor / total | Horas de épocas + val | Fuente |",
       "|---|---|---:|---:|---:|---:|---|"]
    for m,r in results.iterrows():
        lines.append(f"| {NAMES[m]} | {interval(r.f1_observation,r.ci95_low,r.ci95_high)} | {number(r.f1_image)} | {number(r.accuracy_observation*100,1)} | {int(r.best_epoch)} / {int(r.epochs)} | {number(r.hours,2)} | ({r['run']}/metrics_val.json; {r['run']}/history.csv) |")
    lines += ["", "Las horas no incluyen preflight, espera entre sesiones ni evaluación final, y no suman otra vez las historias de origen al reanudar. D usa su carpeta final de continuación (reports/s5/comparison/validation_results.csv; reports/s5/comparison/run_inventory.csv).", "",
       "### Corridas excluidas", ""]
    for r in bad.itertuples():lines.append(f"- {r.run}: {r.invalid_reason} (reports/s5/comparison/run_inventory.csv).")
    lines += ["", "Se excluyen por probabilidades no finitas, aunque exista completed.json. No se interpretan sus métricas como rendimiento del modelo. Las carpetas se conservaron (reports/s5/development_results.json).", "",
       "## Comparación y selección", "",
       "Bootstrap percentil ordinario por observación: 10 000 remuestreos, semilla 42, etiquetas fijas. Los modelos comparten los mismos índices en cada remuestreo; no hay ajuste por comparaciones múltiples (reports/s5/comparison/analysis_manifest.json).", "",
       "| Comparación | Diferencia F1 | IC 95 % pareado | Incluye cero |", "|---|---:|---|---|"]
    for r in paired.itertuples():lines.append(f"| {r.model} menos {r.reference} | {number(r.delta_f1,4)} | [{number(r.ci95_low,4)}; {number(r.ci95_high,4)}] | {'Sí' if r.contains_zero else 'No'} |")
    lines += ["", "Fuente de todas las diferencias: (reports/s5/comparison/bootstrap_paired.csv).", "",
       f"Se selecciona {winner} por la mayor estimación de F1 macro por observación. Su diferencia con A no se distingue del ruido; no se afirma superioridad concluyente. Los MobileNet tienen idéntico coste de arquitectura y no existe benchmark Android que los desempatase. La selección operativa conserva el criterio principal y queda congelada con hashes (reports/s5/selected_model.json; reports/s5/comparison/profile_ratios.json).", "",
       "## Cambios realizados", "",
       "- Reescritura del LaTeX con A/B/C/D, fuentes locales y lista de pendientes (reports/week_05_06_results.tex).",
       "- Curvas sin etiquetas flotantes, con mejores vueltas en la leyenda; diferencias pareadas, métricas por género y tablas de soporte (reports/s5/comparison/).",
       "- Taxonomía del ganador y ResNet-18 con F1 y exactitud e IC; confusiones dirigidas y todas las especies difíciles según el criterio (reports/s6/comparison/).",
       "- Se distingue la exposición histórica a test y se entrega el comando final sin ejecutarlo (scripts/train/evaluate_s5_final.py).",
       "- El lector y selector rechazan probabilidades no finitas; el evaluador final respeta el lote configurado. No se altera la data ni las particiones (scripts/train/pipeline/predictions.py; scripts/train/summarize_s5.py; scripts/train/evaluate_s5_final.py).", "",
       "## Cifras que cambiaron y por qué", "",
       "- ResNet-18 val pasa de 0,954 a 0,955 en las tablas a tres decimales: el valor exacto sigue siendo 0,9545427706 (runs/20261004T133123181686Z_ma_resnet18_v002/metrics_val.json).",
       "- Los IC de ResNet val y A val se recalcularon con el método común; se reportan los nuevos extremos en la tabla de resultados. La amplitud mayor de A se relaciona con su error en Morchella, un género de soporte muy pequeño (reports/s5/comparison/bootstrap_genus_uncertainty.csv).",
       "- La razón de velocidad histórica es 1,7928058711, redondeada a 1,793, y la reducción de tamaño 61,9228247881 %, redondeada a 61,923 %. No son medidas Android (reports/s5/comparison/profile_ratios.json).",
       "- Se reemplaza A como candidato por C siguiendo la estimación principal, con empate no resuelto estadísticamente frente a A; se añade B/C/D y se retiran sus corridas corruptas de la selección (reports/s5/development_results.json; reports/s5/comparison/bootstrap_paired.csv).",
       "- La lista de especies usa el criterio en al menos una evaluación disponible, y añade los soportes de val y test; ya no conserva únicamente las tres especies del PDF original (reports/s6/comparison/species_difficult.csv).", "",
       "## Supuestos y límites", "",
       "- Se conserva la taxonomía de FungiTastic, incluida Polyporaceae para Ganoderma; cambiar esa clasificación cambiaría los aciertos familiares (data/taxonomy/v002/genus_taxonomy.csv).",
       "- La observación es la unidad de remuestreo. Los IC son condicionales a los modelos y no cubren entrenamiento ni dependencia de sitio-día; los CSV redondeados reproducen las matrices guardadas (reports/s5/comparison/analysis_manifest.json).",
       "- A usa CPU FP32 y B/C/D CUDA AMP. Una sola semilla por receta impide atribuir causalmente toda diferencia A-C a la ponderación (reports/s5/comparison/configuration.csv).",
       "- La explicación por aspecto visual se presenta como hipótesis; los conteos de fotos no demuestran esa causa (reports/s6/comparison/species_difficult.csv).",
       "- El test de ResNet-18 ya fue expuesto: el futuro test de C será una nueva evaluación de otro modelo sobre una partición conocida históricamente, no una partición ciega durante todo el proyecto (runs/20261004T133123181686Z_ma_resnet18_v002/completed.json).", "",
       "## Alcance y trabajo futuro", "",
       "- Test de C: métricas, predicciones, IC, taxonomía, confusiones y especies (no existe final_test/completed.json en la corrida seleccionada).",
       "- Teléfono Android concreto, runtime/formato final, latencia, memoria y tamaño del despliegue (reports/s5/architecture_profile.json).",
       "- Réplicas independientes con otras semillas (reports/s5/comparison/validation_results.csv).",
       "- Verificación de clonación limpia en Windows con finales de línea reproducibles; el checkout actual sí coincide con el manifiesto (reports/s5/comparison/git_windows_audit.json).",
       "- Modelo exacto de CPU de las corridas A y ResNet (sus run.json registran CPU, sin identificar el procesador).",
       "- Transcripciones completas de terminal de B/C/D y tiempo de pared con preparación/evaluación final; history.csv solo contiene el tiempo de épocas más val (reports/s5/comparison/run_inventory.csv).",
       "- Descriptores o revisión morfológica para contrastar el aspecto diferente de las especies difíciles (reports/s6/comparison/species_difficult.csv).", "",
       "## Reproducción", "", "Desde D:\\Dev\\Virtus\\edge-ai-virtus, en PowerShell:", "", "```powershell",
       ".venv\\Scripts\\python.exe -u scripts\\reports\\analyze_s5_s6.py --resamples 10000 --seed 42",
       ".venv\\Scripts\\python.exe scripts\\reports\\build_s5_s6_tables.py", "```", "",
       "El primer comando reanuda automáticamente el checkpoint del bootstrap. Si cambian las predicciones o el método, exige --restart. Las tablas se escriben por separado y son regenerables (scripts/reports/analyze_s5_s6.py; scripts/reports/build_s5_s6_tables.py).", "",
       "Compilación del PDF, con Tectonic instalado (se usó el ejecutable auxiliar local tmp/tectonic/tectonic.exe):", "", "```powershell",
       "tectonic --keep-logs --outdir reports reports/week_05_06_results.tex", "```", "",
       "La selección ya está congelada. El siguiente comando se entrega para ejecutar deliberadamente el test pendiente una vez; NO se ejecutó durante esta revisión:", "", "```powershell",
       ".venv\\Scripts\\python.exe -u scripts\\train\\evaluate_s5_final.py", "```", "",
       "El marcador de campaña bloquea repeticiones automáticas. No se debe borrarlo para repetir una evaluación (scripts/train/evaluate_s5_final.py).", "",
       "Comprobaciones numéricas y de reanudación: registro en (reports/s5/comparison/validation_checks.txt). Los CSV y JSON de comparison son la evidencia de esta revisión; los exports anteriores de reports/s6 describen el estado previo y no sustituyen las nuevas tablas."]
    (S5/"AUDITORIA_Y_REPRODUCCION.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(f"Tablas guardadas: {len(files)}. Cada archivo es un punto de salida independiente.")


if __name__=="__main__":main()
