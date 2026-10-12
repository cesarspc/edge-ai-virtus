"""Rank completed S5 trials using saved VAL metrics only; optionally freeze a selection."""
import argparse
import json

from data import REPO_ROOT, resolve, sha256_file

CAMPAIGN="mobilenet_v3_large_v002"


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--select-run",help="explicitly close tuning and freeze this completed best-val run")
    args=ap.parse_args()
    out=REPO_ROOT/"reports/s5"
    out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for folder in (REPO_ROOT/"runs").iterdir():
        if not folder.is_dir() or not (folder/"completed.json").is_file():
            continue
        info=json.loads((folder/"run.json").read_text(encoding="utf-8"))
        experiment=info["config"].get("experiment",{})
        if experiment.get("campaign")!=CAMPAIGN or info["mode"]!="train":
            continue
        if experiment.get("trial") not in set("abcd") or info["dataset"]["version"]!="v002" or info["config"]["model"]["architecture"]!="mobilenet_v3_large":
            raise SystemExit(f"Unexpected trial definition: {folder.name}")
        done=json.loads((folder/"completed.json").read_text(encoding="utf-8"))
        if done["evaluated_splits"]!=["val"] or done["test_evaluations"]!=0:
            raise SystemExit(f"Development trial used test: {folder.name}")
        m=json.loads((folder/"metrics_val.json").read_text(encoding="utf-8"))
        rows.append({"run":str(folder.relative_to(REPO_ROOT)),"trial":experiment["trial"],
                     "macro_f1_observation":m["observation_level"]["macro_f1"],
                     "macro_f1_image":m["image_level"]["macro_f1"],
                     "class_weighting":info["config"]["training"]["class_weighting"],
                     "lr":info["config"]["training"]["lr"],
                     "dataset_fingerprint":info["dataset"]["content_fingerprint"],
                     "checkpoint_sha256":sha256_file(folder/"best.pt"),
                     "run_json_sha256":sha256_file(folder/"run.json"),
                     "metrics_val_sha256":sha256_file(folder/"metrics_val.json")})
    rows.sort(key=lambda r:(-r["macro_f1_observation"],r["run"]))
    if len({r["dataset_fingerprint"] for r in rows}) > 1:
        raise SystemExit("Trials use different dataset fingerprints and cannot be ranked together.")
    if len({r["trial"] for r in rows})!=len(rows):
        raise SystemExit("Multiple completed runs for the same trial: review duplicates before selecting.")
    payload={"campaign":CAMPAIGN,"selection_metric":"val observation macro-F1",
             "test_used_for_selection":False,"completed_trials":rows,
             "pending_trials":sorted(set("abcd")-{r["trial"] for r in rows})}
    (out/"development_results.json").write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
    lines=["# S5: corridas completas de desarrollo","", "Selección: macro-F1 por observación de VAL.","",
           "| Ensayo | LR | Pesos | F1 obs. val | F1 imagen val | Corrida |",
           "|---|---:|---|---:|---:|---|"]
    for r in rows:
        lines.append(f"| {r['trial']} | {r['lr']} | {r['class_weighting']} | {r['macro_f1_observation']:.4f} | {r['macro_f1_image']:.4f} | {r['run']} |")
    if not rows:
        lines+= ["", "Todavía no hay ensayos S5 completos. No se ha seleccionado ni entrenado el modelo móvil."]
    (out/"development_results.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    if args.select_run:
        if (out/"final_test_started.json").exists():
            raise SystemExit("The campaign already started final test evaluation; selection is locked.")
        selected=resolve(args.select_run).resolve()
        row=next((r for r in rows if resolve(r["run"]).resolve()==selected),None)
        if row is None or row!=rows[0]:
            raise SystemExit("Selection must be the highest-val completed eligible S5 trial.")
        payload["selected"]=row
        payload["selection_scope"]="completed trials only; pending trials were not compared"
        target=out/"selected_model.json"
        # Repeating the same selection is safe; different selections need explicit review.
        if target.exists() and json.loads(target.read_text(encoding="utf-8")).get("selected")!=row:
            raise SystemExit("A different model is already selected; review the selection before changing it.")
        target.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
        print(f"Frozen selection: {target}")
    print(json.dumps(payload,indent=2))


if __name__=="__main__":
    main()
