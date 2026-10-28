"""Evaluate the explicitly selected S5 checkpoint on test once; no training."""
import json
from datetime import datetime, timezone

import torch
from torch.utils.data import DataLoader

from data import REPO_ROOT, load_tables, resolve, sha256_file
from train_ma import ImageTable, build_model, build_transforms, predict, evaluate, save_predictions


def main():
    study=REPO_ROOT/"reports/s5"
    marker=study/"final_test_started.json"
    if marker.exists():
        raise SystemExit("S5 final test already started. Inspect its saved results; automatic repeats are blocked.")
    selection=json.loads((study/"selected_model.json").read_text(encoding="utf-8"))
    selected=selection["selected"]
    run=resolve(selected["run"])
    best=run/"best.pt"
    if sha256_file(best)!=selected["checkpoint_sha256"] or sha256_file(run/"run.json")!=selected["run_json_sha256"] or sha256_file(run/"metrics_val.json")!=selected["metrics_val_sha256"]:
        raise SystemExit("Selected checkpoint, config or val metrics changed after selection.")
    done=json.loads((run/"completed.json").read_text(encoding="utf-8"))
    if done["test_evaluations"]!=0 or done["evaluated_splits"]!=["val"]:
        raise SystemExit("Final evaluation requires a completed val-only trial.")
    info=json.loads((run/"run.json").read_text(encoding="utf-8"))
    cfg=info["config"]
    if info["mode"]!="train" or cfg.get("experiment",{}).get("campaign")!="mobilenet_v3_large_v002":
        raise SystemExit("Selected run is not a full S5 development trial.")
    tables=load_tables(cfg,show_progress=True,splits=("test",))
    if len(tables.missing) or tables.fingerprint!=selected["dataset_fingerprint"]:
        raise SystemExit("Missing/modified test images or changed dataset.")
    checkpoint=torch.load(best,map_location="cpu",weights_only=True)
    if checkpoint["classes"]!=tables.classes or checkpoint["dataset_fingerprint"]!=tables.fingerprint or checkpoint["config"]!=cfg:
        raise SystemExit("Checkpoint does not match the selected configuration/dataset.")
    device=torch.device(info["device"])
    model=build_model(cfg,pretrained=False).to(device)
    model.load_state_dict(checkpoint["model"])
    _,tf=build_transforms(cfg)
    frame=tables.frames["test"]
    loader=DataLoader(ImageTable(frame,tf),batch_size=cfg["evaluation"].get("batch_size",cfg["training"]["batch_size"]),
                      shuffle=False,num_workers=cfg["training"]["num_workers"],pin_memory=device.type=="cuda")
    destination=run/"final_test"
    destination.mkdir(exist_ok=False)
    # Exclusive creation locks the whole S5 campaign, including concurrent launches.
    with marker.open("x",encoding="utf-8") as stream:
        json.dump({"started_utc":datetime.now(timezone.utc).isoformat(),"selected":selected,
                   "destination":str(destination.relative_to(REPO_ROOT))},stream,indent=2)
    probs=predict(model,loader,device,bool(info["amp"]),"S5 - test final seleccionado")
    result=evaluate(frame,probs,tables.classes,cfg["evaluation"]["top_k"])
    (destination/"metrics_test.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    save_predictions(destination/"predictions_test.csv",frame,probs,tables.classes)
    (destination/"selection.json").write_text(json.dumps(selection,indent=2),encoding="utf-8")
    (destination/"completed.json").write_text(json.dumps({"evaluated_splits":["test"],"test_evaluations":1,
                                "source_run":selected["run"],"checkpoint_sha256":sha256_file(best)},indent=2),encoding="utf-8")
    print(f"Final test saved: {destination}\nObservation macro-F1: {result['observation_level']['macro_f1']:.4f}")


if __name__=="__main__":
    main()
