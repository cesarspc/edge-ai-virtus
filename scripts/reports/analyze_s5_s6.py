"""Rebuild the S5/S6 report evidence from saved artifacts, without model inference.

Usage: python scripts/reports/analyze_s5_s6.py --resamples 10000 --seed 42
Bootstrap progress is checkpointed every 250 resamples and resumes automatically.
The official test photographs are never opened. Only the existing ResNet test CSV is read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/train"))
from metrics import aggregate_observations, classification_metrics
from pipeline.predictions import load_predictions
from pipeline.taxonomy import LEVELS, level_index

OUT = ROOT / "reports/s5/comparison"
TAXOUT = ROOT / "reports/s6/comparison"
RUNS = {
    "R": "20261004T133123181686Z_ma_resnet18_v002",
    "A": "20261012T161041257797Z_s5_mobilenet_v3_large_v002_a",
    "B": "20261024T022849507889Z_s5_mobilenet_v3_large_v002_b",
    "C": "20261024T031650621510Z_s5_mobilenet_v3_large_v002_c",
    "D": "20261024T132723373718Z_s5_mobilenet_v3_large_v002_d",
}
NAMES = {"R": "ResNet-18", **{k: f"MobileNet {k}" for k in "ABCD"}}
LEVEL_ES = dict(zip(LEVELS, ["Género", "Familia", "Orden", "Clase", "Filo"]))
COLORS = dict(zip(RUNS, ["#657584", "#1976b8", "#d3840b", "#168060", "#a24c9c"]))


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def relative(path):
    return str(path.relative_to(ROOT)).replace("\\", "/")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv(path, rows):
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(path, index=False, lineterminator="\n")
    return frame


def metric_counts(y, pred, n):
    cm = np.bincount(y * n + pred, minlength=n*n).reshape(n, n)
    return cm


def bootstrap(jobs, n_samples, seed, path, source_signature, restart=False):
    """One set of observation indices for every job in the same split: paired by construction."""
    keys = list(jobs)
    n = len(jobs[keys[0]][0])
    assert all(len(j[0]) == n for j in jobs.values())
    signature = hashlib.sha256(json.dumps([keys, n_samples, seed, source_signature], sort_keys=True).encode()).hexdigest()
    scores = np.zeros((n_samples, len(keys)))
    accuracies = np.zeros_like(scores)
    genus_keys = [k for k in keys if k.endswith("_genus")]
    per_genus = np.zeros((n_samples, len(genus_keys), 10))
    rng = np.random.default_rng(seed)
    done = 0
    if path.exists() and not restart:
        with np.load(path, allow_pickle=False) as cache:
            if str(cache["signature"]) != signature:
                raise ValueError(f"{path}: inputs changed; use --restart to replace the bootstrap checkpoint")
            done = int(cache["done"])
            scores[:] = cache["scores"]
            accuracies[:] = cache["accuracies"]
            per_genus[:] = cache["per_genus"]
            rng.bit_generator.state = json.loads(str(cache["rng_state"]))
    with tqdm(total=n_samples, initial=done, desc=f"Bootstrap {path.parent.name}: {n} obs.", unit="remuestreo") as progress:
        while done < n_samples:
            end = min(done + 250, n_samples)
            idx = rng.integers(0, n, size=(end-done, n))
            for col, key in enumerate(keys):
                y, pred, n_classes = jobs[key]
                codes = (y*n_classes + pred)[idx]
                offsets = np.arange(end-done)[:, None] * n_classes*n_classes
                cm = np.bincount((codes+offsets).ravel(), minlength=(end-done)*n_classes*n_classes).reshape(-1, n_classes, n_classes)
                tp = np.diagonal(cm, axis1=1, axis2=2)
                denom = cm.sum(axis=1) + cm.sum(axis=2)
                f1 = np.divide(2*tp, denom, out=np.zeros_like(tp, dtype=float), where=denom > 0)
                scores[done:end, col] = f1.mean(axis=1)
                accuracies[done:end, col] = tp.sum(axis=1) / n
                if key in genus_keys:
                    per_genus[done:end, genus_keys.index(key)] = f1
            done = end
            temporary = path.with_suffix(".npz.tmp")
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, signature=signature, done=done, scores=scores, accuracies=accuracies,
                                    per_genus=per_genus, rng_state=json.dumps(rng.bit_generator.state))
            temporary.replace(path)
            progress.update(len(idx))
    return {k: scores[:, i] for i, k in enumerate(keys)}, {k: accuracies[:, i] for i, k in enumerate(keys)}, {
        k: per_genus[:, i] for i, k in enumerate(genus_keys)}


def ci(values):
    return np.quantile(values, [.025, .975]).tolist()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--resamples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--restart", action="store_true")
    args = ap.parse_args()
    if args.resamples < 100:
        ap.error("at least 100 resamples are required")
    OUT.mkdir(parents=True, exist_ok=True)
    TAXOUT.mkdir(parents=True, exist_ok=True)
    inputs = {}

    def track(path):
        inputs[relative(path)] = digest(path)
        return path

    print("[1/6] Inventario e integridad de metadatos y predicciones", flush=True)
    manifest = read("data/versions/v002/manifest.json")
    classes = pd.read_csv(track(ROOT / "data/versions/v002/classes.csv")).sort_values("label_id").genus.tolist()
    for entry in manifest["outputs"]:
        name = entry["name"]
        if name in ["observations.csv", "images.csv", "classes.csv", "image_integrity.csv"]:
            base = "data/processed/v002" if name in ["observations.csv", "images.csv"] else "data/versions/v002"
            path = track(ROOT / base / name)
            if digest(path) != entry["sha256"]:
                raise ValueError(f"{path}: published hash differs (including line endings)")
    obs = pd.read_csv(ROOT / "data/processed/v002/observations.csv", low_memory=False)
    images = pd.read_csv(ROOT / "data/processed/v002/images.csv", low_memory=False)
    species = pd.read_csv(track(ROOT / "data/taxonomy/v002/observation_species.csv"))
    tax = pd.read_csv(track(ROOT / "data/taxonomy/v002/genus_taxonomy.csv")).sort_values("label_id")
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT)
    write_json(OUT/"git_windows_audit.json", {
        "core_autocrlf":git("config","--get","core.autocrlf").decode().strip(),
        "dataset_git_status":git("status","--short","--","data").decode().strip(),
        "published_hashes_match_bytes":True,
        "gitattributes":(ROOT/".gitattributes").read_text(),
        "fresh_checkout_verified":False})
    legacy=git("show","1031da4:scripts/train/train_ma.py")
    recorded=read(f"runs/{RUNS['A']}/run.json")["code_sha256"]["train_ma.py"]
    legacy_hashes={"LF":hashlib.sha256(legacy).hexdigest(),"CRLF":hashlib.sha256(legacy.replace(b"\n",b"\r\n")).hexdigest()}
    write_json(OUT/"historical_code_audit.json",{"git_revision":"1031da4","path":"scripts/train/train_ma.py",
        "recorded_A_code_sha256":recorded,"candidate_sha256":legacy_hashes,"matches_run_record":recorded in legacy_hashes.values(),
        "evaluation_batch_rule":[line.strip() for line in legacy.decode().splitlines() if 'batch_size=tr["batch_size"] * 2' in line],
        "effective_A_evaluation_batch":read(f"runs/{RUNS['A']}/run.json")["config"]["training"]["batch_size"]*2})
    assert tax.genus.tolist() == classes and not tax[LEVELS].isna().any().any()
    assert set(obs.split) == {"train", "val", "test"}
    assert not obs.observation_uid.duplicated().any() and not images.image_uid.duplicated().any()
    joined = images.merge(obs[["observation_uid", "split", "label_id"]], on="observation_uid", suffixes=("", "_obs"), validate="many_to_one")
    assert joined.split.eq(joined.split_obs).all() and joined.label_id.eq(joined.label_id_obs).all()
    split_counts = obs.groupby("split").size().to_dict()
    image_counts = images.groupby("split").size().to_dict()
    inventory = []
    infos, histories, predictions = {}, {}, {}
    for folder in tqdm(sorted((ROOT/"runs").iterdir()), desc="Corridas", unit="carpeta"):
        if not folder.is_dir() or not (folder/"run.json").exists():
            continue
        info = read(relative(folder/"run.json"))
        row = {"run": relative(folder), "files": ";".join(p.name for p in sorted(folder.iterdir()) if p.is_file()),
               "complete": (folder/"completed.json").exists(), "trial": info["config"].get("experiment", {}).get("trial", "R"),
               "resumed_from": info.get("resumed_from", ""), "selected_for_report": folder.name in RUNS.values()}
        if row["complete"]:
            done = read(relative(folder/"completed.json"))
            row["test_evaluations"] = done["test_evaluations"]
            try:
                frame, probs = load_predictions(folder/"predictions_val.csv", classes)
                row.update(valid_probabilities=True, invalid_reason="", validation_photos=len(frame))
            except ValueError as exc:
                row.update(valid_probabilities=False, invalid_reason=str(exc).replace(str(ROOT)+"\\", ""))
        inventory.append(row)
    csv(OUT/"run_inventory.csv", inventory)
    write_json(OUT/"artifact_tree.json", {name: [relative(p) for p in sorted((ROOT/name).rglob("*")) if p.is_file() and "comparison" not in p.parts] for name in ["runs", "reports"]})

    rows, configurations, per_class = [], [], []
    print("[2/6] Reproducción de métricas, checkpoints y observaciones", flush=True)
    for model, run_id in RUNS.items():
        run = ROOT/"runs"/run_id
        info = read(relative(track(run/"run.json")))
        assert info["dataset"]["content_fingerprint"] == manifest["content_fingerprint"]
        assert info["dataset"]["version"] == "v002" and info["dataset"]["split_column"] == "split"
        assert info["dataset"]["images"]["train"] == image_counts["train"]
        assert info["dataset"]["images"]["val"] == image_counts["val"]
        infos[model] = info
        history = pd.read_csv(track(run/"history.csv"))
        histories[model] = history
        assert history.epoch.tolist() == list(range(1, len(history)+1))
        if info.get("resumed_from"):
            ancestor=ROOT/"runs"/Path(info["resumed_from"].replace("\\","/")).parent.name/"history.csv"
            ancestor_history=pd.read_csv(track(ancestor))
            pd.testing.assert_frame_equal(history.head(len(ancestor_history)).reset_index(drop=True),ancestor_history.reset_index(drop=True),check_exact=True)
        saved = read(relative(track(run/"metrics_val.json")))
        frame, probs = load_predictions(track(run/"predictions_val.csv"), classes)
        expected = images.loc[images.split.eq("val"), ["image_uid", "observation_uid", "label_id"]].sort_values("image_uid").reset_index(drop=True)
        pd.testing.assert_frame_equal(frame[expected.columns].sort_values("image_uid").reset_index(drop=True), expected, check_dtype=False)
        assert (frame.pred_label_id.to_numpy() == probs.argmax(1)).all()
        yo, po = aggregate_observations(frame.image_uid, frame.observation_uid, frame.label_id.to_numpy(), probs)
        ids = sorted(frame.observation_uid.unique())
        observation = pd.DataFrame({"observation_uid":ids, "label_id":yo.astype(int), "pred_label_id":po.argmax(1)})
        for i, genus in enumerate(classes):
            observation[f"p_{genus}"] = po[:, i]
        observation = observation.merge(species[["observation_uid", "species"]], on="observation_uid", validate="one_to_one")
        assert set(ids) == set(obs.loc[obs.split.eq("val"), "observation_uid"])
        predictions[model] = observation
        csv(OUT/f"predictions_observation_{model}_val.csv", observation)
        for unit, y, p in [("image_level",frame.label_id.to_numpy(),probs), ("observation_level",yo,po)]:
            calculated = classification_metrics(y,p,classes,[1,3])
            assert calculated["confusion_matrix"] == saved[unit]["confusion_matrix"], (model,unit,"confusion changed")
            assert abs(calculated["macro_f1"]-saved[unit]["macro_f1"]) < 1e-12
        metrics = saved["observation_level"]
        for item in metrics["per_class"]:
            per_class.append({"model":model, **item, "source":relative(run/"metrics_val.json")})
        best = history.loc[history.val_obs_macro_f1.idxmax()]
        tail = history.tail(info["config"]["training"]["early_stopping_patience"])
        rows.append({"model":model,"run":relative(run),"f1_observation":metrics["macro_f1"],
                     "f1_image":saved["image_level"]["macro_f1"],"accuracy_observation":metrics["accuracy"],
                     "best_epoch":int(best.epoch),"epochs":len(history),"stop":"early stopping",
                     "epoch_seconds_total":float(history.seconds.sum()),"hours":float(history.seconds.sum()/3600),
                     "last5_f1_min":float(tail.val_obs_macro_f1.min()),"last5_f1_max":float(tail.val_obs_macro_f1.max()),
                     "device":info["device"],"gpu":info.get("gpu") or "[FALTA DATO: modelo CPU]",
                     "amp_effective":info["amp"],"seed":info["config"]["training"]["seed"],
                     "metric_source":relative(run/"metrics_val.json"),"history_source":relative(run/"history.csv")})
        cfg = info["config"]
        def flatten(value, prefix=""):
            for key, val in value.items():
                path = f"{prefix}.{key}" if prefix else key
                if isinstance(val,dict): yield from flatten(val,path)
                else: yield path,val
        for key,val in flatten(cfg):
            configurations.append({"model":model,"setting":key,"value":json.dumps(val, ensure_ascii=False),"source":relative(run/"run.json")})
    # Selection is computed from validation before reading the historical test predictions.
    winner = max("ABCD", key=lambda k: next(r["f1_observation"] for r in rows if r["model"] == k))
    write_json(OUT/"validation_recommendation.json", {"winner_point_estimate":winner,"run":f"runs/{RUNS[winner]}",
               "test_used_for_selection":False,"policy":"highest observation macro-F1; no significance claimed until paired bootstrap"})
    csv(OUT/"configuration.csv", configurations)
    csv(OUT/"validation_per_genus.csv", per_class)
    print("Leyendo checkpoints en CPU para verificar metadatos y estado finito...", flush=True)
    import torch
    checkpoints = []
    for model, run_id in tqdm(RUNS.items(), desc="Checkpoints", unit="modelo"):
        run = ROOT/"runs"/run_id
        best = torch.load(track(run/"best.pt"), map_location="cpu", weights_only=True)
        last = torch.load(track(run/"last.pt"), map_location="cpu", weights_only=True)
        assert best["classes"] == last["classes"] == classes
        assert best["config"] == last["config"] == infos[model]["config"]
        assert best["dataset_fingerprint"] == last["dataset_fingerprint"] == manifest["content_fingerprint"]
        assert last["training_finished"] and last["bad_epochs"] >= infos[model]["config"]["training"]["early_stopping_patience"]
        assert int(best["epoch"]) == next(r["best_epoch"] for r in rows if r["model"]==model)
        bad = [key for key,v in best["model"].items() if torch.is_floating_point(v) and not torch.isfinite(v).all()]
        assert not bad
        checkpoints.append({"model":model,"best_epoch":best["epoch"],"last_epoch":last["epoch"],"bad_epochs":last["bad_epochs"],
                            "training_finished":last["training_finished"],"nonfinite_tensors":bad,"classes":best["classes"],
                            "best_source":relative(run/"best.pt"),"last_source":relative(run/"last.pt")})
        del best,last
    write_json(OUT/"checkpoint_audit.json",checkpoints)

    distributions, ratios = [], []
    for split in ["train","val","test"]:
        counts_o = obs[obs.split.eq(split)].groupby("genus").size().reindex(classes)
        counts_i = images[images.split.eq(split)].groupby("genus").size().reindex(classes)
        for genus in classes:
            distributions.append({"split":split,"genus":genus,"observations":int(counts_o[genus]),"photos":int(counts_i[genus]),
                                  "observation_pct":float(counts_o[genus]/counts_o.sum()*100),"photo_pct":float(counts_i[genus]/counts_i.sum()*100)})
        ratios.append({"split":split,"observations":int(counts_o.sum()),"photos":int(counts_i.sum()),
                       "observation_max_min":float(counts_o.max()/counts_o.min()),"photo_max_min":float(counts_i.max()/counts_i.min()),
                       "observation_largest":counts_o.idxmax(),"observation_smallest":counts_o.idxmin()})
    csv(OUT/"class_distribution.csv",distributions)
    csv(OUT/"imbalance_ratios.csv",ratios)

    print("[3/6] Bootstrap pareado y métricas taxonómicas de validación",flush=True)
    jobs = {}
    for model, frame in predictions.items():
        jobs[f"{model}_val_genus"] = (frame.label_id.to_numpy(),frame.pred_label_id.to_numpy(),len(classes))
    for model in ["R","A",winner]:
        for level in LEVELS[1:]:
            names,g2t = level_index(tax,level)
            frame = predictions[model]
            jobs[f"{model}_val_{level}"] = (g2t[frame.label_id.to_numpy()],g2t[frame.pred_label_id.to_numpy()],len(names))
    boot_signature={k:v for k,v in inputs.items() if k.endswith("predictions_val.csv") or k.endswith("genus_taxonomy.csv")}
    scores,accs,genus_boot = bootstrap(jobs,args.resamples,args.seed,OUT/"bootstrap_state.npz",boot_signature,args.restart)
    for row in rows:
        key=f"{row['model']}_val_genus"
        row["ci95_low"],row["ci95_high"] = ci(scores[key])
    validation=csv(OUT/"validation_results.csv",rows)
    paired=[]
    for model,ref in [(m,r) for m in "ABCD" for r in ["A","R"] if m!=r]+[(winner,"B"),(winner,"D"),("D","B")]:
        delta=scores[f"{model}_val_genus"]-scores[f"{ref}_val_genus"]
        low,high=ci(delta)
        point=float(validation.set_index("model").loc[model,"f1_observation"]-validation.set_index("model").loc[ref,"f1_observation"])
        paired.append({"model":model,"reference":ref,"delta_f1":point,"ci95_low":low,"ci95_high":high,
                       "contains_zero":bool(low<=0<=high),"resamples":args.resamples,"seed":args.seed,"unit":"observation","n":len(predictions[model])})
    csv(OUT/"bootstrap_paired.csv",paired)
    uncertainty=[]
    for key,values in genus_boot.items():
        model=key.split("_")[0]
        for i,genus in enumerate(classes):
            low,high=ci(values[:,i])
            uncertainty.append({"model":model,"genus":genus,"bootstrap_f1_sd":float(values[:,i].std(ddof=1)),"ci95_low":low,"ci95_high":high})
    csv(OUT/"bootstrap_genus_uncertainty.csv",uncertainty)

    print("[4/6] Reutilizando únicamente el test histórico de ResNet-18",flush=True)
    historical=ROOT/"runs"/RUNS["R"]
    frame,probs=load_predictions(track(historical/"predictions_test.csv"),classes)
    yo,po=aggregate_observations(frame.image_uid,frame.observation_uid,frame.label_id.to_numpy(),probs)
    test_obs=pd.DataFrame({"observation_uid":sorted(frame.observation_uid.unique()),"label_id":yo.astype(int),"pred_label_id":po.argmax(1)})
    test_obs=test_obs.merge(species[["observation_uid","species"]],on="observation_uid",validate="one_to_one")
    assert set(test_obs.observation_uid)==set(obs.loc[obs.split.eq("test"),"observation_uid"])
    saved_test=read(relative(track(historical/"metrics_test.json")))
    computed_test=classification_metrics(yo,po,classes,[1,3])
    assert computed_test["confusion_matrix"]==saved_test["observation_level"]["confusion_matrix"]
    csv(TAXOUT/"predictions_observation_R_test.csv",test_obs)
    test_jobs={}
    for level in LEVELS:
        names,g2t=level_index(tax,level)
        test_jobs[f"R_test_{level}"]=(g2t[test_obs.label_id.to_numpy()],g2t[test_obs.pred_label_id.to_numpy()],len(names))
    test_scores,test_accs,_=bootstrap(test_jobs,args.resamples,args.seed,TAXOUT/"bootstrap_state.npz",{relative(historical/"predictions_test.csv"):digest(historical/"predictions_test.csv")},args.restart)
    all_jobs={**jobs,**test_jobs};all_scores={**scores,**test_scores};all_accs={**accs,**test_accs}
    taxonomy_rows,taxclass=[],[]
    for key,(y,pred,n) in all_jobs.items():
        model,split,level=key.split("_")
        if model not in ["R","A",winner]:continue
        names,_=level_index(tax,level)
        m=classification_metrics(y,np.eye(n)[pred],names,[],name_key="taxon")
        low,high=ci(all_scores[key]);alo,ahi=ci(all_accs[key])
        taxonomy_rows.append({"model":model,"split":split,"level":level,"n":len(y),"n_taxa":n,"macro_f1":m["macro_f1"],
                              "ci95_low":low,"ci95_high":high,"accuracy":m["accuracy"],"accuracy_ci95_low":alo,"accuracy_ci95_high":ahi})
        for item in m["per_class"]:taxclass.append({"model":model,"split":split,"level":level,**item})
    csv(TAXOUT/"taxonomic_summary.csv",taxonomy_rows)
    csv(TAXOUT/"taxonomic_per_class.csv",taxclass)
    confusions,confusion_pairs=[],[]
    for model,split,frame in [(m,"val",predictions[m]) for m in ["R",winner]]+[("R","test",test_obs)]:
        y,pred=frame.label_id.to_numpy(),frame.pred_label_id.to_numpy()
        wrong=y!=pred
        _,g2f=level_index(tax,"family");_,g2o=level_index(tax,"order")
        confusions.append({"model":model,"split":split,"errors":int(wrong.sum()),"same_family_errors":int((wrong&(g2f[y]==g2f[pred])).sum()),
                           "same_order_errors":int((wrong&(g2o[y]==g2o[pred])).sum()),"n":len(y)})
        cm=metric_counts(y,pred,len(classes));pairs=[(int(cm[i,j]),i,j) for i in range(len(classes)) for j in range(len(classes)) if i!=j and cm[i,j]>0]
        for count,i,j in sorted(pairs,key=lambda v:(-v[0],v[1],v[2]))[:5]:
            confusion_pairs.append({"model":model,"split":split,"true_genus":classes[i],"predicted_genus":classes[j],"count":count})
    csv(TAXOUT/"error_hierarchy.csv",confusions)
    csv(TAXOUT/"confusion_pairs_top5.csv",confusion_pairs)
    breakdown=[]
    for model,split,frame in [(m,"val",f) for m,f in predictions.items()]+[("R","test",test_obs)]:
        for sp,g in frame.groupby("species",sort=True):
            breakdown.append({"model":model,"split":split,"species":sp,"observations":len(g),"correct_genus":int(g.label_id.eq(g.pred_label_id).sum()),
                              "accuracy":float(g.label_id.eq(g.pred_label_id).mean())})
    breakdown=csv(TAXOUT/"species_breakdown.csv",breakdown)
    weak=sorted(set(breakdown.loc[(breakdown.observations>=5)&(breakdown.accuracy<.9),"species"]))
    annotated_obs=obs.merge(species[["observation_uid","species"]],on="observation_uid",validate="one_to_one")
    annotated_images=images.merge(species[["observation_uid","species"]],on="observation_uid",validate="many_to_one")
    difficult=[]
    for sp in weak:
        row={"species":sp}
        for split in ["train","val","test"]:
            row[f"obs_{split}"]=int(((annotated_obs.species==sp)&(annotated_obs.split==split)).sum())
            row[f"photos_{split}"]=int(((annotated_images.species==sp)&(annotated_images.split==split)).sum())
        for model,split in [(m,"val") for m in RUNS]+[("R","test")]:
            found=breakdown[(breakdown.species==sp)&(breakdown.model==model)&(breakdown.split==split)]
            row[f"accuracy_{model}_{split}"]=float(found.iloc[0].accuracy) if len(found) else None
        difficult.append(row)
    csv(TAXOUT/"species_difficult.csv",difficult)
    profile=read(relative(track(ROOT/"reports/s5/architecture_profile.json")))
    r,m=profile["models"]
    write_json(OUT/"profile_ratios.json",{"latency_ratio_resnet_over_mobile":r["latency_median_ms"]/m["latency_median_ms"],
              "serialized_size_reduction_pct":(1-m["serialized_state_mb"]/r["serialized_state_mb"])*100,
              "profile_reused":True,"reason":"all four trials retain mobilenet_v3_large and a ten-class head","source":"reports/s5/architecture_profile.json"})

    print("[5/6] Figuras",flush=True)
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig,ax=plt.subplots(figsize=(10,4.8),layout="constrained")
    for model,h in histories.items():
        b=h.loc[h.val_obs_macro_f1.idxmax()]
        ax.plot(h.epoch,h.val_obs_macro_f1,color=COLORS[model],lw=1.6,alpha=.9,label=f"{NAMES[model]} (mejor vuelta {int(b.epoch)}; {b.val_obs_macro_f1:.3f})".replace(".",","))
        ax.scatter([b.epoch],[b.val_obs_macro_f1],s=90,marker="*",color=COLORS[model],edgecolors="white",linewidths=.5,zorder=5)
    ax.set(xlabel="Vuelta de entrenamiento",ylabel="F1 macro por observación (validación)")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_:f"{v:.2f}".replace(".",",")))
    ax.legend(loc="lower right",frameon=True,fontsize=9);ax.grid(axis="y",alpha=.22)
    fig.savefig(OUT/"validation_curves.png",dpi=220);fig.savefig(OUT/"validation_curves.pdf");plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.4),layout="constrained")
    for i,row in enumerate(paired):
        ax.errorbar(row["delta_f1"],i,xerr=[[row["delta_f1"]-row["ci95_low"]],[row["ci95_high"]-row["delta_f1"]]],fmt="o",color=COLORS[row["model"]],capsize=3)
    ax.set_yticks(range(len(paired)),[f"{NAMES[x['model']]} menos {NAMES[x['reference']]}" for x in paired])
    ax.axvline(0,color="#777",ls="--",lw=1);ax.set_xlabel("Diferencia de F1 macro por observación; IC 95 % pareado")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v,_:f"{v:.3f}".replace(".",",")))
    ax.grid(axis="x",alpha=.22);fig.savefig(OUT/"paired_differences.png",dpi=220);plt.close(fig)

    print("[6/6] Manifiesto y auditoría",flush=True)
    write_json(OUT/"analysis_manifest.json",{"dataset_fingerprint":manifest["content_fingerprint"],"classes":classes,
               "bootstrap":{"resamples":args.resamples,"seed":args.seed,"unit":"observation","method":"ordinary paired percentile bootstrap; fixed label universe; quantiles .025 and .975; no multiple-comparison correction"},
               "winner_point_estimate":winner,"source_sha256":inputs,"script_sha256":digest(Path(__file__)),
               "split_observations":split_counts,"split_images":image_counts,"test_inference_executed":False,
               "known_limitations":["one seed per recipe","A CPU FP32 versus B/C/D CUDA AMP","ResNet official test was historically exposed","CPU model for training not recorded","bootstrap ignores site-day dependence and training-seed variance","no Android benchmark"]})
    print(validation[["model","f1_observation","ci95_low","ci95_high","best_epoch","epochs"]].to_string(index=False))
    print(pd.DataFrame(paired).to_string(index=False))
    print(f"Outputs: {OUT} and {TAXOUT}; winner by validation point estimate: {winner}",flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
