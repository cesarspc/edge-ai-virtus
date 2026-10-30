"""S8 val-only visual, environmental, and fused experiments, resumable at epoch boundaries."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from progress_ui import bar, ready_batches, stage

with stage("Cargando PyTorch y bibliotecas", active=__name__ == "__main__"):
    import numpy as np
    import pandas as pd
    import torch
    import yaml
    from torch.utils.data import DataLoader
    from data import REPO_ROOT, load_tables, resolve, sha256_file, summary
    from pipeline.artifacts import ProgressFile, code_hashes, environment, git_commit, results_markdown, write_json
    from pipeline.evaluation import evaluate, save_predictions
    from pipeline.models import build_model
    from pipeline.multimodal import (EnvironmentalEncoder, EnvironmentalOnlyModel, MultimodalTable,
                                     VisualEnvironmentalModel, VisualOnlyModel)
    from pipeline.training import (atomic_checkpoint, build_criterion, build_optimizer, build_scheduler,
                                   class_weights, random_state, restore_random_state, set_seed)
    from pipeline.transforms import build_transforms


FIELDS = ["observation_uid", "split", "label_id", "habitat_class", "substrate_class", "elevation_m",
          "elevation__quality", "event_date", "event_date__quality"]
ENV_FIELDS = FIELDS[3:]


def join_observations(frames, observations):
    if observations.observation_uid.duplicated().any():
        raise ValueError("duplicate observation_uid in observations")
    result = {}
    for split, images in frames.items():
        joined = images.merge(observations, on="observation_uid", how="left", validate="many_to_one",
                              suffixes=("", "_obs"), sort=False)
        if len(joined) != len(images) or joined["split"].isna().any():
            raise ValueError("missing or duplicated observation for an image")
        if not joined["split"].eq(split).all() or not joined.label_id.eq(joined.label_id_obs).all():
            raise ValueError("observation split or label disagrees with image")
        result[split] = joined
    return result


def condition_frame(frame, condition, seed):
    """Perturb metadata at observation level, keeping repeated images together."""
    result = frame.copy()
    if condition == "missing":
        for column in ("habitat_class", "substrate_class", "event_date"):
            result[column] = pd.Series(pd.NA, index=result.index, dtype="string")
        result["elevation_m"] = np.nan
        result["elevation__quality"] = "MISSING"
        result["event_date__quality"] = "MISSING"
    elif condition == "shuffled":
        obs = result.drop_duplicates("observation_uid").sort_values("observation_uid")
        shuffled = obs[ENV_FIELDS].iloc[np.random.default_rng(seed).permutation(len(obs))].reset_index(drop=True)
        shuffled["observation_uid"] = obs.observation_uid.to_numpy()
        result = result.drop(columns=ENV_FIELDS).merge(shuffled, on="observation_uid", how="left",
                                                       validate="many_to_one", sort=False)
    elif condition != "normal":
        raise ValueError(condition)
    return result


def predict(model, loader, device, amp, description):
    model.eval()
    probabilities = []
    with torch.no_grad():
        for images, metadata, _ in bar(ready_batches(loader, f"Preparando primer lote: {description}"),
                                        total=len(loader), desc=description, unit="lote"):
            with torch.autocast(device_type=device.type, enabled=amp):
                logits = model(images.to(device), metadata.to(device))
            if not torch.isfinite(logits).all():
                raise FloatingPointError("non-finite validation logits")
            probabilities.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
    return np.concatenate(probabilities)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", required=True, choices=["visual", "metadata", "multimodal"])
    parser.add_argument("--resume", help="last.pt of an unfinished S8 run")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--pause-after-epoch", action="store_true")
    args = parser.parse_args()
    cfg_path = resolve(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["mode"] = args.mode
    if cfg["evaluation"].get("evaluate_test_at_end") or cfg["data"].get("split_column", "split") != "split":
        raise SystemExit("S8 uses official train/val only; test is forbidden in this script")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA no está disponible. Verifica PyTorch o usa --device cpu explícitamente.")
    tr = cfg["training"]
    set_seed(tr["seed"], tr["deterministic"])
    amp = tr["amp"] and device.type == "cuda"
    print("Preparando train y val; verificando dataset e imágenes...", flush=True)
    tables = load_tables(cfg, show_progress=True, splits=("train", "val"))
    print(summary(tables).to_string(index=False), flush=True)
    if len(tables.missing) or any(f.empty for f in tables.frames.values()):
        raise SystemExit("Faltan imágenes de train o val; completa la descarga antes de entrenar.")
    if len(tables.classes) != cfg["model"]["num_classes"]:
        raise SystemExit("El número de clases no coincide con el dataset")
    observations = pd.read_csv(REPO_ROOT / "data" / "processed" / tables.version / "observations.csv",
                               usecols=FIELDS, low_memory=False)
    observations = observations[observations.split.isin(["train", "val"])]
    frames = join_observations(tables.frames, observations)
    train_obs = frames["train"].drop_duplicates("observation_uid")
    encoder = EnvironmentalEncoder.fit(train_obs)
    encoded = {s: encoder.transform(frame) for s, frame in frames.items()}
    feature_count = encoded["train"].shape[1]
    init_path = resolve(cfg["initialization"]["checkpoint"])
    if not init_path.is_file():
        raise SystemExit(f"Checkpoint visual no encontrado: {init_path}")
    hashes = code_hashes()
    for name in ("train_s8.py", "pipeline/multimodal.py"):
        hashes[name] = sha256_file(Path(__file__).parent / name)
    init_hash = sha256_file(init_path)
    resumed = torch.load(resolve(args.resume), map_location="cpu", weights_only=True) if args.resume else None
    if resumed is not None:
        expected = (cfg, tables.fingerprint, tables.classes, device.type, hashes, init_hash, encoder.state_dict())
        actual = (resumed.get("config"), resumed.get("dataset_fingerprint"), resumed.get("classes"),
                  resumed.get("device_type"), resumed.get("code_sha256"), resumed.get("init_sha256"),
                  resumed.get("encoder"))
        if resumed.get("format") != "s8-v1" or expected != actual:
            raise SystemExit("No se puede reanudar: cambió configuración, datos, código, dispositivo o checkpoint inicial.")
        run_dir = resolve(args.resume).parent
        if (run_dir / "completed.json").exists():
            raise SystemExit("Esta corrida ya finalizó.")
    else:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_s8_" + args.mode
        run_dir = resolve(cfg["output"]["runs_dir"]) / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        write_json(run_dir / "run.json", {"run_id": run_id, "config": cfg, "dataset_fingerprint": tables.fingerprint,
                   "classes": tables.classes, "encoder": encoder.state_dict(), "initial_checkpoint": str(init_path),
                   "init_sha256": init_hash, "code_sha256": hashes, "git": git_commit(REPO_ROOT),
                   "device": str(device), **environment(device)})
    print(f"Carpeta de la corrida: {run_dir}", flush=True)
    progress = ProgressFile(run_dir, run_dir.name)
    progress("initializing")
    tf_train, tf_eval = build_transforms(cfg)
    generator = torch.Generator().manual_seed(tr["seed"])
    kw = dict(num_workers=tr["num_workers"], pin_memory=device.type == "cuda", persistent_workers=False)
    load_images = args.mode != "metadata"
    train_loader = DataLoader(MultimodalTable(frames["train"], encoded["train"], tf_train, load_images),
                              batch_size=tr["batch_size"], shuffle=True, generator=generator, **kw)
    def val_loader(condition):
        frame = condition_frame(frames["val"], condition, tr["seed"])
        return DataLoader(MultimodalTable(frame, encoder.transform(frame), tf_eval, load_images),
                          batch_size=cfg["evaluation"]["batch_size"], shuffle=False, **kw)
    val = val_loader("normal")
    if args.mode == "metadata":
        model = EnvironmentalOnlyModel(feature_count, len(tables.classes))
    else:
        checkpoint = torch.load(init_path, map_location="cpu", weights_only=True)
        if (checkpoint.get("classes") != tables.classes or checkpoint.get("dataset_fingerprint") != tables.fingerprint
                or checkpoint.get("config", {}).get("model", {}).get("architecture") != cfg["model"]["architecture"]):
            raise SystemExit("El checkpoint visual no coincide con clases, dataset o arquitectura")
        visual = build_model(cfg, pretrained=False)
        visual.load_state_dict(checkpoint["model"], strict=True)
        model = (VisualOnlyModel(visual) if args.mode == "visual" else
                 VisualEnvironmentalModel(visual, feature_count, len(tables.classes)))
    model.to(device)
    weights = class_weights(frames["train"], len(tables.classes), tr["class_weighting"])
    criterion = build_criterion(tr, weights, device)
    opt = build_optimizer(model, tr)
    sched = build_scheduler(opt, tr, len(train_loader))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    best, bad, history, first, finished, best_checkpoint = -1.0, 0, [], 1, False, None
    if resumed:
        model.load_state_dict(resumed["model"])
        opt.load_state_dict(resumed["optimizer"])
        sched.load_state_dict(resumed["scheduler"])
        scaler.load_state_dict(resumed["scaler"])
        best, bad, history = resumed["best_score"], resumed["bad_epochs"], resumed["history"]
        first, finished, best_checkpoint = resumed["epoch"] + 1, resumed["training_finished"], resumed["best_checkpoint"]
        restore_random_state(resumed["rng"], generator)
        atomic_checkpoint(best_checkpoint, run_dir / "best.pt")
        print(f"Reanudando después de la época {first - 1}", flush=True)
    for epoch in range(first, tr["epochs"] + 1) if not finished else ():
        start = time.time()
        model.train()
        losses = []
        progress("training", epoch=epoch, batches_total=len(train_loader))
        batches = bar(ready_batches(train_loader, f"Preparando época {epoch}"), total=len(train_loader),
                      desc=f"Epoca {epoch}/{tr['epochs']} - train", unit="lote")
        for step, (images, metadata, labels) in enumerate(batches, 1):
            images, metadata, labels = images.to(device), metadata.to(device), labels.to(device)
            with torch.autocast(device_type=device.type, enabled=amp):
                loss = criterion(model(images, metadata), labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite training loss")
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.item())
            batches.set_postfix(loss=f"{np.mean(losses):.4f}")
            if step % 10 == 0 or step == len(train_loader):
                progress("training", epoch=epoch, batches_completed=step, batches_total=len(train_loader))
        progress("validation", epoch=epoch)
        probabilities = predict(model, val, device, amp, f"Epoca {epoch} - val")
        metrics = evaluate(frames["val"], probabilities, tables.classes, cfg["evaluation"]["top_k"])
        score = metrics["observation_level"]["macro_f1"]
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_obs_macro_f1": score,
               "lr": opt.param_groups[0]["lr"], "seconds": round(time.time() - start, 1)}
        history.append(row)
        print(json.dumps(row), flush=True)
        if score > best:
            best, bad = score, 0
            best_checkpoint = {"format": "s8-best-v1", "model": model.state_dict(), "epoch": epoch,
                               "score": score, "encoder": encoder.state_dict(), "config": cfg,
                               "classes": tables.classes, "dataset_fingerprint": tables.fingerprint}
            atomic_checkpoint(best_checkpoint, run_dir / "best.pt")
            # Reload so later updates cannot mutate the in-memory best state.
            best_checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
        else:
            bad += 1
        finished = bad >= tr["early_stopping_patience"] or epoch >= tr["epochs"]
        state = {"format": "s8-v1", "epoch": epoch, "model": model.state_dict(), "optimizer": opt.state_dict(),
                 "scheduler": sched.state_dict(), "scaler": scaler.state_dict(), "rng": random_state(generator),
                 "history": history, "best_score": best, "bad_epochs": bad, "training_finished": finished,
                 "best_checkpoint": best_checkpoint, "encoder": encoder.state_dict(), "config": cfg,
                 "classes": tables.classes, "dataset_fingerprint": tables.fingerprint, "device_type": device.type,
                 "code_sha256": hashes, "init_sha256": init_hash}
        atomic_checkpoint(state, run_dir / "last.pt")
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False, lineterminator="\n")
        print(f"Checkpoint guardado: {run_dir / 'last.pt'}", flush=True)
        if args.pause_after_epoch or (run_dir / "pause.request").exists():
            (run_dir / "pause.request").unlink(missing_ok=True)
            progress("paused", epoch=epoch)
            print("Pausado al finalizar la época. Reanuda con --resume y last.pt.", flush=True)
            return 0
        if finished:
            if bad >= tr["early_stopping_patience"]:
                print(f"Early stopping en la época {epoch}", flush=True)
            break
    model.load_state_dict(best_checkpoint["model"])
    results = {}
    for condition in (["normal"] if args.mode == "visual" else ["normal", "missing", "shuffled"]):
        progress("evaluation", split="val", condition=condition)
        probabilities = predict(model, val if condition == "normal" else val_loader(condition), device, amp,
                                f"Evaluacion val - {condition}")
        metrics = evaluate(frames["val"], probabilities, tables.classes, cfg["evaluation"]["top_k"])
        suffix = "val" if condition == "normal" else f"val_{condition}"
        write_json(run_dir / f"metrics_{suffix}.json", metrics)
        save_predictions(run_dir / f"predictions_{suffix}.csv", frames["val"], probabilities, tables.classes)
        results[suffix] = metrics
        print(f"{suffix}: F1 macro observacion {metrics['observation_level']['macro_f1']:.4f}", flush=True)
    (run_dir / "RESULTS.md").write_text(results_markdown(run_dir.name, tables.version, tables.fingerprint, device,
                                                         "train", results, False), encoding="utf-8")
    write_json(run_dir / "completed.json", {"evaluated_splits": ["val"], "test_evaluations": 0,
                                              "best_epoch": best_checkpoint["epoch"]})
    progress("completed", evaluated_splits=["val"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
