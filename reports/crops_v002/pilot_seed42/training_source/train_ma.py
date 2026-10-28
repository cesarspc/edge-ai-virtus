"""Train and evaluate M_A = f(I) (image-only genus classifier) on a processed dataset version.

Usage:
    python scripts/train/train_ma.py --config configs/train/ma_resnet18_v001.yaml
    python scripts/train/train_ma.py --config configs/train/ma_resnet18_v001.yaml --smoke     # 2 min pipeline test
    python scripts/train/train_ma.py --config ... --eval-only runs/<run_id>/best.pt           # re-evaluate a checkpoint
    python scripts/train/train_ma.py --config ... --resume runs/<run_id>/last.pt

Each run writes runs/<run_id>/:
    run.json            config + SHA-256, dataset version/fingerprint, git commit, code hashes, library versions, device
    history.csv         per-epoch train loss and validation metrics
    best.pt             checkpoint with the best validation `selection_metric`
    metrics_val.json / metrics_test.json    image- and observation-level metrics (macro-F1, accuracy, balanced
                        accuracy, top-k, per class, confusion matrix)
    predictions_<split>.csv                 per-image probabilities (for later calibration / abstention studies)

Rules: model selection and early stopping use only `val`; `test` is evaluated once at the end with the best checkpoint.

This file only orchestrates the run (CLI, guards, run directory). The building blocks live in
``scripts/train/pipeline/`` (S6 modularization); see that package's docstring for the layer map.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from progress_ui import enabled as progress_enabled, stage

with stage("Cargando PyTorch y bibliotecas", active=__name__ == "__main__"):
    import pandas as pd
    import torch
    import yaml
    from torch.utils.data import DataLoader
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from data import REPO_ROOT, load_tables, resolve, sha256_file, summary  # noqa: E402
    from pipeline.artifacts import ProgressFile, code_hashes, environment, git_commit, results_markdown, write_json
    from pipeline.datasets import ImageTable
    from pipeline.drift import initialize_drift, measure_drift, optimizer_step_count
    from pipeline.evaluation import evaluate, predict, save_predictions
    from pipeline.models import build_model
    from pipeline.training import (atomic_checkpoint, best_state, build_criterion, build_optimizer, build_scheduler,
                                   check_resumable, class_balance_record, class_weights, restore_random_state,
                                   resume_state, set_seed, train_one_epoch)
    from pipeline.transforms import build_transforms

# Public names used by evaluate_s5_final.py, profile_s5.py and tests (stable import surface).
__all__ = ["ImageTable", "build_model", "build_transforms", "predict", "evaluate", "save_predictions",
           "class_weights", "set_seed", "main"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true", help="tiny run (few batches, 1 epoch) to test the pipeline")
    ap.add_argument("--eval-only", default=None, help="checkpoint to evaluate on val only (no training or test access)")
    ap.add_argument("--device", default=None, help="cuda | cpu | mps (default: best available)")
    ap.add_argument("--resume", help="last.pt from an interrupted run; continues at the next epoch")
    ap.add_argument("--init-checkpoint", help="start a NEW run from model weights; reset optimizer, scheduler and epochs")
    ap.add_argument("--pause-after-epoch", action="store_true", help="save last.pt after the next epoch and exit without test")
    args = ap.parse_args()
    if args.smoke and args.eval_only:
        ap.error("--smoke and --eval-only cannot be combined")
    if args.resume and (args.smoke or args.eval_only):
        ap.error("--resume cannot be combined with --smoke or --eval-only")
    if args.init_checkpoint and (args.resume or args.eval_only):
        ap.error("--init-checkpoint cannot be combined with --resume or --eval-only")
    if args.pause_after_epoch and (args.smoke or args.eval_only):
        ap.error("--pause-after-epoch requires full training")

    cfg_path = resolve(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if cfg.get("experiment", {}).get("phase") == "CROPS" and cfg["evaluation"]["evaluate_test_at_end"]:
        raise SystemExit("Crop development uses val only; test requires a separate final selection.")
    if cfg.get("experiment", {}).get("phase") == "S5":
        if cfg["evaluation"]["evaluate_test_at_end"]:
            raise SystemExit("S5 development is val-only; use evaluate_s5_final.py after explicit selection.")
        if (REPO_ROOT / "reports/s5/final_test_started.json").exists() and not args.eval_only:
            raise SystemExit("S5 final test already started; the tuning campaign is closed.")
    tr = cfg["training"]
    if args.smoke:
        tr.update(epochs=1, batch_size=8, num_workers=0, early_stopping_patience=1)
        cfg["data"]["require_all_images"] = False
    evaluate_test = bool(cfg["evaluation"]["evaluate_test_at_end"]) and not (args.smoke or args.eval_only)
    set_seed(tr["seed"], tr["deterministic"])
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available()
                                          else "mps" if torch.backends.mps.is_available() else "cpu"))
    amp = bool(tr["amp"]) and device.type == "cuda"

    # ------------------------------------------------------------------ data (version + integrity)
    if args.resume:
        resume_path = resolve(args.resume)
        if not resume_path.is_file():
            raise SystemExit(f"Resume checkpoint not found: {resume_path}")
        print(f"Checkpoint para reanudar: {resume_path}", flush=True)
    print("Preparando dataset y verificando integridad antes de entrenar...", flush=True)
    required_splits = ["val"] if args.eval_only else ["train", "val"] + (["test"] if evaluate_test else [])
    t = load_tables(cfg, show_progress=True, splits=tuple(required_splits))
    if cfg["model"]["num_classes"] != len(t.classes):
        raise SystemExit("model.num_classes does not match dataset classes")
    print(summary(t).to_string(index=False))
    missing = t.missing[t.missing["split"].isin(required_splits)]
    if cfg["data"]["require_all_images"] and len(missing):
        raise SystemExit(f"{len(missing)} images of {required_splits} are not available locally "
                         f"({missing['status'].replace('', 'no manifest').value_counts().to_dict()}). "
                         "Run `python scripts/data/download_images.py` or set data.require_all_images: false.")
    frames = {s: t.frames[s] for s in required_splits}
    if args.smoke:
        frames = {s: f.groupby("label_id", group_keys=False).head(4).reset_index(drop=True) for s, f in frames.items()}
    if any(len(f) == 0 for f in frames.values()):
        raise SystemExit("a split has no images available; download them first")

    resumed = None
    if args.resume:
        resume_path = resolve(args.resume)
        if (resume_path.parent / "completed.json").exists() or (resume_path.parent / "test_started.json").exists():
            raise SystemExit("This run already finished or started test evaluation; cannot resume it.")
        resumed = torch.load(resume_path, map_location="cpu", weights_only=True)
        check_resumable(resumed, t.classes, t.fingerprint, cfg, device.type)

    initialization = None
    init_info = None
    init_path = args.init_checkpoint or cfg.get("initialization", {}).get("checkpoint")
    if init_path and not (args.resume or args.eval_only):
        init_path = resolve(init_path)
        with stage("Cargando inicializacion para una corrida nueva"):
            initialization = torch.load(init_path, map_location="cpu", weights_only=True)
            if (initialization.get("classes") != t.classes or initialization.get("dataset_fingerprint") != t.fingerprint
                    or initialization.get("config", {}).get("model", {}).get("architecture") != cfg["model"]["architecture"]):
                raise SystemExit("Initial checkpoint must match architecture, dataset fingerprint and ordered classes.")
            init_info = {"checkpoint": str(init_path), "sha256": sha256_file(init_path),
                         "optimizer_reset": True, "scheduler_reset": True, "epoch_reset": True}
    track_drift = bool(tr.get("track_parameter_drift", False)) and not args.eval_only
    if track_drift and initialization is None and resumed is None:
        raise SystemExit("Parameter drift requires an explicit initial checkpoint or its saved resume state.")
    if track_drift and resumed is not None and "parameter_drift" not in resumed:
        raise SystemExit("This checkpoint has no parameter-drift reference to resume.")

    # ------------------------------------------------------------------ run directory + provenance
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + cfg["run_name"] + ("_smoke" if args.smoke else "")
    run_dir = resolve(cfg["output"]["runs_dir"]) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    run_info = {
        "run_id": run_id, "mode": "smoke" if args.smoke else ("eval_only" if args.eval_only else "train"),
        "config_path": str(cfg_path.relative_to(REPO_ROOT)) if cfg_path.is_relative_to(REPO_ROOT) else str(cfg_path),
        "config_sha256": hashlib.sha256(cfg_path.read_bytes()).hexdigest(), "config": cfg,
        "dataset": {"version": t.version, "content_fingerprint": t.fingerprint, "split_column": cfg["data"]["split_column"],
                    "images": {s: len(f) for s, f in frames.items()}, "images_missing": len(missing)},
        "git": git_commit(REPO_ROOT), "device": str(device), "amp": amp,
        "code_sha256": code_hashes(), **environment(device),
    }
    if resumed is not None:
        run_info["resumed_from"] = str(resume_path)
        run_info["resumed_after_epoch"] = resumed["epoch"]
    if init_info is not None:
        run_info["initialization"] = init_info
    elif resumed is not None and track_drift:
        run_info["initialization"] = resumed["parameter_drift"]["provenance"]
    write_json(run_dir / "run.json", run_info)
    print(f"run directory: {run_dir}")
    progress = ProgressFile(run_dir, run_id)
    progress("initializing")

    # ------------------------------------------------------------------ loaders + model
    tf_train, tf_eval = build_transforms(cfg)
    g = torch.Generator().manual_seed(tr["seed"])
    # Recreate workers each epoch so their RNG can be reproduced after a restart.
    kw = dict(num_workers=tr["num_workers"], pin_memory=device.type == "cuda", persistent_workers=False)
    loaders = {}
    if not args.eval_only:
        loaders["train"] = DataLoader(ImageTable(frames["train"], tf_train), batch_size=tr["batch_size"], shuffle=True,
                                      drop_last=len(frames["train"]) > tr["batch_size"], generator=g, **kw)
    eval_batch_size = int(cfg["evaluation"].get("batch_size", tr["batch_size"]))
    if eval_batch_size < 1:
        raise SystemExit("evaluation.batch_size must be a positive integer")
    for s in (["val", "test"] if evaluate_test else ["val"]):
        loaders[s] = DataLoader(ImageTable(frames[s], tf_eval), batch_size=eval_batch_size, shuffle=False, **kw)

    with stage(f"Cargando modelo {cfg['model']['architecture']} y sus pesos"):
        model = build_model(cfg, pretrained=not (args.eval_only or args.resume or initialization is not None)).to(device)
        if initialization is not None:
            model.load_state_dict(initialization["model"], strict=True)
    drift = None
    if track_drift:
        drift = resumed["parameter_drift"] if resumed is not None else initialize_drift(model, init_info)
        atomic_checkpoint({"model": drift["initial_state"], "classes": t.classes, "dataset_fingerprint": t.fingerprint,
                           "initial_state_sha256": drift["initial_state_sha256"], "provenance": drift["provenance"]}, run_dir / "initial.pt")
        write_json(run_dir / "initialization.json", {"state_sha256": drift["initial_state_sha256"], **drift["provenance"]})
    del initialization
    top_k = cfg["evaluation"]["top_k"]
    best_path = run_dir / "best.pt"

    # ------------------------------------------------------------------ training (val-only selection)
    if args.eval_only:
        checkpoint = torch.load(resolve(args.eval_only), map_location=device)
        if checkpoint.get("classes") != t.classes or checkpoint.get("dataset_fingerprint", t.fingerprint) != t.fingerprint:
            raise SystemExit("checkpoint labels/dataset do not match this dataset version")
        model.load_state_dict(checkpoint["model"])
    else:
        w = class_weights(frames["train"], len(t.classes), tr["class_weighting"])
        write_json(run_dir / "class_balance.json", class_balance_record(frames["train"], t.classes, tr["class_weighting"], w))
        criterion = build_criterion(tr, w, device)
        opt = build_optimizer(model, tr)
        steps_per_epoch = max(1, len(loaders["train"]))
        if args.smoke:
            steps_per_epoch = min(steps_per_epoch, 3)
        sched = build_scheduler(opt, tr, steps_per_epoch)
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        metric = tr["selection_metric"]
        best, bad_epochs, history = -1.0, 0, []
        first_epoch = 1
        training_finished = False
        if resumed is not None:
            model.load_state_dict(resumed["model"])
            opt.load_state_dict(resumed["optimizer"])
            sched.load_state_dict(resumed["scheduler"])
            scaler.load_state_dict(resumed["scaler"])
            best, bad_epochs, history = resumed["best_score"], resumed["bad_epochs"], resumed["history"]
            first_epoch, training_finished = resumed["epoch"] + 1, resumed["training_finished"]
            atomic_checkpoint(resumed["best_checkpoint"], best_path)
            atomic_checkpoint(resumed, run_dir / "last.pt")
            pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False, lineterminator="\n")
            if drift is not None:
                pd.DataFrame(drift["rows"]).to_csv(run_dir / "parameter_drift.csv", index=False)
                pd.DataFrame(drift["summaries"]).to_csv(run_dir / "parameter_drift_summary.csv", index=False)
            restore_random_state(resumed["rng"], g)
            print(f"Resuming after epoch {resumed['epoch']}; next epoch {first_epoch}", flush=True)
        for epoch in range(first_epoch, tr["epochs"] + 1) if not training_finished else ():
            t0 = time.time()

            def on_progress(done: int, mean_loss: float, epoch: int = epoch, t0: float = t0) -> None:
                progress("training", epoch=epoch, batches_completed=done, batches_total=steps_per_epoch,
                         seconds=round(time.time() - t0, 1), train_loss=mean_loss)
                if not progress_enabled():
                    print(f"epoch {epoch}: batch {done}/{steps_per_epoch}, loss {mean_loss:.4f}", flush=True)

            train_loss = train_one_epoch(model, loaders["train"], steps_per_epoch, criterion, opt, sched, scaler,
                                         device, amp, f"Epoca {epoch}/{tr['epochs']} - train", on_progress)
            progress("validation", epoch=epoch)
            probs = predict(model, loaders["val"], device, amp, f"Epoca {epoch} - val")
            m = evaluate(frames["val"], probs, t.classes, top_k)
            row = {"epoch": epoch, "train_loss": train_loss, "lr": opt.param_groups[0]["lr"],
                   "val_img_macro_f1": m["image_level"]["macro_f1"], "val_img_accuracy": m["image_level"]["accuracy"],
                   "val_obs_macro_f1": m["observation_level"]["macro_f1"],
                   "val_obs_balanced_accuracy": m["observation_level"]["balanced_accuracy"],
                   "seconds": round(time.time() - t0, 1)}
            history.append(row)
            pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False, lineterminator="\n")
            print(json.dumps(row))
            if row[metric] > best:
                best, bad_epochs = row[metric], 0
                atomic_checkpoint(best_state(model, epoch, metric, best, t.classes, cfg, t.fingerprint), best_path)
            else:
                bad_epochs += 1
            training_finished = bad_epochs >= tr["early_stopping_patience"] or epoch >= tr["epochs"]
            if drift is not None:
                measure_drift(model, drift, epoch, len(history)*steps_per_epoch, optimizer_step_count(opt))
                pd.DataFrame(drift["rows"]).to_csv(run_dir / "parameter_drift.csv", index=False)
                pd.DataFrame(drift["summaries"]).to_csv(run_dir / "parameter_drift_summary.csv", index=False)
            checkpoint_state = resume_state(epoch=epoch, model=model, opt=opt, sched=sched, scaler=scaler, generator=g,
                                           history=history, best_score=best, bad_epochs=bad_epochs,
                                           training_finished=training_finished, best_path=best_path,
                                           classes=t.classes, fingerprint=t.fingerprint, cfg=cfg,
                                           device_type=device.type)
            if drift is not None:
                checkpoint_state["parameter_drift"] = drift
            atomic_checkpoint(checkpoint_state, run_dir / "last.pt")
            del checkpoint_state
            print(f"Checkpoint saved: {run_dir / 'last.pt'}", flush=True)
            if args.pause_after_epoch or (run_dir / "pause.request").exists():
                progress("paused", epoch=epoch, checkpoint="last.pt")
                print("Paused at epoch boundary; resume with --resume last.pt. Test was not evaluated.", flush=True)
                return 0
            if training_finished:
                if bad_epochs >= tr["early_stopping_patience"]:
                    print(f"early stopping at epoch {epoch}")
                break
        model.load_state_dict(torch.load(best_path, map_location=device)["model"])

    # ------------------------------------------------------------------ final evaluation + export
    splits = ["val"] + (["test"] if evaluate_test else [])
    results = {}
    for s in splits:
        if s == "test":
            (run_dir / "test_started.json").write_text(json.dumps({"started_utc": datetime.now(timezone.utc).isoformat()}), encoding="utf-8")
        progress("evaluation", split=s)
        probs = predict(model, loaders[s], device, amp, f"Evaluacion final - {s}")
        m = evaluate(frames[s], probs, t.classes, top_k)
        results[s] = m
        write_json(run_dir / f"metrics_{s}.json", m)
        save_predictions(run_dir / f"predictions_{s}.csv", frames[s], probs, t.classes)
        print(f"{s}: obs macro-F1 {m['observation_level']['macro_f1']:.4f} | img macro-F1 {m['image_level']['macro_f1']:.4f} "
              f"| img acc {m['image_level']['accuracy']:.4f}")
    (run_dir / "RESULTS.md").write_text(results_markdown(run_id, t.version, t.fingerprint, device, run_info["mode"],
                                                         results, args.smoke), encoding="utf-8")
    (run_dir / "completed.json").write_text(json.dumps({"run_id": run_id, "evaluated_splits": splits,
                                                        "test_evaluations": int(evaluate_test)}, indent=2), encoding="utf-8")
    progress("completed", evaluated_splits=splits)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
