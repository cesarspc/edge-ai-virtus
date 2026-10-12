"""Train and evaluate M_A = f(I) (image-only genus classifier) on a processed dataset version.

Usage:
    python scripts/train/train_ma.py --config configs/train/ma_resnet18_v001.yaml
    python scripts/train/train_ma.py --config configs/train/ma_resnet18_v001.yaml --smoke     # 2 min pipeline test
    python scripts/train/train_ma.py --config ... --eval-only runs/<run_id>/best.pt           # re-evaluate a checkpoint
    python scripts/train/train_ma.py --config ... --resume runs/<run_id>/last.pt

Each run writes runs/<run_id>/:
    run.json            config + SHA-256, dataset version/fingerprint, git commit, library versions, device, seed
    history.csv         per-epoch train loss and validation metrics
    best.pt             checkpoint with the best validation `selection_metric`
    metrics_val.json / metrics_test.json    image- and observation-level metrics (macro-F1, accuracy, balanced
                        accuracy, top-k, per class, confusion matrix)
    predictions_<split>.csv                 per-image probabilities (for later calibration / abstention studies)

Rules: model selection and early stopping use only `val`; `test` is evaluated once at the end with the best checkpoint.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from itertools import islice
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from progress_ui import bar, enabled as progress_enabled, stage

with stage("Cargando PyTorch y bibliotecas", active=__name__ == "__main__"):
    import numpy as np
    import pandas as pd
    import torch
    import torchvision
    import yaml
    from PIL import Image
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from data import REPO_ROOT, load_tables, resolve, summary  # noqa: E402
    from metrics import aggregate_observations, classification_metrics  # noqa: E402


# --------------------------------------------------------------------------- data
class ImageTable(Dataset):
    def __init__(self, frame: pd.DataFrame, transform):
        self.paths = frame["path"].tolist()
        self.labels = frame["label_id"].astype(int).tolist()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        with Image.open(self.paths[i]) as im:
            x = self.transform(im.convert("RGB"))
        return x, self.labels[i]


def build_transforms(cfg: dict[str, Any]):
    inp = cfg["input"]
    norm = transforms.Normalize(inp["mean"], inp["std"])
    aug = inp["train_augment"]
    train = [transforms.RandomResizedCrop(inp["size"], scale=tuple(aug["random_resized_crop_scale"]))]
    if aug.get("horizontal_flip"):
        train.append(transforms.RandomHorizontalFlip())
    if aug.get("color_jitter"):
        train.append(transforms.ColorJitter(*aug["color_jitter"]))
    train += [transforms.ToTensor(), norm]
    evaluation = [transforms.Resize(inp["eval_resize"]), transforms.CenterCrop(inp["size"]), transforms.ToTensor(), norm]
    return transforms.Compose(train), transforms.Compose(evaluation)


# --------------------------------------------------------------------------- model
def build_model(cfg: dict[str, Any], pretrained: bool = True) -> nn.Module:
    os.environ.setdefault("TORCH_HOME", str(REPO_ROOT / ".cache" / "torch"))
    m = cfg["model"]
    ctor = getattr(torchvision.models, m["architecture"])
    weights = m.get("pretrained_weights") if pretrained else None
    model = ctor(weights=weights) if weights else ctor(weights=None)
    n = m["num_classes"]
    # replace the classification head (supports the families listed in the project scope)
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):                       # ResNet
        model.fc = nn.Linear(model.fc.in_features, n)
    elif hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):  # MobileNet / EfficientNet
        last = [i for i, l in enumerate(model.classifier) if isinstance(l, nn.Linear)][-1]
        model.classifier[last] = nn.Linear(model.classifier[last].in_features, n)
    else:
        raise SystemExit(f"do not know how to replace the head of {m['architecture']}")
    return model


# --------------------------------------------------------------------------- helpers
def set_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False


def git_commit() -> dict[str, Any]:
    def run(*a):
        try:
            return subprocess.run(["git", *a], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()
        except Exception:  # noqa: BLE001
            return ""
    return {"commit": run("rev-parse", "HEAD") or None, "dirty": bool(run("status", "--porcelain"))}


def class_weights(train: pd.DataFrame, n: int, mode: str) -> torch.Tensor | None:
    if mode == "none":
        return None
    counts = np.bincount(train["label_id"].astype(int), minlength=n).astype(float)
    if mode == "inverse_sqrt":
        w = 1.0 / np.sqrt(np.maximum(counts, 1))
        return torch.tensor(w / w.mean(), dtype=torch.float32)
    raise SystemExit(f"unknown class_weighting {mode}")


def atomic_checkpoint(state: dict, path: Path) -> None:
    temporary = path.with_suffix(".pt.tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def random_state(generator: torch.Generator) -> dict:
    numpy_state = np.random.get_state()
    return {"python": random.getstate(),
            "numpy": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
            "torch": torch.get_rng_state(), "loader": generator.get_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_random_state(state: dict, generator: torch.Generator) -> None:
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), *n[2:]))
    torch.set_rng_state(state["torch"].cpu())
    generator.set_state(state["loader"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device, amp: bool,
            description: str = "Evaluacion") -> np.ndarray:
    model.eval()
    out = []
    for x, _ in bar(loader, desc=description, unit="lote"):
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(x.to(device, non_blocking=True))
        out.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 0))


def evaluate(frame: pd.DataFrame, probs: np.ndarray, classes: list[str], top_k: list[int]) -> dict[str, Any]:
    y = frame["label_id"].to_numpy()
    img = classification_metrics(y, probs, classes, top_k)
    yo, po = aggregate_observations(frame["image_uid"], frame["observation_uid"], y, probs)
    return {"image_level": img, "observation_level": classification_metrics(yo, po, classes, top_k)}


def save_predictions(path: Path, frame: pd.DataFrame, probs: np.ndarray, classes: list[str]) -> None:
    df = frame[["image_uid", "observation_uid", "label_id"]].copy()
    df["pred_label_id"] = probs.argmax(axis=1)
    for i, c in enumerate(classes):
        df[f"p_{c}"] = probs[:, i].round(6)
    df.to_csv(path, index=False, lineterminator="\n")


# --------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true", help="tiny run (few batches, 1 epoch) to test the pipeline")
    ap.add_argument("--eval-only", default=None, help="checkpoint to evaluate on val only (no training or test access)")
    ap.add_argument("--device", default=None, help="cuda | cpu | mps (default: best available)")
    ap.add_argument("--resume", help="last.pt from an interrupted run; continues at the next epoch")
    ap.add_argument("--pause-after-epoch", action="store_true", help="save last.pt after the next epoch and exit without test")
    args = ap.parse_args()
    if args.smoke and args.eval_only:
        ap.error("--smoke and --eval-only cannot be combined")
    if args.resume and (args.smoke or args.eval_only):
        ap.error("--resume cannot be combined with --smoke or --eval-only")
    if args.pause_after_epoch and (args.smoke or args.eval_only):
        ap.error("--pause-after-epoch requires full training")

    cfg_path = resolve(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
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
        if resumed.get("resume_format") != 1:
            raise SystemExit("Resume requires last.pt with optimizer state; legacy best.pt is not resumable.")
        if resumed["classes"] != t.classes or resumed["dataset_fingerprint"] != t.fingerprint:
            raise SystemExit("Resume labels/dataset differ from this dataset.")
        if resumed["config"] != cfg or resumed["device_type"] != device.type:
            raise SystemExit("Resume requires the same config and device type as the saved run.")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + cfg["run_name"] + ("_smoke" if args.smoke else "")
    run_dir = resolve(cfg["output"]["runs_dir"]) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    run_info = {
        "run_id": run_id, "mode": "smoke" if args.smoke else ("eval_only" if args.eval_only else "train"),
        "config_path": str(cfg_path.relative_to(REPO_ROOT)) if cfg_path.is_relative_to(REPO_ROOT) else str(cfg_path),
        "config_sha256": hashlib.sha256(cfg_path.read_bytes()).hexdigest(), "config": cfg,
        "dataset": {"version": t.version, "content_fingerprint": t.fingerprint, "split_column": cfg["data"]["split_column"],
                    "images": {s: len(f) for s, f in frames.items()}, "images_missing": len(missing)},
        "git": git_commit(), "device": str(device), "amp": amp,
        "code_sha256": {p: hashlib.sha256((Path(__file__).parent / p).read_bytes()).hexdigest()
                        for p in ("train_ma.py", "data.py", "metrics.py", "progress_ui.py")},
        "versions": {"python": platform.python_version(), "torch": torch.__version__, "torchvision": torchvision.__version__,
                     "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None},
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    (run_dir / "run.json").write_text(json.dumps(run_info, indent=2, ensure_ascii=False), encoding="utf-8")
    if resumed is not None:
        run_info["resumed_from"] = str(resume_path)
        run_info["resumed_after_epoch"] = resumed["epoch"]
        (run_dir / "run.json").write_text(json.dumps(run_info, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"run directory: {run_dir}")
    def progress(stage: str, **values) -> None:
        state = {"run_id": run_id, "stage": stage, "updated_utc": datetime.now(timezone.utc).isoformat(), **values}
        target = run_dir / "progress.json"
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
        temporary.replace(target)
    progress("initializing")

    tf_train, tf_eval = build_transforms(cfg)
    g = torch.Generator().manual_seed(tr["seed"])
    # Recreate workers each epoch so their RNG can be reproduced after a restart.
    kw = dict(num_workers=tr["num_workers"], pin_memory=device.type == "cuda", persistent_workers=False)
    loaders = {}
    if not args.eval_only:
        loaders["train"] = DataLoader(ImageTable(frames["train"], tf_train), batch_size=tr["batch_size"], shuffle=True,
                                      drop_last=len(frames["train"]) > tr["batch_size"], generator=g, **kw)
    for s in (["val", "test"] if evaluate_test else ["val"]):
        loaders[s] = DataLoader(ImageTable(frames[s], tf_eval), batch_size=tr["batch_size"] * 2, shuffle=False, **kw)

    with stage(f"Cargando modelo {cfg['model']['architecture']} y sus pesos"):
        model = build_model(cfg, pretrained=not (args.eval_only or args.resume)).to(device)
    top_k = cfg["evaluation"]["top_k"]
    best_path = run_dir / "best.pt"

    if args.eval_only:
        checkpoint = torch.load(resolve(args.eval_only), map_location=device)
        if checkpoint.get("classes") != t.classes or checkpoint.get("dataset_fingerprint", t.fingerprint) != t.fingerprint:
            raise SystemExit("checkpoint labels/dataset do not match this dataset version")
        model.load_state_dict(checkpoint["model"])
    else:
        w = class_weights(frames["train"], len(t.classes), tr["class_weighting"])
        balance = {"unit": "image", "computed_from": "train only", "mode": tr["class_weighting"],
                   "classes": t.classes, "train_counts": np.bincount(frames["train"].label_id.astype(int), minlength=len(t.classes)).tolist(),
                   "loss_weights": w.tolist() if w is not None else [1.0] * len(t.classes)}
        (run_dir / "class_balance.json").write_text(json.dumps(balance, indent=2), encoding="utf-8")
        criterion = nn.CrossEntropyLoss(weight=w.to(device) if w is not None else None, label_smoothing=tr["label_smoothing"])
        if tr["optimizer"] != "adamw":
            raise SystemExit("only adamw is implemented")
        opt = torch.optim.AdamW(model.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"])
        steps_per_epoch = max(1, len(loaders["train"]))
        if args.smoke:
            steps_per_epoch = min(steps_per_epoch, 3)
        total, warm = tr["epochs"] * steps_per_epoch, tr["warmup_epochs"] * steps_per_epoch
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / max(1, warm) if s < warm
                                                  else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
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
            restore_random_state(resumed["rng"], g)
            print(f"Resuming after epoch {resumed['epoch']}; next epoch {first_epoch}", flush=True)
        for epoch in range(first_epoch, tr["epochs"] + 1) if not training_finished else ():
            model.train()
            t0, losses = time.time(), []
            batches = bar(islice(loaders["train"], steps_per_epoch), total=steps_per_epoch,
                          desc=f"Epoca {epoch}/{tr['epochs']} - train", unit="lote")
            for step, (x, y) in enumerate(batches):
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, enabled=amp):
                    loss = criterion(model(x), y)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sched.step()
                losses.append(loss.item())
                batches.set_postfix(loss=f"{np.mean(losses):.4f}", lr=f"{opt.param_groups[0]['lr']:.2g}")
                if (step + 1) % 10 == 0 or step + 1 == steps_per_epoch:
                    progress("training", epoch=epoch, batches_completed=step + 1, batches_total=steps_per_epoch,
                             seconds=round(time.time() - t0, 1), train_loss=float(np.mean(losses)))
                    if not progress_enabled():
                        print(f"epoch {epoch}: batch {step + 1}/{steps_per_epoch}, loss {np.mean(losses):.4f}", flush=True)
            progress("validation", epoch=epoch)
            probs = predict(model, loaders["val"], device, amp, f"Epoca {epoch} - val")
            m = evaluate(frames["val"], probs, t.classes, top_k)
            row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "lr": opt.param_groups[0]["lr"],
                   "val_img_macro_f1": m["image_level"]["macro_f1"], "val_img_accuracy": m["image_level"]["accuracy"],
                   "val_obs_macro_f1": m["observation_level"]["macro_f1"],
                   "val_obs_balanced_accuracy": m["observation_level"]["balanced_accuracy"],
                   "seconds": round(time.time() - t0, 1)}
            history.append(row)
            pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False, lineterminator="\n")
            print(json.dumps(row))
            if row[metric] > best:
                best, bad_epochs = row[metric], 0
                atomic_checkpoint({"model": model.state_dict(), "epoch": epoch, metric: best, "classes": t.classes,
                                   "config": cfg, "dataset_fingerprint": t.fingerprint}, best_path)
            else:
                bad_epochs += 1
            training_finished = bad_epochs >= tr["early_stopping_patience"] or epoch >= tr["epochs"]
            atomic_checkpoint({"resume_format": 1, "epoch": epoch, "model": model.state_dict(),
                               "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                               "scaler": scaler.state_dict(), "rng": random_state(g), "history": history,
                               "best_score": best, "bad_epochs": bad_epochs, "training_finished": training_finished,
                               "best_checkpoint": torch.load(best_path, map_location="cpu", weights_only=True),
                               "classes": t.classes, "dataset_fingerprint": t.fingerprint,
                               "config": cfg, "device_type": device.type}, run_dir / "last.pt")
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

    splits = ["val"] + (["test"] if evaluate_test else [])
    results = {}
    for s in splits:
        if s == "test":
            (run_dir / "test_started.json").write_text(json.dumps({"started_utc": datetime.now(timezone.utc).isoformat()}), encoding="utf-8")
        progress("evaluation", split=s)
        probs = predict(model, loaders[s], device, amp, f"Evaluacion final - {s}")
        m = evaluate(frames[s], probs, t.classes, top_k)
        results[s] = m
        (run_dir / f"metrics_{s}.json").write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
        save_predictions(run_dir / f"predictions_{s}.csv", frames[s], probs, t.classes)
        print(f"{s}: obs macro-F1 {m['observation_level']['macro_f1']:.4f} | img macro-F1 {m['image_level']['macro_f1']:.4f} "
              f"| img acc {m['image_level']['accuracy']:.4f}")
    lines = [f"# Resultado {run_id}", "", f"Dataset: `{t.version}` · huella: `{t.fingerprint}`.",
             f"Dispositivo: `{device}` · modo: `{run_info['mode']}`.", "",
             "| Partición | Imágenes | Observaciones | Macro-F1 imagen | Macro-F1 observación | Accuracy observación |",
             "|---|---:|---:|---:|---:|---:|"]
    for s, m in results.items():
        a, b = m["image_level"], m["observation_level"]
        lines.append(f"| {s} | {a['n']} | {b['n']} | {a['macro_f1']:.4f} | {b['macro_f1']:.4f} | {b['accuracy']:.4f} |")
    lines += ["", "| Partición | Género | Observaciones | F1 |", "|---|---|---:|---:|"]
    for s, m in results.items():
        for c in m["observation_level"]["per_class"]:
            lines.append(f"| {s} | {c['genus']} | {c['support']} | {c['f1']:.4f} |")
    if args.smoke:
        lines += ["", "Prueba de ejecución: estas métricas no estiman rendimiento del modelo."]
    (run_dir / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (run_dir / "completed.json").write_text(json.dumps({"run_id": run_id, "evaluated_splits": splits,
                                                        "test_evaluations": int(evaluate_test)}, indent=2), encoding="utf-8")
    progress("completed", evaluated_splits=splits)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
