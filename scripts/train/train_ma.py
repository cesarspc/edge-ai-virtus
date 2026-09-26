"""Train and evaluate M_A = f(I) (image-only genus classifier) on a processed dataset version.

Usage:
    python scripts/train/train_ma.py --config configs/train/ma_resnet18_v001.yaml
    python scripts/train/train_ma.py --config configs/train/ma_resnet18_v001.yaml --smoke     # 2 min pipeline test
    python scripts/train/train_ma.py --config ... --eval-only runs/<run_id>/best.pt           # re-evaluate a checkpoint

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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
def build_model(cfg: dict[str, Any]) -> nn.Module:
    m = cfg["model"]
    ctor = getattr(torchvision.models, m["architecture"])
    weights = m.get("pretrained_weights")
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


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device, amp: bool) -> np.ndarray:
    model.eval()
    out = []
    for x, _ in loader:
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
    ap.add_argument("--eval-only", default=None, help="checkpoint to evaluate on val and test (no training)")
    ap.add_argument("--device", default=None, help="cuda | cpu | mps (default: best available)")
    args = ap.parse_args()

    cfg_path = resolve(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    tr = cfg["training"]
    if args.smoke:
        tr.update(epochs=1, batch_size=8, num_workers=0, early_stopping_patience=1)
        cfg["data"]["require_all_images"] = False
    set_seed(tr["seed"], tr["deterministic"])
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available()
                                          else "mps" if torch.backends.mps.is_available() else "cpu"))
    amp = bool(tr["amp"]) and device.type == "cuda"

    t = load_tables(cfg)
    print(summary(t).to_string(index=False))
    if cfg["data"]["require_all_images"] and len(t.missing):
        raise SystemExit(f"{len(t.missing)} images of train/val/test are not available locally "
                         f"({t.missing['status'].replace('', 'no manifest').value_counts().to_dict()}). "
                         "Run `python scripts/data/download_images.py` or set data.require_all_images: false.")
    frames = t.frames
    if args.smoke:
        frames = {s: f.groupby("label_id", group_keys=False).head(4).reset_index(drop=True) for s, f in frames.items()}
    if any(len(f) == 0 for f in frames.values()):
        raise SystemExit("a split has no images available; download them first")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + cfg["run_name"] + ("_smoke" if args.smoke else "")
    run_dir = resolve(cfg["output"]["runs_dir"]) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    run_info = {
        "run_id": run_id, "mode": "smoke" if args.smoke else ("eval_only" if args.eval_only else "train"),
        "config_path": str(cfg_path.relative_to(REPO_ROOT)) if cfg_path.is_relative_to(REPO_ROOT) else str(cfg_path),
        "config_sha256": hashlib.sha256(cfg_path.read_bytes()).hexdigest(), "config": cfg,
        "dataset": {"version": t.version, "content_fingerprint": t.fingerprint, "split_column": cfg["data"]["split_column"],
                    "images": {s: len(f) for s, f in frames.items()}, "images_missing": len(t.missing)},
        "git": git_commit(), "device": str(device), "amp": amp,
        "versions": {"python": platform.python_version(), "torch": torch.__version__, "torchvision": torchvision.__version__,
                     "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None},
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    (run_dir / "run.json").write_text(json.dumps(run_info, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"run directory: {run_dir}")

    tf_train, tf_eval = build_transforms(cfg)
    g = torch.Generator().manual_seed(tr["seed"])
    kw = dict(num_workers=tr["num_workers"], pin_memory=device.type == "cuda", persistent_workers=tr["num_workers"] > 0)
    loaders = {"train": DataLoader(ImageTable(frames["train"], tf_train), batch_size=tr["batch_size"], shuffle=True,
                                   drop_last=len(frames["train"]) > tr["batch_size"], generator=g, **kw)}
    for s in ("val", "test"):
        loaders[s] = DataLoader(ImageTable(frames[s], tf_eval), batch_size=tr["batch_size"] * 2, shuffle=False, **kw)

    model = build_model(cfg).to(device)
    top_k = cfg["evaluation"]["top_k"]
    best_path = run_dir / "best.pt"

    if args.eval_only:
        model.load_state_dict(torch.load(resolve(args.eval_only), map_location=device)["model"])
    else:
        w = class_weights(frames["train"], len(t.classes), tr["class_weighting"])
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
        for epoch in range(1, tr["epochs"] + 1):
            model.train()
            t0, losses = time.time(), []
            for step, (x, y) in enumerate(loaders["train"]):
                if step >= steps_per_epoch:
                    break
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, enabled=amp):
                    loss = criterion(model(x), y)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sched.step()
                losses.append(loss.item())
            probs = predict(model, loaders["val"], device, amp)
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
                torch.save({"model": model.state_dict(), "epoch": epoch, metric: best, "classes": t.classes,
                            "config": cfg}, best_path)
            else:
                bad_epochs += 1
                if bad_epochs >= tr["early_stopping_patience"]:
                    print(f"early stopping at epoch {epoch}")
                    break
        model.load_state_dict(torch.load(best_path, map_location=device)["model"])

    splits = ["val"] + (["test"] if cfg["evaluation"]["evaluate_test_at_end"] else [])
    for s in splits:
        probs = predict(model, loaders[s], device, amp)
        m = evaluate(frames[s], probs, t.classes, top_k)
        (run_dir / f"metrics_{s}.json").write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
        save_predictions(run_dir / f"predictions_{s}.csv", frames[s], probs, t.classes)
        print(f"{s}: obs macro-F1 {m['observation_level']['macro_f1']:.4f} | img macro-F1 {m['image_level']['macro_f1']:.4f} "
              f"| img acc {m['image_level']['accuracy']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
