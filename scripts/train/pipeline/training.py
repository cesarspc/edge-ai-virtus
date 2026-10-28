"""Training building blocks: reproducibility, class weighting, optimization, one epoch and checkpoint states.

The S5 recipe is unchanged: AdamW, linear warm-up + cosine per step, optional ``inverse_sqrt`` class weights
computed from train only, AMP only on CUDA. Checkpoint dictionaries keep the exact keys used by existing runs,
so ``best.pt``/``last.pt`` written before the refactor stay loadable and resumable.
"""
from __future__ import annotations

import math
import os
import random
from itertools import islice
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from progress_ui import bar, ready_batches

RESUME_FORMAT = 1


# --------------------------------------------------------------------------- reproducibility
def set_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False


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


# --------------------------------------------------------------------------- class imbalance
def class_weights(train: pd.DataFrame, n: int, mode: str) -> torch.Tensor | None:
    """Loss weights from train images only: ``none`` or ``inverse_sqrt`` (1/sqrt(n_c), normalized to mean 1)."""
    if mode == "none":
        return None
    counts = np.bincount(train["label_id"].astype(int), minlength=n).astype(float)
    if mode == "inverse_sqrt":
        w = 1.0 / np.sqrt(np.maximum(counts, 1))
        return torch.tensor(w / w.mean(), dtype=torch.float32)
    raise SystemExit(f"unknown class_weighting {mode}")


def class_balance_record(train: pd.DataFrame, classes: list[str], mode: str, weights: torch.Tensor | None) -> dict:
    return {"unit": "image", "computed_from": "train only", "mode": mode, "classes": classes,
            "train_counts": np.bincount(train.label_id.astype(int), minlength=len(classes)).tolist(),
            "loss_weights": weights.tolist() if weights is not None else [1.0] * len(classes)}


# --------------------------------------------------------------------------- optimization
def build_criterion(tr: dict[str, Any], weights: torch.Tensor | None, device: torch.device) -> nn.Module:
    return nn.CrossEntropyLoss(weight=weights.to(device) if weights is not None else None,
                               label_smoothing=tr["label_smoothing"])


def build_optimizer(model: nn.Module, tr: dict[str, Any]) -> torch.optim.Optimizer:
    if tr["optimizer"] != "adamw":
        raise SystemExit("only adamw is implemented")
    return torch.optim.AdamW(model.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"])


def build_scheduler(opt: torch.optim.Optimizer, tr: dict[str, Any], steps_per_epoch: int):
    """Per-step linear warm-up followed by cosine decay to zero."""
    total, warm = tr["epochs"] * steps_per_epoch, tr["warmup_epochs"] * steps_per_epoch
    return torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / max(1, warm) if s < warm
                                             else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))


def train_one_epoch(model: nn.Module, loader: DataLoader, steps: int, criterion: nn.Module,
                    opt: torch.optim.Optimizer, sched, scaler, device: torch.device, amp: bool, description: str,
                    on_progress: Callable[[int, float], None] | None = None) -> float:
    """Run ``steps`` optimization steps; ``on_progress(batches_done, mean_loss)`` every 10 batches and at the end."""
    model.train()
    losses = []
    batches = bar(islice(ready_batches(loader, f"Preparando primer lote: {description}"), steps),
                  total=steps, desc=description, unit="lote")
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
        if on_progress and ((step + 1) % 10 == 0 or step + 1 == steps):
            on_progress(step + 1, float(np.mean(losses)))
    return float(np.mean(losses))


# --------------------------------------------------------------------------- checkpoints
def atomic_checkpoint(state: dict, path: Path) -> None:
    temporary = path.with_suffix(".pt.tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def best_state(model: nn.Module, epoch: int, metric: str, score: float, classes: list[str], cfg: dict,
               fingerprint: str) -> dict:
    return {"model": model.state_dict(), "epoch": epoch, metric: score, "classes": classes,
            "config": cfg, "dataset_fingerprint": fingerprint}


def resume_state(*, epoch: int, model: nn.Module, opt, sched, scaler, generator: torch.Generator, history: list,
                 best_score: float, bad_epochs: int, training_finished: bool, best_path: Path, classes: list[str],
                 fingerprint: str, cfg: dict, device_type: str) -> dict:
    return {"resume_format": RESUME_FORMAT, "epoch": epoch, "model": model.state_dict(),
            "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
            "scaler": scaler.state_dict(), "rng": random_state(generator), "history": history,
            "best_score": best_score, "bad_epochs": bad_epochs, "training_finished": training_finished,
            "best_checkpoint": torch.load(best_path, map_location="cpu", weights_only=True),
            "classes": classes, "dataset_fingerprint": fingerprint,
            "config": cfg, "device_type": device_type}


def check_resumable(resumed: dict, classes: list[str], fingerprint: str, cfg: dict, device_type: str) -> None:
    if resumed.get("resume_format") != RESUME_FORMAT:
        raise SystemExit("Resume requires last.pt with optimizer state; legacy best.pt is not resumable.")
    if resumed["classes"] != classes or resumed["dataset_fingerprint"] != fingerprint:
        raise SystemExit("Resume labels/dataset differ from this dataset.")
    if resumed["config"] != cfg or resumed["device_type"] != device_type:
        raise SystemExit("Resume requires the same config and device type as the saved run.")
