"""Frozen C crop study on official val ONLY, with atomic blocks and exact-input resume.

python -u scripts/train/evaluate_crop_sensitivity.py
python -u scripts/train/evaluate_crop_sensitivity.py --resume reports/crops_v002/<study>
"""
from __future__ import annotations

import argparse
import json
from importlib.metadata import version as package_version
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from progress_ui import bar, stage, ready_batches

with stage("Cargando bibliotecas del estudio de recortes", active=__name__ == "__main__"):
    import numpy as np
    import pandas as pd
    import torch
    import yaml
    from torch.utils.data import DataLoader
    from data import REPO_ROOT, load_tables, resolve, sha256_file
    from pipeline.artifacts import ProgressFile, environment, git_commit
    from pipeline.crop_state import atomic_json, atomic_npz, load_blocks, signature, state_digest
    from pipeline.crops import CropTable, VIEWS
    from pipeline.evaluation import evaluate
    from pipeline.models import build_model
    from pipeline.predictions import load_predictions
    from pipeline.training import set_seed
    from pipeline.transforms import build_transforms


INFERENCE_FILES = ("evaluate_crop_sensitivity.py", "pipeline/crops.py", "pipeline/crop_state.py",
                   "pipeline/transforms.py", "pipeline/models.py", "data.py", "progress_ui.py")


def check_control(folder, reference, frame, probs, classes):
    old, old_probs = load_predictions(reference / "predictions_val.csv", classes)
    old = old.assign(_index=np.arange(len(old))).set_index("image_uid").loc[frame.image_uid]
    old_probs = old_probs[old._index.to_numpy()]
    if not np.array_equal(old.observation_uid.to_numpy(), frame.observation_uid.to_numpy()) or not np.array_equal(old.label_id.to_numpy(), frame.label_id.to_numpy()):
        raise ValueError("C reference and current val rows/labels do not match")
    saved = json.loads((reference / "metrics_val.json").read_text(encoding="utf-8"))
    m = evaluate(frame, probs, classes, [1, 3])
    image_changes = int((probs.argmax(1) != old.pred_label_id.to_numpy()).sum())
    confusion_matches = all(m[level]["confusion_matrix"] == saved[level]["confusion_matrix"] for level in m)
    result = {"identical_image_decisions": image_changes == 0, "changed_image_decisions": image_changes,
              "confusion_matrices_match_saved_C": confusion_matches,
              "max_probability_difference_vs_rounded_C_csv": float(np.max(np.abs(probs-old_probs))),
              "macro_f1_observation": m["observation_level"]["macro_f1"]}
    atomic_json(folder / "control_reproduction.json", result)
    if image_changes or not confusion_matches:
        raise ValueError("Control does not reproduce C; inspect control_reproduction.json before interpreting crops")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/experiments/crops_v002.yaml")
    ap.add_argument("--resume", help="existing study directory; never changes input hashes")
    ap.add_argument("--pause-after-batches", type=int, help="exit after this many NEW saved batches")
    args = ap.parse_args()
    if args.pause_after_batches is not None and args.pause_after_batches < 1:
        ap.error("--pause-after-batches must be positive")
    config = yaml.safe_load(resolve(args.config).read_text(encoding="utf-8"))
    ev = config["evaluation"]
    if not 1 <= ev["batch_size"] <= 64:
        ap.error("this campaign uses a maximum of 64 views per GPU forward (AMP stability)")
    selection_path = resolve(config["reference_selection"])
    selection = json.loads(selection_path.read_text(encoding="utf-8"))["selected"]
    reference = resolve(selection["run"])
    for name, key in (("best.pt", "checkpoint_sha256"), ("run.json", "run_json_sha256"), ("metrics_val.json", "metrics_val_sha256")):
        if sha256_file(reference / name) != selection[key]:
            raise ValueError(f"selected reference changed: {name}")
    info = json.loads((reference / "run.json").read_text(encoding="utf-8"))
    cfg = info["config"]
    if cfg["data"].get("split_column") != "split" or cfg["data"]["version"] != "v002":
        raise ValueError("this study requires v002 official partitions")
    folder = resolve(args.resume) if args.resume else resolve(config["output_dir"]) / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ_frozen_C")
    folder.mkdir(parents=True, exist_ok=bool(args.resume))
    (folder / "chunks").mkdir(exist_ok=True)
    print(f"Estudio: {folder}", flush=True)
    progress = ProgressFile(folder, folder.name)
    progress("verifying_validation")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(ev["amp"]) and device.type == "cuda"
    set_seed(ev["seed"], False)
    tables = load_tables(cfg, show_progress=True, splits=("val",))
    frame = tables.frames["val"]
    if len(tables.missing) or len(frame) != info["dataset"]["images"]["val"]:
        raise ValueError("complete official val is required; missing/modified images must be repaired")
    if tables.fingerprint != selection["dataset_fingerprint"]:
        raise ValueError("selected model and dataset fingerprint differ")
    source_dir = Path(__file__).parent
    identity = {"config": config, "checkpoint_sha256": selection["checkpoint_sha256"],
                "reference_run_sha256": selection["run_json_sha256"], "fingerprint": tables.fingerprint,
                "classes": tables.classes, "validation_order_sha256": signature(frame[["image_uid", "observation_uid", "label_id"]].to_dict("list")),
                "views": VIEWS, "device": str(device), "amp": amp, "environment": environment(device),
                "image_libraries": {name: package_version(name) for name in ["Pillow", "numpy", "pandas"]},
                "code_sha256": {name: sha256_file(source_dir / name) for name in INFERENCE_FILES}}
    sig = signature(identity)
    study_path = folder / "study.json"
    if study_path.exists():
        if json.loads(study_path.read_text(encoding="utf-8"))["inference_signature"] != sig:
            raise ValueError("resume requires identical data/model/config/code/device/library versions")
    else:
        atomic_json(study_path, {**identity, "inference_signature": sig, "reference_run": str(reference.relative_to(REPO_ROOT)),
                                 "git": git_commit(REPO_ROOT), "test_images_opened": False, "split": "val"})
        frame.drop(columns="path").to_csv(folder / "validation_images.csv", index=False)
    chunks, times, peaks, done = load_blocks(folder / "chunks", sig, len(frame), ev["batch_size"], len(VIEWS), len(tables.classes))
    boxes = []
    for path in sorted((folder / "chunks").glob("block_*.npz")):
        with np.load(path, allow_pickle=False) as block:
            boxes.append(block["boxes"].copy())
    with stage("Cargando C congelado y comprobando su estado"):
        checkpoint = torch.load(reference / "best.pt", weights_only=True, map_location="cpu")
        if checkpoint["classes"] != tables.classes or checkpoint["dataset_fingerprint"] != tables.fingerprint:
            raise ValueError("checkpoint metadata mismatch")
        before = state_digest(checkpoint["model"])
        model = build_model(cfg, pretrained=False)
        model.load_state_dict(checkpoint["model"], strict=True)
        model.requires_grad_(False).eval().to(device)
        del checkpoint
    batch_size = ev["batch_size"]
    start_at, new_batches = done, 0
    started = time.monotonic()
    if done < len(frame):
        _, tf_eval = build_transforms(cfg)
        kwargs = dict(num_workers=ev["num_workers"], pin_memory=device.type == "cuda", shuffle=False)
        if ev["num_workers"]:
            kwargs.update(prefetch_factor=1, persistent_workers=False)
        loader = DataLoader(CropTable(frame.iloc[done:], tf_eval), batch_size=batch_size, **kwargs)
        # Kernel warm-up is excluded from reported forward time. Eval mode leaves BN unchanged.
        with stage("Preparando kernels de inferencia"), torch.inference_mode():
            dummy = torch.zeros(batch_size, 3, cfg["input"]["size"], cfg["input"]["size"], device=device)
            with torch.autocast(device_type=device.type, enabled=amp):
                model(dummy)
            del dummy
            if device.type == "cuda":
                torch.cuda.synchronize()
        with torch.inference_mode(), bar(total=len(frame), initial=done, desc="Recortes: 7 vistas por foto", unit="foto") as pbar:
            for x, _, geometry in ready_batches(loader, "Preparando primer lote de recortes"):
                values, secs, memory = [], [], []
                for v, (name, _, _) in enumerate(VIEWS):
                    progress("inference", images_completed=done, images_total=len(frame), view=name)
                    if device.type == "cuda":
                        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                    t0 = time.perf_counter()
                    # One view of at most 64 photos per forward, never a batch of 64*7.
                    inputs = x[:, v].contiguous().to(device, non_blocking=True)
                    with torch.autocast(device_type=device.type, enabled=amp):
                        logits = model(inputs)
                    if not torch.isfinite(logits).all():
                        raise FloatingPointError(f"non-finite logits in view {name}; completed blocks remain resumable")
                    p = torch.softmax(logits.float(), 1).cpu().numpy()
                    if not np.isfinite(p).all():
                        raise FloatingPointError("non-finite probabilities")
                    secs.append(time.perf_counter()-t0)
                    memory.append(torch.cuda.max_memory_allocated() if device.type == "cuda" else 0)
                    values.append(p)
                    del inputs, logits
                end = done + len(x)
                block_probs = np.stack(values, axis=1)
                geometry = geometry.numpy()
                atomic_npz(folder / "chunks" / f"block_{done:06d}.npz", signature=sig, start=done, end=end,
                           probs=block_probs, seconds=np.array(secs), peak_bytes=np.array(memory), boxes=geometry)
                chunks.append(block_probs); times.append(np.array(secs)); peaks.append(np.array(memory)); boxes.append(geometry)
                done = end; new_batches += 1; pbar.update(len(x))
                progress("saved", images_completed=done, images_total=len(frame), checkpoint=f"chunks/block_{done-len(x):06d}.npz")
                if args.pause_after_batches and new_batches >= args.pause_after_batches:
                    progress("paused", images_completed=done, images_total=len(frame))
                    print(f"Guardado: {done}/{len(frame)} fotos. Reanudar con --resume \"{folder}\"", flush=True)
                    return 0
    after = state_digest(model.state_dict())
    atomic_json(folder / "weights_unchanged.json", {"initial_state_sha256": before, "final_state_sha256": after,
                "identical": before == after, "includes_parameters_and_BatchNorm_buffers": True, "optimizer_steps": 0})
    if before != after:
        raise ValueError("frozen inference modified model state")
    probs = np.concatenate(chunks)
    atomic_npz(folder / "probabilities_val.npz", probs=probs, seconds=np.stack(times), peak_bytes=np.stack(peaks))
    geometry = np.concatenate(boxes)
    records = []
    for i, uid in enumerate(frame.image_uid):
        for v, (name, fraction, anchor) in enumerate(VIEWS):
            w, h, left, top, right, bottom = map(int, geometry[i, v])
            records.append({"image_uid": uid, "observation_uid": frame.iloc[i].observation_uid, "split": "val", "view": name,
                            "requested_area_fraction": fraction, "anchor": anchor, "width": w, "height": h,
                            "left": left, "top": top, "right": right, "bottom": bottom,
                            "realized_area_fraction": (right-left)*(bottom-top)/(w*h)})
    pd.DataFrame(records).to_csv(folder / "crop_coordinates.csv", index=False)
    progress("checking_control")
    check_control(folder, reference, frame, probs[:, 0], tables.classes)
    progress("analysis")
    from analyze_crop_experiments import analyze
    rows, decision = analyze(folder)
    atomic_json(folder / "completed.json", {"images": len(frame), "observations": int(frame.observation_uid.nunique()),
                "test_evaluations": 0, "weights_unchanged": True, "inference_signature": sig,
                "analysis_script_sha256": sha256_file(source_dir / "analyze_crop_experiments.py"),
                "new_images_this_session": done-start_at, "session_seconds": time.monotonic()-started})
    progress("completed", images_completed=done, images_total=len(frame), recommended_policy=decision["recommended_inference_policy"])
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
    print(f"Resultados y comparaciones: {folder / 'RESULTS.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
