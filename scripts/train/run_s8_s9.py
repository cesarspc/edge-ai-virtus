"""Preflight and sequential A/B/C/D: image, GPS, CHELSA, then coarse substrate.

python scripts/data/prepare_s8_s9_context.py
python scripts/train/run_s8_s9.py --check
python scripts/train/run_s8_s9.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import torch
import yaml

from data import REPO_ROOT, resolve, sha256_file
from pipeline.context import load_context


def preflight(device):
    issues = []
    configs = ["configs/train/s8_multimodal_v002.yaml", "configs/train/s9_climate_v002.yaml",
               "configs/train/s10_substrate_v002.yaml"]
    manifest = json.loads(resolve("data/versions/v002/manifest.json").read_text(encoding="utf-8"))
    if device == "cuda" and not torch.cuda.is_available():
        issues.append("CUDA unavailable in this Python environment")
    for name in configs:
        cfg = yaml.safe_load(resolve(name).read_text(encoding="utf-8"))
        for path in [f"data/processed/{cfg['data']['version']}/observations.csv",
                     f"data/processed/{cfg['data']['version']}/images.csv", cfg["data"]["download_manifest"],
                     cfg["initialization"]["checkpoint"]]:
            if not resolve(path).is_file():
                issues.append(f"Missing: {path}")
        if not (resolve(cfg["data"]["image_root"]) / "300p").is_dir():
            issues.append(f"Missing photographs: {cfg['data']['image_root']}/300p")
        try:
            load_context(cfg, REPO_ROOT, manifest["content_fingerprint"], sha256_file)
        except (OSError, ValueError, KeyError) as exc:
            issues.append(f"Context sidecar: {exc}")
    return sorted(set(issues))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--stage", choices=["s8", "s9", "s10", "all"], default="all")
    args = ap.parse_args()
    issues = preflight(args.device)
    result = {"device": args.device, "ready": not issues, "issues": issues,
              "test_evaluations": 0, "image_integrity": "checked freshly by train_s8.py before each run"}
    path = resolve("reports/s9/preflight.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    if issues:
        return 1
    if args.check:
        return 0
    jobs = []
    if args.stage in ("s8", "all"):
        jobs += [("configs/train/s8_multimodal_v002.yaml", "visual"),
                 ("configs/train/s8_multimodal_v002.yaml", "multimodal")]
    if args.stage in ("s9", "all"):
        jobs += [("configs/train/s9_climate_v002.yaml", "multimodal")]
    if args.stage in ("s10", "all"):
        jobs += [("configs/train/s10_substrate_v002.yaml", "multimodal")]
    for config, mode in jobs:
        subprocess.run([sys.executable, str(Path(__file__).with_name("train_s8.py")), "--config", config,
                        "--mode", mode, "--device", args.device], cwd=REPO_ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
