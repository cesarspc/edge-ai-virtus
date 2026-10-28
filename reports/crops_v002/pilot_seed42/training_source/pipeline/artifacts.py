"""Run provenance and human-readable outputs: run.json content, progress.json heartbeat and RESULTS.md."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TRAIN_DIR = Path(__file__).resolve().parents[1]
# Every source file that can change what a run computes; hashed into run.json["code_sha256"].
CODE_FILES = ("train_ma.py", "data.py", "metrics.py", "progress_ui.py",
              "pipeline/__init__.py", "pipeline/datasets.py", "pipeline/transforms.py", "pipeline/models.py",
              "pipeline/training.py", "pipeline/evaluation.py", "pipeline/artifacts.py",
              "pipeline/crops.py", "pipeline/crop_state.py", "pipeline/drift.py")


def git_commit(repo_root: Path) -> dict[str, Any]:
    def run(*a):
        try:
            return subprocess.run(["git", *a], cwd=repo_root, capture_output=True, text=True, check=True).stdout.strip()
        except Exception:  # noqa: BLE001
            return ""
    return {"commit": run("rev-parse", "HEAD") or None, "dirty": bool(run("status", "--porcelain"))}


def code_hashes() -> dict[str, str]:
    return {p: hashlib.sha256((TRAIN_DIR / p).read_bytes()).hexdigest() for p in CODE_FILES}


def environment(device) -> dict[str, Any]:
    import torch
    import torchvision
    return {"versions": {"python": platform.python_version(), "torch": torch.__version__,
                         "torchvision": torchvision.__version__, "cuda": torch.version.cuda,
                         "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None},
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


class ProgressFile:
    """Atomically rewritten ``progress.json`` so an external watcher never reads a partial file."""

    def __init__(self, run_dir: Path, run_id: str):
        self.target, self.run_id = run_dir / "progress.json", run_id

    def __call__(self, stage: str, **values) -> None:
        state = {"run_id": self.run_id, "stage": stage, "updated_utc": datetime.now(timezone.utc).isoformat(), **values}
        temporary = self.target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
        temporary.replace(self.target)


def results_markdown(run_id: str, version: str, fingerprint: str, device, mode: str,
                     results: dict[str, dict], smoke: bool) -> str:
    lines = [f"# Resultado {run_id}", "", f"Dataset: `{version}` · huella: `{fingerprint}`.",
             f"Dispositivo: `{device}` · modo: `{mode}`.", "",
             "| Partición | Imágenes | Observaciones | Macro-F1 imagen | Macro-F1 observación | Accuracy observación |",
             "|---|---:|---:|---:|---:|---:|"]
    for s, m in results.items():
        a, b = m["image_level"], m["observation_level"]
        lines.append(f"| {s} | {a['n']} | {b['n']} | {a['macro_f1']:.4f} | {b['macro_f1']:.4f} | {b['accuracy']:.4f} |")
    lines += ["", "| Partición | Género | Observaciones | F1 |", "|---|---|---:|---:|"]
    for s, m in results.items():
        for c in m["observation_level"]["per_class"]:
            lines.append(f"| {s} | {c['genus']} | {c['support']} | {c['f1']:.4f} |")
    if smoke:
        lines += ["", "Prueba de ejecución: estas métricas no estiman rendimiento del modelo."]
    return "\n".join(lines) + "\n"
