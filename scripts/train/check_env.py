"""Check that the training environment is ready (libraries, device, dataset version, downloaded images).

Usage:
    python scripts/train/check_env.py --config configs/train/ma_resnet18_v001.yaml

Exit code 0 = ready to train; 1 = something is missing (the output says what to do).
"""
from __future__ import annotations

import argparse
import importlib
import shutil
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import load_tables, resolve, summary  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/train/ma_resnet18_v001.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(resolve(args.config).read_text(encoding="utf-8"))
    ok = True

    print("== bibliotecas")
    for mod in ("torch", "torchvision", "PIL", "sklearn", "pandas", "numpy", "yaml"):
        try:
            m = importlib.import_module(mod)
            print(f"  {mod:12s} {getattr(m, '__version__', '?')}")
        except ImportError:
            ok = False
            print(f"  {mod:12s} FALTA -> ver requirements-train.txt")

    print("== dispositivo")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  CUDA {torch.version.cuda}: {torch.cuda.get_device_name(0)} "
                  f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")
        elif torch.backends.mps.is_available():
            print("  Apple MPS")
        else:
            print("  solo CPU: el entrenamiento completo será lento (horas por época); usar --smoke para probar")
    except ImportError:
        print("  (torch no instalado)")

    print("== dataset e imágenes")
    try:
        t = load_tables(cfg)
        print(f"  versión {t.version}, huella {t.fingerprint[:16]}…, clases: {', '.join(t.classes)}")
        print("  " + summary(t).to_string(index=False).replace("\n", "\n  "))
        if len(t.missing):
            ok = False
            print(f"  faltan {len(t.missing)} imágenes -> python scripts/data/download_images.py"
                  + (f" --out {t.image_root}" if "raw/images" not in t.image_root.as_posix() else ""))
        print(f"  espacio libre en {t.image_root.anchor}: {shutil.disk_usage(t.image_root.anchor).free / 1e9:.1f} GB")
    except SystemExit as exc:
        ok = False
        print(f"  ERROR: {exc}")

    print("== resultado:", "LISTO" if ok else "FALTAN PASOS")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
