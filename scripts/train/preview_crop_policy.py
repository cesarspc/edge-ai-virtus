"""Local gallery: one deterministic TRAIN observation per genus, no test photographs.

Each finished row is saved so interruption does not discard the gallery progress.
Dataset photographs stay local and remain excluded from Git.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from progress_ui import stage, bar

with stage("Preparando galeria de recortes", active=__name__ == "__main__"):
    import numpy as np
    import pandas as pd
    import yaml
    from PIL import Image, ImageDraw, ImageFont, ImageOps
    from data import REPO_ROOT, resolve, sha256_file
    from pipeline.crop_state import atomic_json, signature
    from pipeline.crops import crop_image, VIEWS
    from pipeline.transforms import build_transforms


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    folder = resolve(args.output)
    (folder / "rows").mkdir(parents=True, exist_ok=True)
    version = REPO_ROOT / "data/versions/v002"
    manifest = json.loads((version / "manifest.json").read_text(encoding="utf-8"))
    table_path = REPO_ROOT / "data/processed/v002/images.csv"
    expected = next(o["sha256"] for o in manifest["outputs"] if o["name"] == "images.csv")
    if sha256_file(table_path) != expected:
        raise ValueError("dataset table differs from published v002")
    cfg = yaml.safe_load((REPO_ROOT / "configs/train/s5_mobilenet_v3_large_v002_c.yaml").read_text())
    images = pd.read_csv(table_path, low_memory=False)
    train = images[images.split.eq("train")].sort_values("image_uid").drop_duplicates("observation_uid")
    chosen = train.groupby("label_id", group_keys=False).sample(n=1, random_state=42).sort_values("label_id")
    downloads = pd.read_csv(resolve(cfg["data"]["download_manifest"]), dtype=str)
    chosen = chosen.merge(downloads[["image_uid", "filename"]], on="image_uid", validate="one_to_one")
    integrity = pd.read_csv(version / "image_integrity.csv", dtype=str).set_index("image_uid").sha256
    chosen["source_sha256"] = chosen.image_uid.map(integrity)
    identity = {"dataset": manifest["content_fingerprint"], "images": chosen[["image_uid", "observation_uid", "genus", "source_sha256"]].to_dict("list"),
                "views": VIEWS, "input": cfg["input"], "seed": 42,
                "code": {name: sha256_file(Path(__file__).parent / name) for name in ["preview_crop_policy.py", "pipeline/crops.py", "pipeline/transforms.py"]}}
    sig = signature(identity)
    if (folder / "gallery.json").exists() and json.loads((folder / "gallery.json").read_text())["signature"] != sig:
        raise ValueError("gallery inputs changed; select a new output folder")
    atomic_json(folder / "gallery.json", {**identity, "signature": sig, "split": "train", "test_images_opened": False})
    chosen[["image_uid", "observation_uid", "genus", "split", "source_sha256"]].to_csv(folder / "gallery_index.csv", index=False)
    _, transform = build_transforms(cfg)
    try: font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 14)
    except OSError: font = ImageFont.load_default()
    width, row_height = 8*180, 220
    canvas = Image.new("RGB", (width, row_height*(len(chosen)+1)), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), "Vistas de TRAIN: original con encuadre completo + entradas normalizadas del clasificador", fill="black", font=font)
    labels = ["Original (contexto)"] + [name for name, _, _ in VIEWS]
    for col, label in enumerate(labels): draw.text((col*180+8, 60), label, fill="black", font=font)
    for row_index, row in enumerate(bar(chosen.itertuples(), total=len(chosen), desc="Galeria train", unit="genero")):
        path = resolve(cfg["data"]["image_root"]) / "300p" / row.filename
        if sha256_file(path) != row.source_sha256:
            raise ValueError(f"source image changed: {row.image_uid}")
        saved = folder / "rows" / f"{int(row.label_id):02d}.png"
        if not saved.exists():
            with Image.open(path) as original:
                image = original.convert("RGB")
                sheet = Image.new("RGB", (width, row_height), "white")
                title = ImageDraw.Draw(sheet)
                title.text((8, 2), f"{row.genus} | {row.image_uid}", fill="black", font=font)
                thumb = ImageOps.contain(image, (174, 174))
                sheet.paste(thumb, ((180-thumb.width)//2, 26+(174-thumb.height)//2))
                for col, (_, fraction, anchor) in enumerate(VIEWS, start=1):
                    x = transform(crop_image(image, fraction, anchor)).numpy().transpose(1, 2, 0)
                    x = np.clip(x*np.asarray(cfg["input"]["std"])+np.asarray(cfg["input"]["mean"]), 0, 1)
                    tile = Image.fromarray(np.round(x*255).astype(np.uint8)).resize((174, 174))
                    sheet.paste(tile, (col*180+3, 26))
                tmp = saved.with_suffix(".png.tmp")
                sheet.save(tmp, format="PNG"); tmp.replace(saved)
        with Image.open(saved) as sheet: canvas.paste(sheet, (0, (row_index+1)*row_height))
        atomic_json(folder / "progress.json", {"stage": "gallery", "rows_completed": row_index+1, "rows_total": len(chosen)})
    tmp = folder / "gallery_train.png.tmp"
    canvas.save(tmp, format="PNG"); tmp.replace(folder / "gallery_train.png")
    atomic_json(folder / "progress.json", {"stage": "completed", "rows_completed": len(chosen), "rows_total": len(chosen)})
    print(f"Galeria local: {folder / 'gallery_train.png'}", flush=True)


if __name__ == "__main__":
    main()
