"""Exercise a real ResNet training/checkpoint cycle offline without touching test images."""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import yaml
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "train"))
import data as tables
import train_ma


class TrainingSmokeTests(unittest.TestCase):
    def test_smoke_and_checkpoint_evaluation_never_read_test_images(self):
        temporary_root = REPO / "tmp"
        temporary_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as directory:
            root = Path(directory)
            assert root.resolve().is_relative_to(temporary_root.resolve())
            proc, version = root / "data/processed/test_version", root / "data/versions/test_version"
            proc.mkdir(parents=True)
            version.mkdir(parents=True)
            image_root = root / "photos"
            (image_root / "300p").mkdir(parents=True)
            cfg = yaml.safe_load((REPO / "configs/train/ma_resnet18_v002.yaml").read_text(encoding="utf-8"))
            cfg["data"].update(version="test_version", image_root=str(image_root), download_manifest=str(root / "downloads.csv"))
            cfg["input"].update(size=32, eval_resize=32)
            cfg["model"]["pretrained_weights"] = None
            cfg["training"].update(num_workers=0, batch_size=8)
            cfg["output"]["runs_dir"] = str(root / "runs")
            classes = [f"Genus{i}" for i in range(10)]
            img, obs, downloads, integrity = [], [], [], []
            for split in ("train", "val", "test"):
                for label, genus in enumerate(classes):
                    uid = f"{split}_{label}"
                    filename = uid + ".jpg"
                    img.append({"image_uid": uid, "observation_uid": uid, "source_id": "fungitastic",
                                "source_image_id": filename, "split": split, "label_id": label, "genus": genus})
                    obs.append({"observation_uid": uid, "split": split, "split_temporal": split})
                    if split != "test":  # No test file exists: any attempt to predict test must fail.
                        Image.new("RGB", (40, 40), (label * 20, 40, 80)).save(image_root / "300p" / filename)
                        downloads.append({"image_uid": uid, "filename": filename, "status": "ok"})
                    integrity.append({"image_uid": uid, "filename": filename,
                                      "sha256": hashlib.sha256((image_root / "300p" / filename).read_bytes()).hexdigest()
                                      if split != "test" else "0" * 64})
            pd.DataFrame(img).to_csv(proc / "images.csv", index=False)
            pd.DataFrame(obs).to_csv(proc / "observations.csv", index=False)
            pd.DataFrame({"label_id": range(10), "genus": classes}).to_csv(version / "classes.csv", index=False)
            pd.DataFrame(downloads).to_csv(root / "downloads.csv", index=False)
            pd.DataFrame(integrity).to_csv(version / "image_integrity.csv", index=False)
            outputs = [{"name": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                       for p in (proc / "images.csv", proc / "observations.csv", version / "classes.csv", version / "image_integrity.csv")]
            (version / "manifest.json").write_text(json.dumps({"outputs": outputs, "content_fingerprint": "fixture"}))
            config = root / "config.yaml"
            config.write_text(yaml.safe_dump(cfg), encoding="utf-8")
            with patch.object(tables, "REPO_ROOT", root), patch.object(train_ma, "REPO_ROOT", root):
                with patch.object(sys, "argv", ["train_ma", "--config", str(config), "--smoke", "--device", "cpu"]):
                    self.assertEqual(train_ma.main(), 0)
                run = next((root / "runs").iterdir())
                self.assertTrue((run / "best.pt").is_file())
                self.assertTrue((run / "metrics_val.json").is_file())
                self.assertFalse((run / "metrics_test.json").exists())
                self.assertEqual(json.loads((run / "completed.json").read_text())["test_evaluations"], 0)
                with patch.object(sys, "argv", ["train_ma", "--config", str(config), "--eval-only", str(run / "best.pt"), "--device", "cpu"]):
                    self.assertEqual(train_ma.main(), 0)
                self.assertFalse(list((root / "runs").rglob("metrics_test.json")))
                (image_root / "300p/train_0.jpg").write_bytes(b"changed after publication")
                loaded = tables.load_tables(cfg)
                self.assertEqual(loaded.missing.loc[loaded.missing["image_uid"] == "train_0", "status"].tolist(), ["hash_mismatch"])


if __name__ == "__main__":
    unittest.main()
