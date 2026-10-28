"""Fresh initialization resets optimization; drift and its reference survive resume."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
from PIL import Image
import torch
from torch import nn
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/train"))
import data
import train_ma
from pipeline.crop_state import state_digest
from pipeline.drift import initialize_drift, measure_drift
from pipeline.transforms import build_transforms
from pipeline.evaluation import predict


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        self.classifier = nn.Linear(4, 10)

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


class CropFineTuningTests(unittest.TestCase):
    def test_extra_crop_is_optional_and_multiview_averages_probabilities(self):
        from torchvision import transforms
        from torch.utils.data import DataLoader, TensorDataset
        cfg = yaml.safe_load((REPO / "configs/train/s5_mobilenet_v3_large_v002_c.yaml").read_text())
        _, control = build_transforms(cfg)
        image = Image.new("RGB", (400, 300), (100, 50, 25))
        inp = cfg["input"]
        original = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(inp["mean"], inp["std"])])
        self.assertTrue(torch.equal(control(image), original(image)))
        cfg["input"]["extra_crop"] = dict(area_fraction=.75, train_anchors=["center"], eval_anchors=["center", "top_left"])
        _, multi = build_transforms(cfg)
        self.assertEqual(tuple(multi(image).shape), (2, 3, 224, 224))
        model = nn.Sequential(nn.Flatten(), nn.Linear(3*2*2, 3)).eval()
        x = torch.rand(5, 2, 3, 2, 2)
        loader = DataLoader(TensorDataset(x, torch.zeros(5)), batch_size=3)
        with torch.no_grad():
            expected = torch.stack([model(x[:, v]).softmax(1) for v in range(2)]).mean(0).numpy()
        actual = predict(model, loader, torch.device("cpu"), False)
        __import__("numpy").testing.assert_allclose(actual, expected, atol=1e-7)

    def test_frozen_state_and_buffer_drift_are_distinct(self):
        model = TinyModel()
        record = initialize_drift(model, {})
        measure_drift(model, record, 0, 0, 0)
        self.assertTrue(all(r["absolute_delta_l2"] == 0 for r in record["rows"]))
        digest = state_digest(model.state_dict())
        with torch.no_grad(): model.features[1].running_mean.add_(1)
        self.assertNotEqual(digest, state_digest(model.state_dict()))
        measure_drift(model, record, 1, 0, 0)
        epoch = [r for r in record["rows"] if r["epoch"] == 1]
        self.assertTrue(all(r["absolute_delta_l2"] == 0 for r in epoch if r["kind"] == "parameter"))
        self.assertTrue(any(r["absolute_delta_l2"] > 0 for r in epoch if r["group"] == "bn_running_statistics"))

    def test_fresh_start_and_resume_keep_the_same_initial_reference(self):
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory(dir=REPO / "tmp") as directory:
                root = Path(directory)
                proc, version, photos = root / "data/processed/fixture", root / "data/versions/fixture", root / "photos/300p"
                for p in (proc, version, photos): p.mkdir(parents=True)
                classes = [f"Genus{i}" for i in range(10)]
                images, observations, downloads = [], [], []
                for split in ["train", "val"]:
                    for label, genus in enumerate(classes):
                        uid = f"{split}_{label}"
                        Image.new("RGB", (25, 20), (label*20, 50, 60)).save(photos / f"{uid}.jpg")
                        images.append(dict(image_uid=uid, observation_uid=uid, source_id="fungitastic", source_image_id=f"{uid}.jpg", split=split, label_id=label, genus=genus))
                        observations.append(dict(observation_uid=uid, split=split, split_temporal=split))
                        downloads.append(dict(image_uid=uid, filename=f"{uid}.jpg", status="ok"))
                pd.DataFrame(images).to_csv(proc / "images.csv", index=False)
                pd.DataFrame(observations).to_csv(proc / "observations.csv", index=False)
                pd.DataFrame(dict(label_id=range(10), genus=classes)).to_csv(version / "classes.csv", index=False)
                pd.DataFrame(downloads).to_csv(root / "downloads.csv", index=False)
                outputs = [{"name": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in [proc / "images.csv", proc / "observations.csv", version / "classes.csv"]]
                (version / "manifest.json").write_text(json.dumps(dict(outputs=outputs, content_fingerprint="fixture")))
                cfg = yaml.safe_load((REPO / "configs/train/s5_mobilenet_v3_large_v002_c.yaml").read_text())
                cfg["experiment"]["phase"] = "CROPS"
                cfg["data"].update(version="fixture", image_root=str(photos.parent), download_manifest=str(root / "downloads.csv"))
                cfg["input"].update(size=16, eval_resize=16)
                cfg["training"].update(epochs=2, num_workers=0, batch_size=4, track_parameter_drift=True, early_stopping_patience=5, amp=False)
                cfg["evaluation"]["batch_size"] = 4
                cfg["output"]["runs_dir"] = str(root / "runs")
                origin = TinyModel()
                original_hash = state_digest(origin.state_dict())
                init = root / "origin.pt"
                torch.save(dict(model=origin.state_dict(), classes=classes, dataset_fingerprint="fixture", config=cfg, epoch=99), init)
                cfg["initialization"] = {"checkpoint": str(init)}
                config = root / "config.yaml"
                config.write_text(yaml.safe_dump(cfg))
                with patch.object(data, "REPO_ROOT", root), patch.object(train_ma, "REPO_ROOT", root), patch.object(train_ma, "build_model", side_effect=lambda *a, **kw: TinyModel()):
                    with patch.object(sys, "argv", ["train_ma", "--config", str(config), "--device", "cpu", "--pause-after-epoch"]):
                        self.assertEqual(train_ma.main(), 0)
                    first = next((root / "runs").iterdir())
                    saved = torch.load(first / "last.pt", map_location="cpu", weights_only=True)
                    self.assertEqual(saved["epoch"], 1)  # not source epoch 99
                    self.assertEqual(saved["parameter_drift"]["initial_state_sha256"], original_hash)
                    self.assertFalse((first / "completed.json").exists())
                    self.assertTrue(saved["parameter_drift"]["rows"])
                    with patch.object(sys, "argv", ["train_ma", "--config", str(config), "--device", "cpu", "--resume", str(first / "last.pt")]):
                        self.assertEqual(train_ma.main(), 0)
                    resumed = [p for p in (root / "runs").iterdir() if p != first][0]
                    last = torch.load(resumed / "last.pt", map_location="cpu", weights_only=True)
                    self.assertEqual(last["epoch"], 2)
                    self.assertEqual(last["parameter_drift"]["initial_state_sha256"], original_hash)
                    self.assertEqual({r["epoch"] for r in last["parameter_drift"]["rows"]}, {1, 2})
                    self.assertEqual(last["parameter_drift"]["rows"][:len(saved["parameter_drift"]["rows"])], saved["parameter_drift"]["rows"])
                    self.assertEqual(json.loads((resumed / "completed.json").read_text())["test_evaluations"], 0)
                    self.assertFalse(list(root.rglob("metrics_test.json")))
        finally:
            torch.set_num_threads(old_threads)


if __name__ == "__main__":
    unittest.main()
