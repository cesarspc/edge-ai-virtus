"""End-to-end 2x2 aggregation and inference-block resume on a tiny local fixture."""
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
from torch.utils.data import DataLoader
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/train"))
import evaluate_crop_pair as pair
from data import DatasetTables
from pipeline.drift import initialize_drift, measure_drift
from pipeline.crop_state import state_digest


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(3, 10)

    def forward(self, x):
        return self.classifier(self.pool(x).flatten(1))


class CropPairTests(unittest.TestCase):
    def test_matrix_and_resume_without_repeating_inference(self):
        import copy
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory(dir=REPO / "tmp") as directory:
                root = Path(directory)
                cfg = yaml.safe_load((REPO / "configs/train/cropping_mobilenet_v3_large_v002_control.yaml").read_text())
                cfg["input"].update(size=16, eval_resize=16)
                classes = [f"G{i}" for i in range(10)]
                records = []
                for label, genus in enumerate(classes):
                    path = root / f"{label}.png"
                    Image.new("RGB", (25, 20), (label*20, 60, 40)).save(path)
                    records.append(dict(image_uid=f"i{label}", observation_uid=f"o{label}", label_id=label, genus=genus, path=str(path)))
                frame = pd.DataFrame(records)
                tables = DatasetTables("fixture", "fixture", classes, {"val": frame}, pd.DataFrame(), root)
                origin = Tiny()
                initial = {key: value.clone() for key, value in origin.state_dict().items()}
                for branch in ["control", "cropped"]:
                    run = root / branch
                    run.mkdir()
                    config = copy.deepcopy(cfg)
                    if branch == "cropped": config["input"]["extra_crop"] = dict(area_fraction=.75, train_anchors=["center"], eval_anchors=["center"])
                    model = Tiny(); model.load_state_dict(initial)
                    drift = initialize_drift(model, {})
                    with torch.no_grad(): model.classifier.weight.add_(.001 if branch == "control" else .002)
                    measure_drift(model, drift, 1, 1, 1)
                    torch.save(dict(model=model.state_dict(), classes=classes, dataset_fingerprint="fixture", epoch=1), run / "best.pt")
                    torch.save(dict(model=initial), run / "initial.pt")
                    (run / "run.json").write_text(json.dumps(dict(config=config)))
                    (run / "initialization.json").write_text(json.dumps(dict(state_sha256=state_digest(initial), sha256="same")))
                    pd.DataFrame(drift["summaries"]).to_csv(run / "parameter_drift_summary.csv", index=False)
                    pd.DataFrame([dict(epoch=1, train_loss=1)]).to_csv(run / "history.csv", index=False)
                output = root / "analysis"
                argv = ["pair", "--control", str(root / "control"), "--cropped", str(root / "cropped"), "--output", str(output)]
                def loader(*args, **kwargs):
                    kwargs.update(num_workers=0, pin_memory=False)
                    kwargs.pop("prefetch_factor", None)
                    return DataLoader(*args, **kwargs)
                with patch.object(pair, "REPO_ROOT", root), patch.object(pair, "load_tables", return_value=tables), patch.object(pair, "build_model", side_effect=lambda *a, **kw: Tiny()), patch.object(torch.cuda, "is_available", return_value=False), patch.object(sys, "argv", argv):
                    with patch.object(pair, "DataLoader", side_effect=loader): pair.main()
                    before = (output / "validation_2x2.csv").read_bytes()
                    with patch.object(pair, "DataLoader", side_effect=AssertionError("cached inference was repeated")): pair.main()
                self.assertEqual((output / "validation_2x2.csv").read_bytes(), before)
                self.assertEqual(len(pd.read_csv(output / "validation_2x2.csv")), 4)
                self.assertEqual(len(pd.read_csv(output / "classifier_rows_per_genus.csv")), 20)
                self.assertEqual(json.loads((output / "completed.json").read_text())["test_evaluations"], 0)
        finally:
            torch.set_num_threads(old_threads)


if __name__ == "__main__": unittest.main()
