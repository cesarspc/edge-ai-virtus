"""Exercise corrected S9 training, pause/resume and all val sensitivities offline."""
import json
from pathlib import Path
import sys
import tempfile

import pandas as pd
import pytest
from PIL import Image
import torch
from torch import nn
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/train"))
import train_s8
from data import DatasetTables
from pipeline.context import BIO


@pytest.mark.parametrize("stage", ["s9", "s10"])
def test_corrected_training_resume_and_val_only_conditions(monkeypatch, stage):
    temporary_root = Path(__file__).resolve().parents[1] / "tmp"
    temporary_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=temporary_root) as directory:
        root = Path(directory)
        config_name = "s9_climate_v002.yaml" if stage == "s9" else "s10_substrate_v002.yaml"
        cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/train" / config_name).read_text())
        cfg["training"].update(epochs=1, batch_size=2, num_workers=0, amp=False)
        cfg["input"].update(size=8, eval_resize=8)
        cfg["evaluation"]["batch_size"] = 2
        cfg["model"]["num_classes"] = 2
        cfg["output"]["runs_dir"] = str(root / "runs")
        cfg["initialization"]["checkpoint"] = str(root / "initial.pt")
        frames, observations = {}, []
        for split in ("train", "val"):
            rows = []
            for label in range(2):
                uid = f"{split}_{label}"
                path = root / f"{uid}.png"
                Image.new("RGB", (10, 10), (20 + label * 100, 40, 60)).save(path)
                rows.append({"image_uid": uid, "observation_uid": uid, "label_id": label, "path": str(path)})
                obs = {"observation_uid": uid, "split": split, "label_id": label,
                       "latitude": 55. + label, "longitude": 10., "coordinate_uncertainty_m": 10., "geo__quality": "OK"}
                obs.update(substrate_class="soil" if label else "wood_dead", substrate__quality="OK")
                obs.update({bio: 100. + label for bio in BIO})
                observations.append(obs)
            frames[split] = pd.DataFrame(rows)
        tables = DatasetTables("fixture", "fixture_fingerprint", ["A", "B"], frames,
                               pd.DataFrame(columns=["split"]), root)
        def loader(config, show_progress, splits, **kwargs):
            assert splits == ("train", "val")  # No test images exist in this fixture.
            return tables
        def build(config, pretrained):
            return nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 2))
        torch.save({"classes": tables.classes, "dataset_fingerprint": tables.fingerprint,
                    "config": cfg, "model": build(cfg, False).state_dict()}, root / "initial.pt")
        config_path = root / "config.yaml"
        config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        monkeypatch.setattr(train_s8, "load_tables", loader)
        monkeypatch.setattr(train_s8, "load_context", lambda *a: (pd.DataFrame(observations), "fixture_context"))
        monkeypatch.setattr(train_s8, "build_model", build)
        monkeypatch.setattr(train_s8, "git_commit", lambda *a: {"commit": "fixture"})
        argv = ["train_s8", "--config", str(config_path), "--mode", "multimodal", "--device", "cpu"]
        monkeypatch.setattr(sys, "argv", argv + ["--pause-after-epoch"])
        assert train_s8.main() == 0
        run = next((root / "runs").iterdir())
        assert not (run / "completed.json").exists()
        monkeypatch.setattr(sys, "argv", argv + ["--resume", str(run / "last.pt")])
        assert train_s8.main() == 0
        completion = json.loads((run / "completed.json").read_text())
        assert completion["evaluated_splits"] == ["val"] and completion["test_evaluations"] == 0
        for suffix in ("val", "val_missing_all", "val_missing_geo", "val_shuffled_geo", "val_imprecise_geo",
                       "val_missing_climate", "val_shuffled_climate"):
            assert (run / f"predictions_{suffix}.csv").is_file()
        assert not list(run.glob("*test*"))
        state = torch.load(run / "best.pt", weights_only=True)
        assert state["encoder"]["feature_set"] == cfg["context"]["feature_set"]
        if stage == "s10":
            assert (run / "predictions_val_missing_substrate.csv").is_file()
            assert (run / "predictions_val_shuffled_substrate.csv").is_file()
        assert state["config"]["context_manifest_sha256"] == "fixture_context"
