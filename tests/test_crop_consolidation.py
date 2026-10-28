"""Cached crop inference requires exact parameters, buffers, class order and recipe."""
from pathlib import Path
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/train"))
from consolidate_crop_runs import verify_identical_checkpoints


class CropConsolidationTests(unittest.TestCase):
    def fixture(self):
        return dict(model={"weight": torch.tensor([1., 2.]), "num_batches_tracked": torch.tensor(4)},
                    classes=["A", "B"], dataset_fingerprint="fixture", config={"input": {"size": 224}}, epoch=1)

    def test_reuse_allows_reserialization_of_identical_state(self):
        with tempfile.TemporaryDirectory() as folder:
            a, b = Path(folder) / "a.pt", Path(folder) / "b.pt"
            state = self.fixture()
            torch.save(state, a); torch.save(dict(reversed(list(state.items()))), b)
            result = verify_identical_checkpoints(a, b)
            self.assertTrue(result["tensors_identical"])
            self.assertEqual(result["tensors_checked"], 2)
            self.assertNotEqual(result["original_archive_sha256"], result["completed_archive_sha256"])

    def test_changed_parameter_or_buffer_rejects_reuse(self):
        with tempfile.TemporaryDirectory() as folder:
            a, b = Path(folder) / "a.pt", Path(folder) / "b.pt"
            torch.save(self.fixture(), a)
            for name in ["weight", "num_batches_tracked"]:
                state = self.fixture(); state["model"][name] += 1
                torch.save(state, b)
                with self.assertRaisesRegex(ValueError, "model state changed"):
                    verify_identical_checkpoints(a, b)

    def test_changed_class_order_or_recipe_rejects_reuse(self):
        with tempfile.TemporaryDirectory() as folder:
            a, b = Path(folder) / "a.pt", Path(folder) / "b.pt"
            torch.save(self.fixture(), a)
            for key, value in [("classes", ["B", "A"]), ("config", {"input": {"size": 256}}), ("epoch", 2)]:
                state = self.fixture(); state[key] = value; torch.save(state, b)
                with self.assertRaisesRegex(ValueError, "checkpoint metadata changed"):
                    verify_identical_checkpoints(a, b)


if __name__ == "__main__": unittest.main()
