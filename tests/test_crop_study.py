"""Crop geometry, photo-level aggregation and durable paired-analysis checkpoints."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/train"))
from pipeline.crops import crop_box, crop_image, POLICIES
from pipeline.crop_state import atomic_npz, load_blocks, signature
from analyze_crop_experiments import paired_bootstrap
from data import image_hashes


class CropStudyTests(unittest.TestCase):
    def test_parallel_integrity_reads_fresh_bytes_in_the_same_order(self):
        import hashlib
        with tempfile.TemporaryDirectory() as d:
            paths = [Path(d) / str(i) for i in range(20)]
            for i, path in enumerate(paths): path.write_bytes(bytes([i])*100)
            expected = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
            self.assertEqual(image_hashes(paths, 1), expected)
            self.assertEqual(image_hashes(paths, 8), expected)
            paths[3].write_bytes(b"modified")
            self.assertNotEqual(image_hashes(paths, 8)[3], expected[3])
            with self.assertRaises(ValueError): image_hashes(paths, 0)

    def test_geometry_preserves_aspect_and_bounds(self):
        for size in [(400, 300), (1, 1), (301, 900)]:
            for fraction in [.5, .75, 1]:
                for anchor in ["center", "top_left", "top_right", "bottom_left", "bottom_right"]:
                    x, y, r, b = crop_box(size, fraction, anchor)
                    self.assertTrue(0 <= x < r <= size[0] and 0 <= y < b <= size[1])
                    self.assertEqual(r-x, max(1, round(size[0]*np.sqrt(fraction))))
                    self.assertEqual(b-y, max(1, round(size[1]*np.sqrt(fraction))))
        self.assertEqual(crop_box((400, 300), 1), (0, 0, 400, 300))
        with self.assertRaises(ValueError): crop_box((400, 300), 0)
        with self.assertRaises(ValueError): crop_box((400, 300), .5, "invalid")

    def test_control_is_identity_and_multiview_has_one_score_per_photo(self):
        from PIL import Image
        image = Image.new("RGB", (400, 300))
        self.assertIs(crop_image(image, 1), image)
        probs = np.arange(2*7*3).reshape(2, 7, 3)
        p = probs[:, POLICIES["multi5"]].mean(1)
        self.assertEqual(p.shape, (2, 3))
        np.testing.assert_array_equal(p[0], probs[0, 2:7].mean(0))

    def test_resume_rejects_changed_input_or_missing_block(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            p = np.full((4, 7, 3), 1/3, dtype=np.float32)
            args = dict(signature="abc", start=0, end=4, probs=p, seconds=np.ones(7), peak_bytes=np.ones(7))
            atomic_npz(folder / "block_000000.npz", **args)
            out, times, peaks, done = load_blocks(folder, "abc", 10, 4, 7, 3)
            self.assertEqual(done, 4)
            np.testing.assert_array_equal(out[0], p)
            with self.assertRaises(ValueError): load_blocks(folder, "changed", 10, 4, 7, 3)
            atomic_npz(folder / "block_000008.npz", **{**args, "start": 8, "end": 10, "probs": p[:2]})
            with self.assertRaises(ValueError): load_blocks(folder, "abc", 10, 4, 7, 3)

    def test_bootstrap_resume_is_exact_and_matches_sklearn(self):
        rng = np.random.default_rng(3)
        y = rng.integers(0, 3, 30)
        pred = np.where(rng.random(30)<.8, y, rng.integers(0, 3, 30))
        jobs = {"control": y, "crop": pred}
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "bootstrap.npz"
            part = paired_bootstrap(y, jobs, 4, 80, 42, path, "same", stop_after=25)
            self.assertEqual(len(part["crop"]), 25)
            result = paired_bootstrap(y, jobs, 4, 80, 42, path, "same")
            idx = np.random.default_rng(42).integers(0, len(y), size=(80, len(y)))
            expected = [f1_score(y[i], pred[i], labels=list(range(4)), average="macro", zero_division=0) for i in idx]
            np.testing.assert_allclose(result["crop"], expected, atol=1e-14)
            with self.assertRaises(ValueError): paired_bootstrap(y, jobs, 4, 80, 42, path, "different")


if __name__ == "__main__":
    unittest.main()
