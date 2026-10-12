"""Regression tests for partition identity and image leakage controls (offline)."""
import hashlib
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "data"))
import build_dataset as builder
import image_curation as images
import splits
from cleaning import CleaningLog


class BaselineDataTests(unittest.TestCase):
    def test_official_assignment_keeps_original_partition_even_when_site_day_overlaps(self):
        obs = pd.DataFrame({"source_id": ["fungitastic"] * 3, "observation_uid": ["a", "b", "c"],
                            "event_date_precision": ["day"] * 3, "event_date": ["2020-01-01"] * 3,
                            "geo__quality": ["OK"] * 3, "latitude": [4.5] * 3, "longitude": [-74.0] * 3,
                            "split_temporal": ["train", "val", "test"]})
        result = splits.assign(obs, {"splits": {"scheme": "fungitastic_official", "group_coordinate_decimals": 2}})
        self.assertEqual(result["split"].tolist(), ["train", "val", "test"])
        checks = splits.leakage_checks(result, pd.DataFrame({"image_uid": ["ia", "ib", "ic"],
                                                            "observation_uid": ["a", "b", "c"],
                                                            "split": ["train", "val", "test"]}))
        self.assertEqual(checks["groups_in_more_than_one_split"], 1)
        self.assertEqual(checks["observations_with_images_in_more_than_one_split"], 0)

    def test_single_source_needs_no_external_inputs_or_auxiliary_columns(self):
        cfg = {"sources": {"fungitastic": {}}, "image_policy": {}}
        self.assertEqual([p.name for p in builder.input_files(cfg)], ["metadata.zip"])
        obs = pd.DataFrame({"source_id": ["fungitastic"]})
        img = pd.DataFrame({"image_uid": ["a"]})
        retained, retained_images = builder.cross_source_dedup(obs, img, CleaningLog())
        self.assertEqual(len(retained), 1)
        self.assertEqual(len(retained_images), 1)

    def test_hash_curation_protects_test_and_rejects_missing_or_changed_bytes(self):
        temporary_root = Path(__file__).resolve().parents[1] / "tmp"
        temporary_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as directory:
            root = Path(directory)
            assert root.resolve().is_relative_to(temporary_root.resolve())
            (root / "300p").mkdir()
            rows, downloaded = [], []
            for i, (partition, payload) in enumerate([( "train", b"same"), ("val", b"same"),
                                                      ("test", b"same"), ("train", b"unique")]):
                filename = f"{i}.jpg"
                (root / "300p" / filename).write_bytes(payload)
                rows.append({"image_uid": str(i), "observation_uid": str(i), "split": partition,
                             "genus": "Amanita", "source_image_id": filename})
                downloaded.append({"image_uid": str(i), "filename": filename, "status": "ok",
                                   "sha256": hashlib.sha256(payload).hexdigest(), "phash64": f"{i:016x}",
                                   "width": 300, "height": 225})
            frame = pd.DataFrame(rows)
            obs = frame[["observation_uid"]].assign(n_images=1)
            policy = {"download_manifest": "download_manifest.csv", "fungitastic_resolution": "300p",
                      "min_side_px": 128, "near_duplicate_hamming": 4, "small_images": "report",
                      "cross_split_exact_duplicates": "protect_test"}
            downloaded[2].update(status="too_small", height=117)
            pd.DataFrame(downloaded).to_csv(root / "download_manifest.csv", index=False)
            with patch.object(images, "REPO_ROOT", root):
                kept_obs, kept, excluded, near, integrity, small = images.curate(obs, frame, policy)
                self.assertEqual(set(kept["image_uid"]), {"2", "3"})
                self.assertEqual(set(excluded["split"]), {"train", "val"})
                self.assertEqual(set(excluded["protected_split"]), {"test"})
                self.assertEqual(len(kept_obs), 2)
                self.assertEqual(set(integrity["image_uid"]), {"2", "3"})
                self.assertEqual(small["image_uid"].tolist(), ["2"])
                self.assertEqual(len(near), 1)  # similar hash is reported, not removed
                (root / "300p" / "3.jpg").write_bytes(b"modified")
                with self.assertRaisesRegex(SystemExit, "bytes differ"):
                    images.curate(obs, frame, policy)
                pd.DataFrame(downloaded[:-1]).to_csv(root / "download_manifest.csv", index=False)
                with self.assertRaisesRegex(SystemExit, "images missing"):
                    images.curate(obs, frame, policy)

    def test_perceptual_search_matches_exhaustive_hamming_search(self):
        rng = random.Random(42)
        values = [rng.getrandbits(64) for _ in range(25)]
        values += [values[i] ^ ((1 << (i % 64)) | (1 << ((i + 3) % 64))) for i in range(25)]
        frame = pd.DataFrame({"image_uid": [f"{i:03}" for i in range(50)],
                              "phash64": [f"{v:016x}" for v in values], "sha256": [str(i) for i in range(50)],
                              "split": ["train"] * 25 + ["test"] * 25, "genus": ["Amanita"] * 50})
        expected = {(f"{i:03}", f"{j:03}") for i in range(25) for j in range(25, 50)
                    if (values[i] ^ values[j]).bit_count() <= 4}
        result = images.near_pairs(frame, 4)
        self.assertEqual(set(zip(result["image_uid_a"], result["image_uid_b"])), expected)


if __name__ == "__main__":
    unittest.main()
