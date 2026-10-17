"""S6 tests: taxonomic metrics, reproducibility of reports/s6 from saved predictions, and the modular pipeline."""
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "train"))
import taxonomic_metrics as report  # noqa: E402
from metrics import aggregate_observations, classification_metrics  # noqa: E402
from pipeline import taxonomy as T  # noqa: E402
from pipeline.predictions import load_predictions  # noqa: E402

TAX = pd.DataFrame({"label_id": [0, 1, 2, 3], "genus": ["A", "B", "C", "D"], "family": ["F1", "F1", "F2", "F3"],
                    "order": ["O1", "O1", "O1", "O2"], "class": ["C1", "C1", "C1", "C2"], "phylum": ["P1", "P1", "P1", "P2"]})
RUNS = sorted(p for p in (REPO / "runs").glob("*/completed.json")) if (REPO / "runs").is_dir() else []


def random_probs(rng, n, k):
    p = rng.random((n, k)) ** 4
    return p / p.sum(axis=1, keepdims=True)


class TaxonomicMetricTests(unittest.TestCase):
    def test_genus_level_is_identical_to_the_training_metrics(self):
        rng = np.random.default_rng(0)
        y, p = rng.integers(0, 4, 300), random_probs(rng, 300, 4)
        mine = T.level_metrics(y, p, TAX, "genus", [1, 3])
        ref = classification_metrics(y, p, TAX["genus"].tolist(), [1, 3])
        for k in ("macro_f1", "accuracy", "balanced_accuracy", "confusion_matrix"):
            self.assertEqual(mine[k], ref[k], k)

    def test_top1_rule_never_loses_accuracy_when_moving_up(self):
        rng = np.random.default_rng(1)
        y, p = rng.integers(0, 4, 500), random_probs(rng, 500, 4)
        acc = [T.level_metrics(y, p, TAX, level, [1])["accuracy"] for level in T.LEVELS]
        self.assertEqual(acc, sorted(acc))
        # a wrong genus inside the right family counts as correct at family level
        m = T.level_metrics(np.array([0]), np.array([[0.1, 0.8, 0.05, 0.05]]), TAX, "family", [1])
        self.assertEqual(m["accuracy"], 1.0)
        self.assertEqual(m["genera_per_taxon"]["F1"], ["A", "B"])

    def test_marginal_rule_sums_probabilities_and_reports_disagreements(self):
        p = np.array([[0.3, 0.3, 0.4, 0.0]])          # top-1 genus C (F2) but family F1 has 0.6
        np.testing.assert_allclose(T.marginalize(p, T.level_index(TAX, "family")[1], 3), [[0.6, 0.4, 0.0]])
        m = T.level_metrics(np.array([0]), p, TAX, "family", [1])
        self.assertEqual(m["accuracy"], 0.0)
        self.assertEqual(m["marginal"]["accuracy"], 1.0)
        self.assertEqual(m["marginal"]["disagreements_with_top1_genus"], 1)

    def test_fast_macro_f1_matches_sklearn_including_absent_labels(self):
        rng = np.random.default_rng(2)
        for _ in range(20):
            y, pred = rng.integers(0, 5, 60), rng.integers(0, 6, 60)
            self.assertAlmostEqual(T.macro_f1(y, pred, 7),
                                   f1_score(y, pred, labels=list(range(7)), average="macro", zero_division=0), places=12)

    def test_bootstrap_is_deterministic_and_brackets_the_estimate(self):
        rng = np.random.default_rng(3)
        y = rng.integers(0, 3, 400)
        pred = np.where(rng.random(400) < 0.8, y, rng.integers(0, 3, 400))
        a, b = T.bootstrap_macro_f1(y, pred, 3, 300, 42), T.bootstrap_macro_f1(y, pred, 3, 300, 42)
        self.assertEqual(a, b)
        self.assertLess(a[0], T.macro_f1(y, pred, 3))
        self.assertGreater(a[1], T.macro_f1(y, pred, 3))

    def test_observation_report_and_species_breakdown(self):
        pred = pd.DataFrame({"image_uid": ["i1", "i2", "i3", "i4"], "observation_uid": ["o1", "o1", "o2", "o3"],
                             "label_id": [0, 0, 3, 3]})
        probs = np.array([[0.4, 0.5, 0.05, 0.05], [0.9, 0.05, 0.03, 0.02], [0, 0, 0.1, 0.9], [0.7, 0.1, 0.1, 0.1]])
        r = T.taxonomic_report(pred, probs, TAX, [1], resamples=50)
        self.assertEqual(r["observation_level"]["genus"]["n"], 3)                  # o1 averaged -> A correct
        self.assertAlmostEqual(r["observation_level"]["genus"]["accuracy"], 2 / 3)
        self.assertEqual(r["image_level"]["genus"]["n"], 4)
        species = pd.DataFrame({"observation_uid": ["o1", "o2", "o3"], "species": ["A x", "D y", "D z"]})
        sb = T.species_breakdown(pred, probs, species, TAX["genus"].tolist()).set_index("species")
        self.assertEqual(sb.loc["D z", "most_common_wrong_genus"], "A")
        self.assertEqual(sb.loc["A x", "genus_recall"], 1.0)


@unittest.skipUnless((REPO / "data/taxonomy/v002/manifest.json").is_file() and RUNS, "S6 artifacts not available")
class SavedArtifactRegressionTests(unittest.TestCase):
    def test_taxonomy_covers_the_classes_of_v002(self):
        classes = pd.read_csv(REPO / "data/versions/v002/classes.csv").sort_values("label_id")["genus"].tolist()
        tax = T.load_taxonomy(REPO / "data/taxonomy/v002/genus_taxonomy.csv", classes)
        split_counts = pd.read_csv(REPO / "data/versions/v002/splits.csv").groupby("split").size()
        for s in ("train", "val", "test"):
            self.assertEqual(tax[f"n_obs_{s}"].sum(), split_counts[s])
        species = pd.read_csv(REPO / "data/taxonomy/v002/observation_species.csv")
        self.assertEqual(len(species), split_counts.sum())
        self.assertTrue(species["species"].str.split().str[0].eq(species["genus"]).all())

    def test_saved_predictions_reproduce_saved_genus_metrics(self):
        classes = pd.read_csv(REPO / "data/versions/v002/classes.csv").sort_values("label_id")["genus"].tolist()
        for ev in report.discover(REPO / "runs"):
            pred, probs = load_predictions(ev["predictions"], classes)
            saved = json.loads(ev["metrics"].read_text(encoding="utf-8"))
            check = report.reproduce_genus(pred, probs, classes, saved, ev["info"]["config"]["evaluation"]["top_k"])
            for unit, c in check.items():
                self.assertTrue(c["confusion_identical"], (ev["run_id"], ev["split"], unit))
                self.assertLess(c["max_abs_diff"], 1e-9)

    def test_reports_s6_is_reproduced_byte_for_byte(self):
        files = report.serialize(*report.compute("v002", 1000))
        for name, payload in files.items():
            self.assertEqual(report.text_sha256(REPO / "reports/s6" / name), __import__("hashlib").sha256(payload).hexdigest(), name)


class ModularPipelineTests(unittest.TestCase):
    def test_provenance_hashes_every_pipeline_module(self):
        from pipeline import artifacts
        listed = set(artifacts.CODE_FILES)
        for path in (REPO / "scripts/train/pipeline").glob("*.py"):
            if path.name not in ("taxonomy.py", "predictions.py"):      # used only after training
                self.assertIn(f"pipeline/{path.name}", listed)
        self.assertEqual(set(artifacts.code_hashes()), listed)

    def test_train_ma_keeps_its_public_import_surface(self):
        import train_ma
        for name in ("ImageTable", "build_model", "build_transforms", "predict", "evaluate", "save_predictions"):
            self.assertTrue(callable(getattr(train_ma, name)), name)

    @unittest.skipUnless(RUNS, "no saved runs")
    def test_saved_checkpoints_load_and_resume_with_the_refactored_code(self):
        import torch
        from pipeline.models import build_model
        from pipeline.training import check_resumable
        for done in RUNS:
            run = done.parent
            if (run / "best.pt").stat().st_size < 1_000_000:
                self.skipTest("checkpoints are Git LFS pointers; run `git lfs pull`")
            info = json.loads((run / "run.json").read_text(encoding="utf-8"))
            best = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
            build_model(info["config"], pretrained=False).load_state_dict(best["model"], strict=True)
            last = torch.load(run / "last.pt", map_location="cpu", weights_only=True)
            check_resumable(last, best["classes"], best["dataset_fingerprint"], last["config"], last["device_type"])
            self.assertEqual(last["config"], best["config"])


if __name__ == "__main__":
    unittest.main()
