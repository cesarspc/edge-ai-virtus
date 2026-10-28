"""Fixed seed recipe, nonselective job reuse, resume discovery and lock release."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/train"))
from run_control_stability import check_configs, job_to_run, campaign_lock


class StabilityRunnerTests(unittest.TestCase):
    def test_only_seed_and_name_may_change(self):
        base = yaml.safe_load((ROOT / "configs/train/cropping_mobilenet_v3_large_v002_control.yaml").read_text())
        configs = []
        for seed in [43, 44]:
            cfg = copy.deepcopy(base)
            cfg["training"]["seed"] = seed; cfg["run_name"] = str(seed); cfg["experiment"]["seed_pair"] = seed
            configs.append(cfg)
        check_configs(base, configs)
        configs[0]["training"]["lr"] *= 2
        with self.assertRaisesRegex(ValueError, "only run name"):
            check_configs(base, configs)

    def test_first_completed_is_reused_without_choosing_a_score(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = {"output": {"runs_dir": directory}}
            for i, score in [(1, .9), (2, .99)]:
                run = Path(directory) / f"{i:03d}"
                run.mkdir()
                (run / "run.json").write_text(json.dumps(dict(mode="train", config=cfg)))
                (run / "completed.json").write_text("{}")
                (run / "metrics_val.json").write_text(json.dumps(dict(score=score)))
            mode, path = job_to_run(cfg)
            self.assertEqual((mode, path.name), ("completed", "001"))

    def test_latest_saved_incomplete_run_is_resumable(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = {"output": {"runs_dir": directory}}
            self.assertEqual(job_to_run(cfg), ("new", None))
            for i in [1, 2]:
                run = Path(directory) / f"{i:03d}"; run.mkdir()
                (run / "run.json").write_text(json.dumps(dict(mode="train", config=cfg)))
                (run / "last.pt").write_bytes(b"fixture")
            mode, path = job_to_run(cfg)
            self.assertEqual((mode, path.name), ("resume", "002"))

    def test_lock_excludes_other_open_and_releases_after_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "campaign.lock"
            with self.assertRaisesRegex(RuntimeError, "interrupt"):
                with campaign_lock(path):
                    with self.assertRaises(SystemExit):
                        with campaign_lock(path): pass
                    raise RuntimeError("interrupt")
            with campaign_lock(path): pass


if __name__ == "__main__": unittest.main()
