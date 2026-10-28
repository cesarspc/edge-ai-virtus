"""Seed variability is descriptive and cannot silently replace the preselected candidate."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/train"))
from analyze_control_stability import summarize, fixed_candidate, checked_observations
from metrics import classification_metrics


class StabilityAnalysisTests(unittest.TestCase):
    def test_sample_sd_and_range(self):
        s = summarize([.97, .98, .99])
        self.assertAlmostEqual(s["mean"], .98)
        self.assertAlmostEqual(s["sample_sd"], .01)
        self.assertAlmostEqual(s["range"], .02)
        with self.assertRaises(ValueError): summarize([.97, .98])
        with self.assertRaises(ValueError): summarize([.97, .98, np.nan])

    def test_higher_scoring_repeat_cannot_replace_seed42(self):
        rows = [{"seed": 42, "score": .97}, {"seed": 43, "score": .99}, {"seed": 44, "score": .98}]
        self.assertEqual(fixed_candidate(rows)["seed"], 42)
        with self.assertRaises(ValueError): fixed_candidate(rows+[rows[1]])

    def test_observation_aggregation_uses_original_photo_identity(self):
        p = np.array([[.9,.1],[.2,.8],[.8,.2],[.3,.7]])
        expected = pd.DataFrame(dict(image_uid=["i1","i2","i3","i4"], observation_uid=["o1","o1","o1","o2"], label_id=[0,0,0,1]))
        frame = expected.assign(pred_label_id=p.argmax(1), p_A=p[:,0], p_B=p[:,1])
        obs_p = np.array([[(.9+.2+.8)/3, (.1+.8+.2)/3], [.3,.7]])
        saved = dict(image_level=classification_metrics(np.array([0,0,0,1]),p,["A","B"],[1]),
                     observation_level=classification_metrics(np.array([0,1]),obs_p,["A","B"],[1]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"pred.csv"
            frame.iloc[::-1].to_csv(path,index=False)
            _, y, got, metrics = checked_observations(path,["A","B"],expected,saved)
            np.testing.assert_allclose(got,obs_p)
            np.testing.assert_array_equal(y,[0,1])
            self.assertEqual(metrics["observation_level"]["n"],2)
            expected.loc[0,"label_id"]=1
            with self.assertRaises(AssertionError): checked_observations(path,["A","B"],expected,saved)


if __name__ == "__main__": unittest.main()
