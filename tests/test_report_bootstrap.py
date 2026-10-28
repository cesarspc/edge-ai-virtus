"""Numerical and resume checks for the report's paired observation bootstrap."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("s5_analysis",ROOT/"scripts/reports/analyze_s5_s6.py")
analysis=importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)
from pipeline.predictions import load_predictions


class ReportEvidenceTests(unittest.TestCase):
    def test_reject_nonfinite_export(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"pred.csv"
            pd.DataFrame({"image_uid":["i"],"observation_uid":["o"],"label_id":[0],"pred_label_id":[0],
                          "p_A":[np.nan],"p_B":[np.nan]}).to_csv(path,index=False)
            with self.assertRaisesRegex(ValueError,"non-finite"):
                load_predictions(path,["A","B"])

    def test_paired_scores_match_sklearn_and_resume_exactly(self):
        rng=np.random.default_rng(7)
        y=rng.integers(0,5,40)
        a=np.where(rng.random(40)<.8,y,rng.integers(0,5,40))
        b=np.where(rng.random(40)<.9,y,rng.integers(0,5,40))
        jobs={"A_val_genus":(y,a,10),"B_val_genus":(y,b,10)}
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"state.npz"
            got,_,_=analysis.bootstrap(jobs,300,42,path,{"toy":"fixed"})
            idx=np.random.default_rng(42).integers(0,len(y),size=(300,len(y)))
            reference={key:np.array([f1_score(y[i],p[i],labels=list(range(10)),average="macro",zero_division=0) for i in idx]) for key,p in [("A_val_genus",a),("B_val_genus",b)]}
            for key in reference:np.testing.assert_allclose(got[key],reference[key],atol=1e-14,rtol=0)
            np.testing.assert_allclose(got["B_val_genus"]-got["A_val_genus"],reference["B_val_genus"]-reference["A_val_genus"],atol=1e-14,rtol=0)
            with np.load(path,allow_pickle=False) as old:cache={key:old[key].copy() for key in old.files}
            partial_rng=np.random.default_rng(42)
            partial_rng.integers(0,len(y),size=(250,len(y)))
            cache.update(done=np.array(250),rng_state=np.array(json.dumps(partial_rng.bit_generator.state)))
            for key in ["scores","accuracies","per_genus"]:cache[key][250:]=0
            with path.open("wb") as stream:np.savez_compressed(stream,**cache)
            resumed,_,_=analysis.bootstrap(jobs,300,42,path,{"toy":"fixed"})
            for key in got:np.testing.assert_array_equal(got[key],resumed[key])


if __name__=="__main__":unittest.main()
