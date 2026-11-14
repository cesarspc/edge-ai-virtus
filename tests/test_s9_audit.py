import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/reports"))
from audit_s9_validation import observation_errors


def predictions():
    return pd.DataFrame({"image_uid": ["b1", "a1", "a2"], "observation_uid": ["b", "a", "a"],
                         "label_id": [1, 0, 0], "p_A": [.1, .9, .2], "p_B": [.9, .1, .8]})


def test_error_audit_aggregates_before_scoring_and_detects_view_disagreement():
    errors = observation_errors(predictions(), ["A", "B"]).set_index("observation_uid")
    assert errors.correct.all()
    assert errors.loc["a", "n_images"] == 2
    assert errors.loc["a", "distinct_image_predictions"] == 2
    assert errors.loc["a", "disagreement_rate"] == .5
    assert errors.loc["a", "view_js_divergence"] > 0
    assert errors.loc["b", "view_js_divergence"] == pytest.approx(0)
    assert errors.loc["a", "confidence"] == pytest.approx(.55)


def test_audit_rejects_duplicated_images_and_inconsistent_labels():
    frame = predictions()
    with pytest.raises(ValueError):
        observation_errors(pd.concat([frame, frame.iloc[:1]]), ["A", "B"])
    frame.loc[2, "label_id"] = 1
    with pytest.raises(ValueError):
        observation_errors(frame, ["A", "B"])


def test_audit_rejects_non_probabilities():
    frame = predictions()
    frame.loc[0, "p_A"] = np.nan
    with pytest.raises(ValueError):
        observation_errors(frame, ["A", "B"])
