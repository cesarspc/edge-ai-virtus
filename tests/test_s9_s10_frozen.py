import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/train"))
from pipeline.context import BIO, ContextEncoder, ContextFusion, broad_substrate, dropout_context, perturb_context
from run_s9_s10_frozen import aligned_visual, context_available, mix_scores, paired_delta


def context():
    result = pd.DataFrame({"observation_uid": ["a", "b", "c"], "split": "train", "latitude": 55.,
        "longitude": 10., "coordinate_uncertainty_m": 10., "geo__quality": "OK",
        "substrate_class": ["wood_dead", "wood_unspecified", "soil"],
        "substrate__quality": ["OK", "LOW_PRECISION", "OK"]})
    for bio in BIO:
        result[bio] = [100., 200., 300.]
    return result


def test_d_uses_same_coarse_wood_for_different_quality_and_train_vocabulary():
    frame = context()
    assert broad_substrate(frame).tolist() == ["wood", "wood", "soil"]
    encoder = ContextEncoder.fit(frame, "geo_climate_substrate")
    assert encoder.substrate_categories == ["soil", "wood"]
    encoded = encoder.transform(frame)
    np.testing.assert_array_equal(encoded[0, 42:], encoded[1, 42:])
    unknown = frame.iloc[:1].assign(substrate_class="unseen", split="val")
    assert encoder.transform(unknown)[0, -1] == 1
    with pytest.raises(ValueError):
        ContextEncoder.fit(unknown, "geo_climate_substrate")
    restored = ContextEncoder.from_state_dict(encoder.state_dict())
    np.testing.assert_array_equal(encoded, restored.transform(frame))
    assert context_available(frame, encoder).all()


def test_d_missing_all_falls_back_to_visual_and_dropout_has_valid_blocks():
    frame = context()
    encoder = ContextEncoder.fit(frame, "geo_climate_substrate")
    missing = torch.from_numpy(encoder.transform(perturb_context(frame, "missing_all", 42, encoder.feature_set)))
    model = ContextFusion(nn.Linear(2, 3), missing.shape[1], 3)
    with torch.no_grad():
        model.metadata_head.bias.fill_(9)
    image = torch.randn(3, 2)
    torch.testing.assert_close(model(image, missing), model.visual_model(image))
    assert not missing[:, 42:-1].any() and missing[:, -1].eq(1).all()
    assert torch.isfinite(dropout_context(missing, .2)).all()


def test_fusion_weight_and_absence_have_exact_expected_behavior():
    visual = np.array([[.9, .1], [.2, .8]])
    tabular = np.array([[.1, .9], [.8, .2]])
    actual = mix_scores(visual, tabular, [False, True], .1)
    np.testing.assert_array_equal(actual[0], visual[0])
    np.testing.assert_allclose(actual[1], [.26, .74])
    np.testing.assert_allclose(actual.sum(1), 1)


def test_paired_bootstrap_is_zero_for_identical_predictions():
    result = paired_delta(np.array([0, 1, 1, 0]), np.array([0, 0, 1, 0]), np.array([0, 0, 1, 0]), 25, 42)
    assert result["delta_macro_f1"] == 0
    assert result["ci95_delta_macro_f1"] == [0, 0]
    assert result["introduced_errors"] == result["fixed_errors"] == 0


def test_saved_visual_cohort_is_checked_before_reuse(tmp_path):
    predictions = pd.DataFrame({"image_uid": ["x", "y"], "observation_uid": ["a", "b"],
        "label_id": [0, 1], "p_A": [.9, .2], "p_B": [.1, .8]})
    path = tmp_path / "predictions.csv"
    predictions.to_csv(path, index=False)
    obs = pd.DataFrame({"observation_uid": ["b", "a"], "label_id": [1, 0]})
    np.testing.assert_allclose(aligned_visual(path, obs, ["A", "B"]), [[.2, .8], [.9, .1]])
    with pytest.raises(ValueError):
        aligned_visual(path, obs.assign(label_id=0), ["A", "B"])
