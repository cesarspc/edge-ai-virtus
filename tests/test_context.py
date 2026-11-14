import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/train"))
from pipeline.context import BIO, ContextEncoder, ContextFusion, dropout_context, perturb_context


def frame():
    result = pd.DataFrame({"observation_uid": ["b", "a", "a"], "split": ["train"] * 3,
                           "latitude": [55., 56., 56.], "longitude": [10., 11., 11.],
                           "coordinate_uncertainty_m": [10., 30., 30.], "geo__quality": ["OK"] * 3})
    for i, field in enumerate(BIO):
        result[field] = [100. + i, 300. + i, 300. + i]
    return result


def test_incremental_whitelist_ignores_unapproved_fields():
    obs = frame()
    obs["substrate_class"], obs["event_date"], obs["elevation_m"] = "soil", "2020-01-01", 50
    encoder = ContextEncoder.fit(obs, "geo")
    original = encoder.transform(obs)
    changed = obs.assign(substrate_class="wood", event_date="2022-12-31", elevation_m=8000,
                         label_id=9, habitat_class="forest")
    changed[list(BIO)] = -999
    np.testing.assert_array_equal(original, encoder.transform(changed))
    assert original.shape == (3, 4)
    assert ContextEncoder.fit(obs, "geo_climate").transform(obs).shape == (3, 42)


def test_train_only_normalization_roundtrip_and_no_val_fit():
    obs = frame().drop_duplicates("observation_uid")
    encoder = ContextEncoder.fit(obs, "geo_climate")
    assert encoder.climate_mean[0] == 200
    val = obs.assign(split="val")
    val[list(BIO)] = 900
    restored = ContextEncoder.from_state_dict(encoder.state_dict())
    np.testing.assert_array_equal(encoder.transform(val), restored.transform(val))
    assert encoder.climate_mean[0] == 200
    with pytest.raises(ValueError):
        ContextEncoder.fit(val)


def test_missing_invalid_quality_and_unknown_climate_are_finite():
    obs = frame()
    encoder = ContextEncoder.fit(obs, "geo_climate")
    obs.loc[0, "latitude"] = 91
    obs.loc[1, "geo__quality"] = "LOW_PRECISION"
    obs.loc[2, "longitude"] = np.inf
    obs.loc[:, "Bio1"] = np.inf
    encoded = encoder.transform(obs)
    assert np.isfinite(encoded).all()
    assert encoded[:, 3].tolist() == [1, 1, 1]
    assert encoded[:, 23].tolist() == [1, 1, 1]
    assert not encoded[:, :3].any()


def test_perturbations_keep_observation_views_together_and_modality_separate():
    obs = frame()
    shuffled = perturb_context(obs, "shuffled_climate", 3, "geo_climate")
    assert shuffled.observation_uid.tolist() == obs.observation_uid.tolist()
    assert shuffled.loc[1, "Bio1"] == shuffled.loc[2, "Bio1"]
    pd.testing.assert_series_equal(shuffled.latitude, obs.latitude)
    missing = perturb_context(obs, "missing_geo", 3, "geo_climate")
    assert missing.latitude.isna().all()
    pd.testing.assert_series_equal(missing.Bio1, obs.Bio1)
    with pytest.raises(ValueError):
        perturb_context(obs, "missing_climate", 3, "geo")


def test_fusion_fallback_after_training_and_dropout():
    obs = frame()
    encoder = ContextEncoder.fit(obs, "geo_climate")
    missing = torch.from_numpy(encoder.transform(perturb_context(obs, "missing_all", 1, "geo_climate")))
    model = ContextFusion(nn.Linear(2, 3), 42, 3)
    with torch.no_grad():
        model.metadata_head.weight.fill_(2)
        model.metadata_head.bias.fill_(3)
    image = torch.randn(3, 2)
    torch.testing.assert_close(model(image, missing), model.visual_model(image))
    torch.manual_seed(1)
    encoded = torch.from_numpy(encoder.transform(obs))
    dropped = dropout_context(encoded, .99)
    assert not dropped[:, :3].any() and dropped[:, 3].eq(1).all()
    assert not dropped[:, 4:23].any() and dropped[:, 23:].eq(1).all()
    torch.testing.assert_close(encoded, torch.from_numpy(encoder.transform(obs)))
