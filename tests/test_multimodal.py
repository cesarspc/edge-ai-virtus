import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "train"))
from pipeline.multimodal import (EnvironmentalEncoder, EnvironmentalOnlyModel, MultimodalTable,
                                 VisualEnvironmentalModel, VisualOnlyModel)
from train_s8 import condition_frame, join_observations


def frame():
    return pd.DataFrame({"split": ["train", "train"], "habitat_class": ["forest", "open"],
                         "substrate_class": ["soil", "wood"], "elevation_m": [100, 300],
                         "elevation__quality": ["OK", "OK"],
                         "event_date": ["2020-01-01", "2020-07-01"],
                         "event_date__quality": ["OK", "OK"]})


def test_encoder_fit_train_only_and_roundtrip():
    train = frame()
    encoder = EnvironmentalEncoder.fit(train)
    restored = EnvironmentalEncoder.from_state_dict(encoder.state_dict())
    np.testing.assert_array_equal(encoder.transform(train), restored.transform(train))
    with pytest.raises(ValueError):
        EnvironmentalEncoder.fit(train.assign(split="val"))


def test_unknown_and_missing_are_explicit():
    train = frame()
    encoder = EnvironmentalEncoder.fit(train)
    val = train.iloc[:1].copy()
    val.loc[:, "habitat_class"] = "unseen"
    val.loc[:, "elevation__quality"] = "SUSPECT"
    val.loc[:, "event_date"] = None
    result = encoder.transform(val)[0]
    assert result[len(encoder.categories["habitat_class"])] == 1
    assert result[-5] == 0 and result[-4] == 1
    assert result[-3:].tolist() == [0, 0, 1]


def test_initial_fusion_preserves_visual_predictions():
    visual = nn.Linear(4, 3)
    model = VisualEnvironmentalModel(visual, 5, 3)
    image, metadata = torch.randn(2, 4), torch.randn(2, 5)
    torch.testing.assert_close(model(image, metadata), visual(image))


def test_join_checks_labels_splits_and_preserves_order():
    obs = frame().assign(observation_uid=["a", "b"], label_id=[0, 1])
    images = pd.DataFrame({"observation_uid": ["b", "a", "a"], "label_id": [1, 0, 0],
                           "path": ["b.jpg", "a.jpg", "a2.jpg"]})
    joined = join_observations({"train": images}, obs)["train"]
    assert joined.observation_uid.tolist() == ["b", "a", "a"]
    with pytest.raises(ValueError):
        join_observations({"val": images}, obs)
    with pytest.raises(ValueError):
        join_observations({"train": images.assign(label_id=2)}, obs)


def test_perturbations_are_by_observation_and_keep_order():
    obs = frame().assign(observation_uid=["a", "b"], label_id=[0, 1])
    images = pd.DataFrame({"observation_uid": ["b", "a", "a"], "label_id": [1, 0, 0],
                           "path": ["b.jpg", "a.jpg", "a2.jpg"]})
    joined = join_observations({"train": images}, obs)["train"]
    shuffled = condition_frame(joined, "shuffled", 42)
    assert shuffled.observation_uid.tolist() == joined.observation_uid.tolist()
    assert shuffled.loc[1, "habitat_class"] == shuffled.loc[2, "habitat_class"]
    missing = condition_frame(joined, "missing", 42)
    assert missing.habitat_class.isna().all()
    assert missing.elevation__quality.eq("MISSING").all()
    assert np.isfinite(EnvironmentalEncoder.fit(obs).transform(missing)).all()


def test_modes_and_metadata_only_dataset():
    obs = frame()
    encoded = EnvironmentalEncoder.fit(obs).transform(obs)
    dataset = MultimodalTable(pd.DataFrame({"path": ["absent", "absent"], "label_id": [0, 1]}),
                              encoded, None, load_images=False)
    image, metadata, label = dataset[0]
    assert image.shape == (1,) and label == 0
    model = EnvironmentalOnlyModel(len(metadata), 3)
    assert model(image.unsqueeze(0), metadata.unsqueeze(0)).shape == (1, 3)
    visual = nn.Linear(4, 3)
    wrapper = VisualOnlyModel(visual)
    x = torch.randn(1, 4)
    torch.testing.assert_close(wrapper(x, metadata.unsqueeze(0)), visual(x))
