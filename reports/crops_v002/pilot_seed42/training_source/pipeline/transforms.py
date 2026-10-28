"""Image transforms built only from the ``input`` section of a training config."""
from __future__ import annotations

from typing import Any

from torchvision import transforms
from pipeline.crops import CropViews, ExtraCrop


def build_transforms(cfg: dict[str, Any]):
    """Return ``(train, evaluation)`` transforms. Evaluation is deterministic: resize + center crop."""
    inp = cfg["input"]
    norm = transforms.Normalize(inp["mean"], inp["std"])
    aug = inp["train_augment"]
    train = [transforms.RandomResizedCrop(inp["size"], scale=tuple(aug["random_resized_crop_scale"]))]
    if aug.get("horizontal_flip"):
        train.append(transforms.RandomHorizontalFlip())
    if aug.get("color_jitter"):
        train.append(transforms.ColorJitter(*aug["color_jitter"]))
    train += [transforms.ToTensor(), norm]
    evaluation = [transforms.Resize(inp["eval_resize"]), transforms.CenterCrop(inp["size"]), transforms.ToTensor(), norm]
    crop = inp.get("extra_crop")
    if crop:
        fraction = float(crop["area_fraction"])
        train.insert(0, ExtraCrop(fraction, crop.get("train_anchors", ["center"])))
        anchors = crop.get("eval_anchors", ["center"])
        if len(anchors) > 1:
            evaluation = CropViews(transforms.Compose(evaluation),
                                   [(anchor, fraction, anchor) for anchor in anchors])
        else:
            evaluation = transforms.Compose([ExtraCrop(fraction, anchors), *evaluation])
    else:
        evaluation = transforms.Compose(evaluation)
    return transforms.Compose(train), evaluation
