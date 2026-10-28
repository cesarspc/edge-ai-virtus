"""Image transforms built only from the ``input`` section of a training config."""
from __future__ import annotations

from typing import Any

from torchvision import transforms


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
    return transforms.Compose(train), transforms.Compose(evaluation)
