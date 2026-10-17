"""Architecture factory: a torchvision backbone with its classification head replaced by ``num_classes`` outputs."""
from __future__ import annotations

import os
from typing import Any

import torchvision
from torch import nn

import data


def build_model(cfg: dict[str, Any], pretrained: bool = True) -> nn.Module:
    os.environ.setdefault("TORCH_HOME", str(data.REPO_ROOT / ".cache" / "torch"))
    m = cfg["model"]
    ctor = getattr(torchvision.models, m["architecture"])
    weights = m.get("pretrained_weights") if pretrained else None
    model = ctor(weights=weights) if weights else ctor(weights=None)
    n = m["num_classes"]
    # replace the classification head (supports the families listed in the project scope)
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):                       # ResNet
        model.fc = nn.Linear(model.fc.in_features, n)
    elif hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):  # MobileNet / EfficientNet
        last = [i for i, l in enumerate(model.classifier) if isinstance(l, nn.Linear)][-1]
        model.classifier[last] = nn.Linear(model.classifier[last].in_features, n)
    else:
        raise SystemExit(f"do not know how to replace the head of {m['architecture']}")
    return model
