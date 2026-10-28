"""Deterministic extra crops. Fractions refer to original area before normal transforms."""
from __future__ import annotations

import math
import random

ANCHORS = ("center", "top_left", "top_right", "bottom_left", "bottom_right")
VIEWS = (("control", 1.0, "center"), ("center75", .75, "center"),
         ("center50", .5, "center"), ("top_left50", .5, "top_left"),
         ("top_right50", .5, "top_right"), ("bottom_left50", .5, "bottom_left"),
         ("bottom_right50", .5, "bottom_right"))
POLICIES = {"control": [0], "center75": [1], "center50": [2], "multi5": [2, 3, 4, 5, 6]}


def crop_box(size, area_fraction, anchor="center"):
    w, h = size
    if w < 1 or h < 1 or not 0 < area_fraction <= 1 or anchor not in ANCHORS:
        raise ValueError("positive image dimensions, area in (0,1] and a known anchor required")
    cw = max(1, min(w, round(w * math.sqrt(area_fraction))))
    ch = max(1, min(h, round(h * math.sqrt(area_fraction))))
    x = (w - cw) // 2 if anchor == "center" else (w - cw if anchor.endswith("right") else 0)
    y = (h - ch) // 2 if anchor == "center" else (h - ch if anchor.startswith("bottom") else 0)
    return x, y, x + cw, y + ch


def crop_image(image, fraction, anchor="center"):
    return image if fraction == 1 else image.crop(crop_box(image.size, fraction, anchor))


class ExtraCrop:
    """One view per training photo; random anchor only when explicitly configured."""
    def __init__(self, fraction, anchors=("center",)):
        self.fraction, self.anchors = fraction, tuple(anchors)
        for anchor in self.anchors:
            crop_box((100, 100), fraction, anchor)
        if not self.anchors:
            raise ValueError("at least one anchor required")

    def __call__(self, image):
        anchor = self.anchors[0] if len(self.anchors) == 1 else random.choice(self.anchors)
        return crop_image(image, self.fraction, anchor)


class CropViews:
    """Stack views of one original photo; datasets keep the original photo identity."""
    def __init__(self, transform, views=VIEWS):
        self.transform, self.views = transform, tuple(views)

    def __call__(self, image):
        import torch
        return torch.stack([self.transform(crop_image(image, q, anchor)) for _, q, anchor in self.views])


class CropTable:
    """Decode each original once and return seven inputs plus audited crop coordinates."""
    def __init__(self, frame, transform):
        self.paths, self.labels = frame.path.tolist(), frame.label_id.astype(int).tolist()
        self.transform = CropViews(transform)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        import numpy as np
        from PIL import Image
        with Image.open(self.paths[index]) as original:
            image = original.convert("RGB")
            w, h = image.size
            boxes = np.array([[w, h, *crop_box((w, h), q, anchor)] for _, q, anchor in VIEWS], dtype=np.int32)
            return self.transform(image), self.labels[index], boxes
