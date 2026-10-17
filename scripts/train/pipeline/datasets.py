"""PyTorch view of a split table produced by ``data.load_tables`` (columns ``path`` and ``label_id``)."""
from __future__ import annotations

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class ImageTable(Dataset):
    def __init__(self, frame: pd.DataFrame, transform):
        self.paths = frame["path"].tolist()
        self.labels = frame["label_id"].astype(int).tolist()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        with Image.open(self.paths[i]) as im:
            x = self.transform(im.convert("RGB"))
        return x, self.labels[i]
