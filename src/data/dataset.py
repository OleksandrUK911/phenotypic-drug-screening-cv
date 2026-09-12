from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.preprocessing import preprocess_well_image
from src.data.schema import ImageRecord, WellMetadata


class CellPaintingDataset(Dataset):
    """Multi-channel (5-stain) Cell Painting field-of-view dataset.

    Each item is one field of view, preprocessed to a (C, H, W) float tensor.
    Metadata (compound, MOA, control flag) is looked up by (plate, well) so
    downstream code can group per-cell/per-well without re-parsing filenames.
    """

    def __init__(
        self,
        records: list[ImageRecord],
        metadata: dict[tuple[str, str], WellMetadata],
        illumination_sigma: float = 100.0,
        transform=None,
    ) -> None:
        self.records = records
        self.metadata = metadata
        self.illumination_sigma = illumination_sigma
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]
        image = preprocess_well_image(record.channel_paths, self.illumination_sigma)

        well_meta = self.metadata.get((record.plate, record.well))
        if well_meta is None:
            raise KeyError(f"No metadata for plate={record.plate} well={record.well}")

        if self.transform is not None:
            image = self.transform(image=np.moveaxis(image, 0, -1))["image"]
            image = np.moveaxis(image, -1, 0)

        return {
            "image": torch.from_numpy(image.copy()).float(),
            "plate": record.plate,
            "well": record.well,
            "site": record.site,
            "compound": well_meta.compound,
            "moa": well_meta.moa or "unknown",
            "is_control": well_meta.is_control,
        }
