from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.features.simclr import SimCLRModel


@dataclass(frozen=True)
class CellEmbedding:
    plate: str
    well: str
    embedding: np.ndarray


@torch.no_grad()
def extract_cell_embeddings(model: SimCLRModel, loader: DataLoader, device: str = "cpu") -> list[CellEmbedding]:
    model.eval().to(device)
    results = []
    for batch in loader:
        images = batch["image"].to(device)
        embeddings = model.embed(images).cpu().numpy()
        for i in range(len(embeddings)):
            results.append(
                CellEmbedding(plate=batch["plate"][i], well=batch["well"][i], embedding=embeddings[i])
            )
    return results


def aggregate_to_well(cell_embeddings: list[CellEmbedding], method: str = "median") -> dict[tuple[str, str], np.ndarray]:
    """Pool per-cell embeddings to one vector per (plate, well).

    Median pooling (default) is more robust than mean to a handful of
    mis-segmented or dying cells dominating the well-level phenotype signature.
    """
    if method not in {"mean", "median"}:
        raise ValueError(f"Unknown aggregation method: {method}")

    grouped: dict[tuple[str, str], list[np.ndarray]] = {}
    for ce in cell_embeddings:
        grouped.setdefault((ce.plate, ce.well), []).append(ce.embedding)

    pool_fn = np.mean if method == "mean" else np.median
    return {key: pool_fn(np.stack(vecs), axis=0) for key, vecs in grouped.items()}
