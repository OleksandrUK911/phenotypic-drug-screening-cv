from __future__ import annotations

import random

from src.data.schema import WellMetadata


def plate_aware_split(
    wells: list[WellMetadata],
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 0,
) -> tuple[list[WellMetadata], list[WellMetadata], list[WellMetadata]]:
    """Split wells into train/val/test by PLATE, never by individual well/image.

    Splitting by image would leak batch effects: images from the same plate
    share technical artifacts (illumination, staining day, reagent lot), so a
    model can "cheat" by learning plate identity instead of phenotype if the
    same plate appears in both train and eval. See TODO.md section 1.
    """
    plates = sorted({w.plate for w in wells})
    rng = random.Random(seed)
    rng.shuffle(plates)

    n_test = max(1, round(len(plates) * test_frac))
    n_val = max(1, round(len(plates) * val_frac))
    if n_test + n_val >= len(plates):
        raise ValueError(f"Not enough plates ({len(plates)}) to carve out val/test splits.")

    test_plates = set(plates[:n_test])
    val_plates = set(plates[n_test : n_test + n_val])
    train_plates = set(plates[n_test + n_val :])

    train = [w for w in wells if w.plate in train_plates]
    val = [w for w in wells if w.plate in val_plates]
    test = [w for w in wells if w.plate in test_plates]
    return train, val, test


def assert_no_plate_leakage(*splits: list[WellMetadata]) -> None:
    seen: dict[str, int] = {}
    for split_idx, split in enumerate(splits):
        for w in split:
            if w.plate in seen and seen[w.plate] != split_idx:
                raise AssertionError(f"Plate {w.plate} appears in multiple splits — leakage detected.")
            seen[w.plate] = split_idx
