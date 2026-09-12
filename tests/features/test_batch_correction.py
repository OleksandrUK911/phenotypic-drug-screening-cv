import numpy as np

from src.data.schema import WellMetadata
from src.features.batch_correction import typical_variation_normalization
from src.features.embeddings import CellEmbedding, aggregate_to_well


def _meta(plate: str, well: str, compound: str) -> WellMetadata:
    return WellMetadata(plate=plate, well=well, compound=compound, concentration_um=1.0, moa=None, replicate=1)


def test_tvn_centers_each_plates_controls_near_zero():
    rng = np.random.default_rng(0)
    dim = 8

    metadata = {}
    embeddings = {}
    for plate_idx, plate in enumerate(["p1", "p2"]):
        plate_offset = rng.normal(loc=plate_idx * 10, scale=1.0, size=dim)
        for w in range(6):
            well = f"w{w}"
            is_control = w < 3
            compound = "DMSO" if is_control else f"cmpd_{w}"
            metadata[(plate, well)] = _meta(plate, well, compound)
            noise = rng.normal(scale=0.5, size=dim)
            treatment_shift = np.zeros(dim) if is_control else rng.normal(scale=3.0, size=dim)
            embeddings[(plate, well)] = plate_offset + noise + treatment_shift

    corrected = typical_variation_normalization(embeddings, metadata)

    control_vecs = np.stack([corrected[(p, w)] for (p, w), m in metadata.items() if m.is_control])
    assert np.linalg.norm(control_vecs.mean(axis=0)) < 1.0


def test_tvn_raises_with_insufficient_controls():
    metadata = {
        ("p1", "w1"): _meta("p1", "w1", "DMSO"),
        ("p1", "w2"): _meta("p1", "w2", "cmpd_a"),
    }
    embeddings = {("p1", "w1"): np.zeros(4), ("p1", "w2"): np.ones(4)}

    try:
        typical_variation_normalization(embeddings, metadata)
    except ValueError:
        return
    raise AssertionError("Expected ValueError for insufficient control wells")


def test_aggregate_to_well_median_pooling():
    embeddings = [
        CellEmbedding(plate="p1", well="w1", embedding=np.array([1.0, 1.0])),
        CellEmbedding(plate="p1", well="w1", embedding=np.array([3.0, 3.0])),
        CellEmbedding(plate="p1", well="w1", embedding=np.array([100.0, 100.0])),
        CellEmbedding(plate="p1", well="w2", embedding=np.array([5.0, 5.0])),
    ]

    pooled = aggregate_to_well(embeddings, method="median")

    assert np.allclose(pooled[("p1", "w1")], [3.0, 3.0])
    assert np.allclose(pooled[("p1", "w2")], [5.0, 5.0])


def test_aggregate_to_well_rejects_unknown_method():
    try:
        aggregate_to_well([], method="bogus")
    except ValueError:
        return
    raise AssertionError("Expected ValueError for unknown aggregation method")
