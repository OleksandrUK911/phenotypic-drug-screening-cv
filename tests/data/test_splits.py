from src.data.schema import WellMetadata
from src.data.splits import assert_no_plate_leakage, plate_aware_split


def _make_wells(n_plates: int, wells_per_plate: int) -> list[WellMetadata]:
    wells = []
    for p in range(n_plates):
        for w in range(wells_per_plate):
            wells.append(
                WellMetadata(
                    plate=f"plate_{p}",
                    well=f"well_{w}",
                    compound="DMSO" if w == 0 else f"compound_{w}",
                    concentration_um=1.0,
                    moa=None,
                    replicate=1,
                )
            )
    return wells


def test_plate_aware_split_has_no_leakage():
    wells = _make_wells(n_plates=10, wells_per_plate=4)
    train, val, test = plate_aware_split(wells, val_frac=0.2, test_frac=0.2, seed=42)

    assert train and val and test
    assert_no_plate_leakage(train, val, test)


def test_plate_aware_split_is_deterministic():
    wells = _make_wells(n_plates=10, wells_per_plate=4)
    split_a = plate_aware_split(wells, seed=7)
    split_b = plate_aware_split(wells, seed=7)
    assert [w.plate for w in split_a[0]] == [w.plate for w in split_b[0]]


def test_plate_aware_split_raises_when_too_few_plates():
    wells = _make_wells(n_plates=2, wells_per_plate=4)
    try:
        plate_aware_split(wells, val_frac=0.4, test_frac=0.4)
    except ValueError:
        return
    raise AssertionError("Expected ValueError for too few plates")
