import numpy as np

from src.segmentation.postprocessing import filter_small_objects, relabel_sequential, split_touching_cells


def test_filter_small_objects_removes_tiny_regions_and_relabels():
    mask = np.zeros((10, 10), dtype=int)
    mask[0:3, 0:3] = 1
    mask[5:6, 5:6] = 2

    filtered = filter_small_objects(mask, min_area=4)

    assert set(np.unique(filtered)) == {0, 1}
    assert filtered[0, 0] == 1
    assert filtered[5, 5] == 0


def test_relabel_sequential_is_contiguous_from_one():
    mask = np.zeros((5, 5), dtype=int)
    mask[0, 0] = 7
    mask[4, 4] = 12

    relabeled = relabel_sequential(mask)

    assert set(np.unique(relabeled)) == {0, 1, 2}


def test_split_touching_cells_separates_two_blobs_joined_by_a_bridge():
    mask = np.zeros((20, 20), dtype=int)
    mask[2:8, 2:8] = 1
    mask[2:8, 8:9] = 1
    mask[2:8, 12:18] = 1

    split = split_touching_cells(mask, min_distance=3)

    n_instances = len(set(np.unique(split)) - {0})
    assert n_instances >= 2
