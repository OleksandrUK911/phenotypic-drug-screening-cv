import numpy as np

from src.segmentation.qc import qc_mask


def _uniform_cells_mask(n_cells: int, cell_size: int = 4, grid: int = 20) -> np.ndarray:
    mask = np.zeros((grid * cell_size, grid * cell_size), dtype=int)
    label = 1
    for i in range(grid):
        for j in range(grid):
            if label > n_cells:
                return mask
            y, x = i * cell_size, j * cell_size
            mask[y : y + cell_size, x : x + cell_size] = label
            label += 1
    return mask


def test_qc_flags_too_few_cells():
    mask = _uniform_cells_mask(n_cells=2)
    result = qc_mask(mask, min_cells=5)
    assert result.flagged
    assert "too few" in result.reason


def test_qc_passes_healthy_mask():
    mask = _uniform_cells_mask(n_cells=50)
    result = qc_mask(mask, min_cells=5, max_cells=2000, max_area_cv=2.0)
    assert not result.flagged
    assert result.n_cells == 50


def test_qc_flags_high_area_variability():
    mask = np.zeros((50, 50), dtype=int)
    mask[0:2, 0:2] = 1
    mask[10:40, 10:40] = 2
    mask[45:47, 45:47] = 3
    mask[5:7, 30:32] = 4
    mask[42:44, 5:7] = 5
    mask[20:22, 45:47] = 6

    result = qc_mask(mask, min_cells=5, max_cells=2000, max_area_cv=1.0)
    assert result.flagged
    assert "area CV" in result.reason
