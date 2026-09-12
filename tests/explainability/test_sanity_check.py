import numpy as np

from src.explainability.sanity_check import cam_mass_inside_mask, flag_background_heavy_explanations


def test_cam_mass_inside_mask_all_inside():
    cam = np.zeros((10, 10))
    cam[4:6, 4:6] = 1.0
    mask = np.zeros((10, 10), dtype=bool)
    mask[3:7, 3:7] = True

    assert cam_mass_inside_mask(cam, mask) == 1.0


def test_cam_mass_inside_mask_all_outside():
    cam = np.zeros((10, 10))
    cam[0:2, 0:2] = 1.0
    mask = np.zeros((10, 10), dtype=bool)
    mask[8:10, 8:10] = True

    assert cam_mass_inside_mask(cam, mask) == 0.0


def test_cam_mass_inside_mask_handles_zero_cam():
    cam = np.zeros((5, 5))
    mask = np.ones((5, 5), dtype=bool)
    assert cam_mass_inside_mask(cam, mask) == 0.0


def test_flag_background_heavy_explanations():
    good_cam = np.zeros((10, 10))
    good_cam[4:6, 4:6] = 1.0
    bad_cam = np.zeros((10, 10))
    bad_cam[0:2, 0:2] = 1.0

    mask = np.zeros((10, 10), dtype=bool)
    mask[3:7, 3:7] = True

    flagged = flag_background_heavy_explanations([good_cam, bad_cam], [mask, mask], min_inside_fraction=0.5)
    assert flagged == [1]
