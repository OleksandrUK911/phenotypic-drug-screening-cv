import numpy as np

from src.anomaly_detection.assay_quality import assess_plate_quality, ssmd, z_factor


def test_z_factor_excellent_when_well_separated_and_tight():
    rng = np.random.default_rng(0)
    pos = rng.normal(loc=10.0, scale=0.1, size=100)
    neg = rng.normal(loc=0.0, scale=0.1, size=100)

    z = z_factor(pos, neg)
    assert z > 0.5


def test_z_factor_unusable_when_controls_overlap():
    rng = np.random.default_rng(0)
    pos = rng.normal(loc=1.0, scale=5.0, size=100)
    neg = rng.normal(loc=0.0, scale=5.0, size=100)

    z = z_factor(pos, neg)
    assert z <= 0.0


def test_ssmd_sign_matches_direction_of_effect():
    pos = np.array([5.0, 5.5, 4.5])
    neg = np.array([1.0, 1.5, 0.5])
    assert ssmd(pos, neg) > 0
    assert ssmd(neg, pos) < 0


def test_assess_plate_quality_classifies_correctly():
    rng = np.random.default_rng(1)
    pos = rng.normal(loc=10.0, scale=0.1, size=50)
    neg = rng.normal(loc=0.0, scale=0.1, size=50)

    result = assess_plate_quality(pos, neg)
    assert result.quality == "excellent"
    assert result.z_factor > 0.5
