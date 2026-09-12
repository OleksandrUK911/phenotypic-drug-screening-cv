from __future__ import annotations

import albumentations as A


def simclr_augmentations(crop_size: int = 96) -> A.Compose:
    """Two independently-sampled calls to this pipeline produce SimCLR's two views.

    Color-jitter-style augmentations are deliberately mild/absent: unlike
    natural images, per-channel *intensity* in Cell Painting is part of the
    phenotypic signal (stain intensity correlates with biology), so aggressive
    brightness/contrast jitter would corrupt the exact signal we want to learn.
    Geometric augmentations (flip/rotate/crop) are safe and are the main source
    of view diversity here.
    """
    return A.Compose(
        [
            A.RandomResizedCrop(size=(crop_size, crop_size), scale=(0.7, 1.0), p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        ]
    )
