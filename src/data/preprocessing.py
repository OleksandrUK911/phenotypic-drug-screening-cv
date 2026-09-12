from __future__ import annotations

import numpy as np
from skimage import exposure, io


def load_channel(path: str) -> np.ndarray:
    return io.imread(path).astype(np.float32)


def correct_illumination(image: np.ndarray, sigma: float = 100.0) -> np.ndarray:
    """Divide out a smooth illumination field estimated by heavy Gaussian blur.

    Standard fix for microscopy vignetting/uneven illumination across a well —
    without it, intensity-based features confound position-in-well with phenotype.
    """
    from scipy.ndimage import gaussian_filter

    background = gaussian_filter(image, sigma=sigma)
    background[background == 0] = background[background > 0].mean() if (background > 0).any() else 1.0
    corrected = image / background
    return corrected * background.mean()


def normalize_channel(image: np.ndarray, low_pct: float = 0.1, high_pct: float = 99.9) -> np.ndarray:
    """Percentile-clip and rescale a single channel to [0, 1].

    Percentile (not min/max) clipping avoids a handful of saturated pixels
    from a debris speck compressing the dynamic range of the whole image.
    """
    lo, hi = np.percentile(image, [low_pct, high_pct])
    return exposure.rescale_intensity(image, in_range=(lo, hi), out_range=(0.0, 1.0))


def preprocess_well_image(channel_paths: dict[str, str], illumination_sigma: float = 100.0) -> np.ndarray:
    """Load, illumination-correct, and normalize all channels for one field of view.

    Returns an array of shape (n_channels, H, W) with channels ordered by name.
    """
    channels = []
    for name in sorted(channel_paths):
        img = load_channel(channel_paths[name])
        img = correct_illumination(img, sigma=illumination_sigma)
        img = normalize_channel(img)
        channels.append(img)
    return np.stack(channels, axis=0)


def tile_image(image: np.ndarray, tile_size: int = 512, overlap: int = 0) -> list[np.ndarray]:
    """Split a (C, H, W) image into (possibly overlapping) square tiles.

    Large plate scans don't fit a typical model's receptive field / memory
    budget in one shot, so segmentation and SSL pretraining operate on tiles.
    """
    _, h, w = image.shape
    stride = tile_size - overlap
    tiles = []
    for y in range(0, max(h - tile_size, 0) + 1, stride):
        for x in range(0, max(w - tile_size, 0) + 1, stride):
            tiles.append(image[:, y : y + tile_size, x : x + tile_size])
    return tiles
