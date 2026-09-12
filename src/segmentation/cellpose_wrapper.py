from __future__ import annotations

import numpy as np


def segment_cells(
    image: np.ndarray,
    model_type: str = "cyto2",
    diameter: float | None = None,
    channels: tuple[int, int] = (1, 2),
    gpu: bool = False,
) -> np.ndarray:
    """Run a pretrained Cellpose model on one (C, H, W) or (H, W) image.

    `channels` follows Cellpose's convention: (cytoplasm_channel, nucleus_channel),
    0 meaning "grayscale/none". Cellpose is used as a strong pretrained baseline
    (see TODO.md section 2) rather than training segmentation from scratch first —
    a U-Net-from-scratch fallback lives alongside this for the labeled BBBC subset.

    Returns an integer label mask of shape (H, W), 0 = background.
    """
    from cellpose import models

    if image.ndim == 3:
        cyto_idx, nuc_idx = channels
        img_2ch = np.stack(
            [
                image[cyto_idx - 1] if cyto_idx > 0 else np.zeros(image.shape[1:], dtype=image.dtype),
                image[nuc_idx - 1] if nuc_idx > 0 else np.zeros(image.shape[1:], dtype=image.dtype),
            ],
            axis=-1,
        )
    else:
        img_2ch = image

    model = models.Cellpose(model_type=model_type, gpu=gpu)
    masks, _flows, _styles, _diams = model.eval(img_2ch, diameter=diameter, channels=[1, 2])
    return masks
