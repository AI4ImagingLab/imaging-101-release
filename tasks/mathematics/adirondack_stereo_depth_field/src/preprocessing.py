"""Validation of the two acquired rectified RGB images."""
from pathlib import Path
import numpy as np

def load_observations(path):
    """Return left/right HWC uint8 images from the batch-first public NPZ."""
    with np.load(Path(path), allow_pickle=False) as data:
        images = [data[key] for key in ('left_image', 'right_image')]
    for image in images:
        if image.shape != (1, 496, 718, 3) or image.dtype != np.uint8:
            raise ValueError('Images must have shape (1,496,718,3) and dtype uint8')
    return images[0][0], images[1][0]
