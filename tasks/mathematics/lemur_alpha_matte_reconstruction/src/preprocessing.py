"""Public native observation and independent trimap loading."""
from pathlib import Path
import numpy as np

IMAGE_SHAPE = (1, 440, 680, 3)
TRIMAP_SHAPE = (1, 440, 680)


def load_data(path='data/raw_data.npz'):
    with np.load(Path(path), allow_pickle=False) as z:
        if set(z.files) != {'image', 'trimap'}:
            raise ValueError('exactly image and trimap required')
        image = z['image']; trimap = z['trimap']
    for a, shape in [(image, IMAGE_SHAPE), (trimap, TRIMAP_SHAPE)]:
        if a.shape != shape or a.dtype != np.float64 or not np.isfinite(a).all():
            raise ValueError('observation shape, float64 dtype or finite-value violation')
        if a.min() < 0 or a.max() > 1:
            raise ValueError('observation values must lie in [0,1]')
    if set(np.unique(trimap)) != {0., .4, 1.}:
        raise ValueError('authored trimap values must be 0,0.4,1')
    return image[0], trimap[0]


def partition_trimap(trimap):
    foreground = trimap.ravel() >= 1
    background = trimap.ravel() <= 0
    known = foreground | background
    return foreground, background, known, ~known
