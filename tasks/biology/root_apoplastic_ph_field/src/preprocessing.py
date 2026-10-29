"""Unchanged native ratio observations and independent root calibrations."""
from pathlib import Path
import numpy as np

IMAGE_SHAPE=(1024,1024)
GROUPS=('control',)*12+('DCCD',)*12+('FC',)*8+('IAA',)*11
PIXEL_WIDTH_UM=0.18037518037518038


def valid_pixels(ratio):
    """Original rule: non-NaN, nonzero ratio is measured; no pH-dependent mask."""
    ratio=np.asarray(ratio)
    return np.isfinite(ratio)&(ratio!=0)


def load_observations(path):
    """Return native float32 ratio [43,y,x], low [9,y,x], high [8,y,x]."""
    with np.load(Path(path),allow_pickle=False) as a:
        expected={'ratio','calibration_low','calibration_high'}
        if set(a.files)!=expected:raise ValueError('Require exactly ratio and two calibration image groups')
        arrays=[a[key] for key in ['ratio','calibration_low','calibration_high']]
    for array,count in zip(arrays,[43,9,8]):
        if array.shape!=(1,count,*IMAGE_SHAPE) or array.dtype!=np.float32:
            raise ValueError('Incorrect native float32 observation shape')
        if np.isinf(array).any():raise ValueError('Infinite ratio observations are invalid')
        if not np.all(np.any(valid_pixels(array[0]),axis=(1,2))):raise ValueError('Every image needs measured pixels')
    return tuple(a[0] for a in arrays)


def calibration_levels(low,high):
    """Return lower/upper means and all independent per-root image means.

    ImageJ statistics ignore NaNs, include finite zeros, and do not use the
    target root. Individual roots are equally weighted in the final mean.
    """
    means=[]
    for group in [low,high]:
        group=np.asarray(group)
        if group.ndim!=3 or not len(group) or np.isinf(group).any():raise ValueError('Need nonempty finite-or-NaN calibration images')
        values=np.array([np.mean(image[np.isfinite(image)].astype(np.float64)) for image in group])
        if not np.isfinite(values).all():raise ValueError('Empty calibration image')
        means.append(values)
    levels=[sum(float(x) for x in values)/len(values) for values in means]
    if levels[1]<=levels[0]:raise ValueError('Upper calibration ratio must exceed lower')
    return levels[0],levels[1],means[0],means[1]
