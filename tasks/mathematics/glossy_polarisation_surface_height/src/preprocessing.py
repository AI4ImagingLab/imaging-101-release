"""Polarimetric sinusoid decomposition, adapted from William Smith (2016).

Licensed under Apache-2.0; see LICENSE. Source revision and adaptations are
recorded in test/validation.md. Angles are radians clockwise from image-up.
"""
from pathlib import Path
import numpy as np


def load_observations(path):
    """Return images[H,W,A], angles[A], mask[H,W], specular_mask[H,W]."""
    with np.load(Path(path), allow_pickle=False) as z:
        images = z['images'][0].copy()
        angles = z['angles'][0].copy()
        mask = z['mask'][0].copy()
        spec = z['specular_mask'][0].copy()
    if images.shape != (*mask.shape, angles.size) or spec.shape != mask.shape:
        raise ValueError('Observation array shapes disagree')
    if not np.isfinite(images).all() or not np.isfinite(angles).all():
        raise ValueError('Nonfinite observations')
    return images, angles, mask, spec


def decompose(images, angles, mask):
    """Fit I(a)=b0+b1 cos(2a)+b2 sin(2a); return rho, phase, mean intensity."""
    design = np.column_stack((np.ones(angles.size), np.cos(2*angles), np.sin(2*angles)))
    coefficients = np.linalg.lstsq(design, images[mask].T, rcond=None)[0].T
    amplitude = np.hypot(coefficients[:, 1], coefficients[:, 2])
    # Preserve upstream max/min formulas rather than silently changing rounding.
    maximum = coefficients[:, 0] + amplitude
    minimum = coefficients[:, 0] - amplitude
    fields = ((maximum-minimum)/(maximum+minimum),
              np.mod(.5*np.arctan2(coefficients[:, 2], coefficients[:, 1]), np.pi),
              (minimum+maximum)/2)
    outputs = []
    for values in fields:
        field = np.zeros(mask.shape, dtype=np.float64)
        field[mask] = values
        outputs.append(field)
    return tuple(outputs)
