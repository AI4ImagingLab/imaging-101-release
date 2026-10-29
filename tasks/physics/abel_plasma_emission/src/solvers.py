"""Explicit inverse driving derivative and local Hansen-Law reconstruction.

Adapted from PyAbel v0.9.1 abel/hansenlaw.py (MIT).
Copyright (c) 2015 The PyAbel Team. See ../test/validation.md for the full MIT terms.
"""
import numpy as np
from .physics_model import propagate, state_coefficients, validate_half_image
from .preprocessing import DR, SHAPE


def inverse_drive(image, dr=DR, hold_order=1):
    """Spatial derivative in radiance/mm, using upstream finite differences."""
    if hold_order == 1:
        return np.gradient(image, dr, axis=-1)
    drive = np.zeros_like(image)
    drive[:, :-1] = (image[:, 1:]-image[:, :-1])/dr
    return drive


def inverse_transform(projection, dr=DR, hold_order=1):
    """Invert a vector or (z,y) half-image to radiance/mm without clipping."""
    image, vector = validate_half_image(projection, dr, hold_order)
    drive = inverse_drive(image, dr, hold_order)
    result = propagate(drive, state_coefficients(image.shape[1], 'inverse', hold_order))
    return result[0] if vector else result


def reconstruct(projection, dr=DR, hold_order=1):
    """Reconstruct the benchmark's batch-first float64 input without rescaling."""
    if projection.shape != SHAPE or projection.dtype != np.float64:
        raise ValueError(f'Expected float64 projection of shape {SHAPE}')
    return inverse_transform(projection[0], dr, hold_order)[None]
