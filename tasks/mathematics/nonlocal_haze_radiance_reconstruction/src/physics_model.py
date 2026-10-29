"""Atmospheric single-scattering image formation and radiance recovery.

Adapted mathematics: Dana Berman, Tali Treibitz and Shai Avidan,
Non-Local Image Dehazing, CVPR 2016. Non-commercial research license;
see task LICENSE and NOTICE. This module uses no upstream library.
"""
import numpy as np

GAMMA=1.0

def image_formation(radiance,transmission,airlight):
    """Return I=t J+(1-t) A, normalized linear intensity, H x W x 3."""
    return transmission[...,None]*radiance+(1-transmission[...,None])*airlight

def recover_radiance(image,transmission,airlight,gamma=GAMMA,leave_haze=1.06):
    """Source radiometric recovery before its display contrast stretch.

Source's leave_haze=1.06 adds 0.06*A before clipping when t>=0.1.
This aesthetic convention is not exact inversion of atmospheric formation.
"""
    corrected=image**gamma
    value=(corrected-(1-leave_haze*transmission[...,None])*airlight)/np.maximum(transmission[...,None],0.1)
    return np.clip(value,0,1)**(1/gamma)

def rgb_to_gray(image):
    """Octave 8.4 oracle's explicit floating-point luminance convention."""
    return image@np.array([0.298936,0.587043,0.114021])
