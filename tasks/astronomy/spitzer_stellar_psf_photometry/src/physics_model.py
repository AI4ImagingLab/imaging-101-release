"""Explicit sampled, shifted cubic-spline PSF and additive sky model.

ImagePSF evaluation convention extracted from Photutils 2.3.0 (BSD-3-Clause).
See test/validation.md and private provenance for attribution and adaptations.
"""
import numpy as np
from scipy.interpolate import RectBivariateSpline

def make_interpolator(psf):
    """Cubic, exact interpolation in x,y order, matching the instrument grid."""
    return RectBivariateSpline(np.arange(81.),np.arange(81.),np.asarray(psf).T,kx=3,ky=3,s=0)

def model(parameters, interpolator, shape=(17,17)):
    """Evaluate MJy/sr image; parameters are flux [MJy/sr pixel], x,y [pixel], sky [MJy/sr]."""
    flux,x0,y0,sky = parameters
    y,x = np.indices(shape,dtype=float)
    xi=4*(x-x0)+40
    yi=4*(y-y0)+40
    sampled=interpolator(xi,yi,grid=False)
    sampled[(xi<0)|(xi>80)|(yi<0)|(yi>80)] = 0
    return flux*sampled+sky

def residual(parameters, interpolator, patch):
    """Flatten model minus observed image, without weighting or normalization."""
    return (model(parameters,interpolator,patch.shape)-patch).ravel()
