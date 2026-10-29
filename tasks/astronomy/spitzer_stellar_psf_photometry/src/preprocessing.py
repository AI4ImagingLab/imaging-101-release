"""Observation validation and independent instrument PSF resampling."""
from pathlib import Path
import numpy as np
from scipy.interpolate import RectBivariateSpline

PIXEL_ARCSEC = 1.2
FLIGHT_PIXEL_ARCSEC = 0.30325
OVERSAMPLING = 4

def load_data(path):
    """Return float64 images (4,17,17) and flight PSF (81,81)."""
    with np.load(Path(path), allow_pickle=False) as data:
        images = data['images']
        psf = data['flight_psf']
    if images.shape != (1,4,17,17) or psf.shape != (1,81,81):
        raise ValueError('Unexpected acquisition array shape')
    if not np.isfinite(images).all() or not np.isfinite(psf).all():
        raise ValueError('Acquisition contains nonfinite values')
    return images[0].astype(float), psf[0].astype(float)

def prepare_psf(flight):
    """Resample independent flight PSF to 0.3 arcsec and sum-normalize to 16."""
    grid = np.arange(81,dtype=float)
    coordinates = 40+(grid-40)*(PIXEL_ARCSEC/OVERSAMPLING)/FLIGHT_PIXEL_ARCSEC
    psf = RectBivariateSpline(grid,grid,np.asarray(flight).T)(coordinates,coordinates).T
    return psf*(OVERSAMPLING**2/psf.sum())

def initialize(patch):
    """Return [flux,x,y,background] from peak, border median and positive excess."""
    y,x = np.unravel_index(np.argmax(patch),patch.shape)
    border = np.concatenate((patch[0],patch[-1],patch[1:-1,0],patch[1:-1,-1]))
    background = np.median(border)
    return np.array([np.maximum(patch-background,0).sum(),x,y,background],dtype=float)
