"""Direct author pseudoinverse, all wavelengths, original output precision."""
import numpy as np
from .physics_model import oxygenation
def unmix(images,spectra):
    """Return coefficients(T,2,Z,Y,X)float32; no positivity projection/normalization."""
    if images.shape[1]!=spectra.shape[1] or spectra.shape[0]!=2:raise ValueError('Spectral shape mismatch')
    inverse=np.linalg.pinv(spectra)
    return np.einsum('ij...,jk->ik...',images,inverse).astype(np.float32)

def reconstruct(images,spectra,labels,wavelength_indices=None):
    """Return Hb/HbO2fields(T,2,Z,Y,X),saturation(T,Z,Y,X), zero outside fixed geometry."""
    if wavelength_indices is not None:
        images=images[:,wavelength_indices];spectra=spectra[:,wavelength_indices]
    coefficients=unmix(images,spectra);saturation=oxygenation(coefficients);support=labels>0
    if not np.isfinite(coefficients[...,support]).all() or not np.isfinite(saturation[...,support]).all():raise ValueError('Undefined inversion on fixed support')
    return np.where(support,coefficients,0).astype(np.float32),np.where(support,saturation,0).astype(np.float32)
