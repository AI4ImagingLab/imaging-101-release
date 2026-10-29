"""Closed-form finite-thickness Fourier traction inversion; GPL-3.0-or-later."""
from scipy.ndimage import gaussian_filter
from .preprocessing import prepare_displacement,PIXEL_SIZE_UM,GRID_PITCH_UM,YOUNG_PA,POISSON,THICKNESS_UM
from .physics_model import stiffness,apply_spectral

def reconstruct(displacement,mode='finite',smoothing_um=6.0):
    """Return (2,H,W) traction Pa from measured displacement in pixels.

    mode finite uses bonded 300 µm layer; infinite is an explicit comparison.
    Gaussian smoothing sigma 6.0 µm, applied after crop with reflect boundary.
    """
    if mode not in ('finite','infinite'):raise ValueError('Unknown elastic mode')
    padded=prepare_displacement(displacement)
    tensor=stiffness(padded.shape[-1],GRID_PITCH_UM,THICKNESS_UM if mode=='finite' else None,YOUNG_PA,POISSON)
    traction=apply_spectral(padded,tensor)
    h,w=displacement.shape[1:];traction=traction[:,-h:,-w:]
    if smoothing_um:
        traction=gaussian_filter(traction,sigma=(0,smoothing_um/GRID_PITCH_UM,smoothing_um/GRID_PITCH_UM))
    return traction
