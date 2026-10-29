"""Closed-form least-squares Stokes reconstruction; no domain solver."""
import numpy as np
from .preprocessing import split_channels
from .physics_model import polarization

def estimate_stokes(channels: np.ndarray) -> np.ndarray:
    """Solve min_S ||A S-I||^2 independently at every effective superpixel.

    channels (...,4,H,W) -> float64 (...,3,H,W), all intensities in ADU.
    The explicit pseudoinverse is S0=sum(I)/2, S1=I0-I90, S2=I45-I135.
    """
    i0,i45,i90,i135 = np.moveaxis(channels,-3,0)
    return np.stack(((i0+i45+i90+i135)/2,i0-i90,i45-i135),axis=-3)

def reconstruct(raw: np.ndarray, offset: float = 0.0) -> dict[str,np.ndarray]:
    """Reconstruct all planes; no file side effects or spatial regularization."""
    stokes=estimate_stokes(split_channels(raw,offset=offset))
    angle,degree=polarization(stokes)
    return {'reconstruction':stokes,'orientation':angle,'degree':degree}
