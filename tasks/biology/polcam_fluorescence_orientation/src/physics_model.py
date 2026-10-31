"""Explicit linear analyzer model, its adjoint, and polarization coordinates."""
import numpy as np

def analyzer_matrix() -> np.ndarray:
    """Map (S0,S1,S2) to ideal analyzer intensities (0,45,90,135), dimensionless."""
    return np.array([[1,1,0],[1,0,1],[1,-1,0],[1,0,-1]], dtype=np.float64)/2

def forward(stokes: np.ndarray) -> np.ndarray:
    """Return (...,4,H,W) ADU from (...,3,H,W) Stokes ADU."""
    s0,s1,s2 = np.moveaxis(stokes,-3,0)
    return np.stack(((s0+s1)/2,(s0+s2)/2,(s0-s1)/2,(s0-s2)/2),axis=-3)

def adjoint(channels: np.ndarray) -> np.ndarray:
    """Transpose of forward under the real Euclidean inner product."""
    i0,i45,i90,i135 = np.moveaxis(channels,-3,0)
    return np.stack(((i0+i45+i90+i135)/2,(i0-i90)/2,(i45-i135)/2),axis=-3)

def polarization(stokes: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    """Effective AoLP in [-pi/2,pi/2] and DoLP (unclipped dimensionless).

    atan2(0,0)=0 fixes the undefined angle of isotropic superpixels. DoLP=0 if
    S0=0 and S1=S2=0; nonzero polarized signal at zero S0 is rejected.
    """
    s0,s1,s2=np.moveaxis(stokes,-3,0)
    length=np.hypot(s1,s2)
    if np.any((s0 == 0) & (length != 0)):
        raise ValueError('nonzero polarized signal at zero S0')
    degree=np.divide(length,np.abs(s0),out=np.zeros_like(length),where=s0!=0)
    return 0.5*np.arctan2(s2,s1),degree

def relative_residual(channels: np.ndarray, stokes: np.ndarray) -> float:
    """Frobenius analyzer residual divided by observed-channel norm."""
    norm=np.linalg.norm(channels.ravel())
    return float(np.linalg.norm((forward(stokes)-channels).ravel())/norm) if norm else 0.0
