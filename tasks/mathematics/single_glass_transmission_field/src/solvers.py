"""Published convex inverse evaluated by its explicit Neumann cosine formula."""
import numpy as np
from scipy.fft import dctn,idctn
from .physics_model import right_hand_side
def reconstruct(image,h=.033,epsilon=1e-8):
    """Return full unclipped transmission-like field in input intensity units.

    Exact objective: 1/2||L T-div(delta_h grad Y)||² + eps/2||T-Y||².
    L is the local Neumann Laplacian, channel independent. Positive epsilon
    makes the minimizer unique, including the otherwise unknown image mean.
    This selected image-mean convention is not physical reflection-DC truth.
    """
    if isinstance(epsilon,bool) or not np.isfinite(epsilon) or epsilon<=0:raise ValueError('epsilon must be positive and finite')
    a=np.asarray(image,dtype=np.float64)
    if a.ndim not in (2,3) or not np.isfinite(a).all():raise ValueError('Expected finite grayscale or channel-last image')
    height,width=a.shape[:2];row=np.cos(np.pi*np.arange(height)/height);col=np.cos(np.pi*np.arange(width)/width)
    eigenvalue=2*(row[:,None]+col[None,:]-2)
    denominator=eigenvalue**2+epsilon
    if a.ndim==3:denominator=denominator[...,None]
    rhs=right_hand_side(a,h,epsilon)
    return idctn(dctn(rhs,type=2,axes=(0,1),norm='ortho')/denominator,type=2,axes=(0,1),norm='ortho')
