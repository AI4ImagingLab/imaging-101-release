"""Square-aperture Gaussian-PSF membrane diffusion observation model, GPL-3.0."""
import numpy as np
from scipy.special import erf

APERTURE=.32
PSF_SIGMA=.19
CURVATURE=2.1

def correlation_model(lag,molecule_number,diffusion,offset,aperture=APERTURE,sigma=PSF_SIGMA,curvature=CURVATURE):
    """Normalized correlation; diffusion um^2/s, lengths um, lag seconds.

    Fitter intentionally uses source unconstrained LM; invalid trial steps may
    warn, while standard reconstruction is checked for positive final N and D.
    """
    lag=np.asarray(lag,dtype=np.float64)
    k=aperture/(2*np.sqrt(sigma**2+curvature*diffusion*lag))
    overlap=erf(k)+(np.exp(-k**2)-1)/(k*np.sqrt(np.pi))
    return (1/molecule_number)*overlap**2+offset
