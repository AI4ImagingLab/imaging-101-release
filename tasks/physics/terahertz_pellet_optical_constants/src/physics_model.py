"""Local single-pass normal-incidence Fresnel slab model, SI internally.
Adapted from THzPy, LGPL-3.0; no imaging-domain dependency.
"""
import numpy as np
C=299792458.
def slab_inverse(amplitude,phase,frequency_hz,thickness_m,n_medium=1.):
    """Positive-delay phase -> real n and signed apparent alpha (cm^-1)."""
    n=C*phase/(2*np.pi*frequency_hz*thickness_m)+n_medium
    alpha_si=(2/thickness_m)*np.log(4*n*n_medium/(amplitude*(n_medium+n)**2))
    return n,alpha_si*.01
def slab_transfer(real_index,absorption_coefficient,frequency_hz,thickness_m,n_medium=1.):
    """Return amplitude and nonnegative delay phase; alpha input cm^-1.
    Single-pass real-index interface factors neglect absorption in Fresnel
    coefficients, multiple reflections, scattering and beam-path mismatch.
    """
    n=np.asarray(real_index);alpha_si=np.asarray(absorption_coefficient)*100
    amplitude=4*n*n_medium/(n_medium+n)**2*np.exp(-alpha_si*thickness_m/2)
    phase=2*np.pi*frequency_hz*(n-n_medium)*thickness_m/C
    return amplitude,phase
