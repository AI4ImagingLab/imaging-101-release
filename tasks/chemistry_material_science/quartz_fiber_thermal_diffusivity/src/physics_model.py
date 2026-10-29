"""Explicit harmonic1D heat equation with a linear distributed loss term."""
import numpy as np
def thermal_profile(distance,diffusivity,frequency,loss_rate=0.,amplitude=1.,phase=0.):
    """Complex temperature(mK), distance mm, alpha mm²/s, frequency Hz, loss s^-1."""
    propagation=np.sqrt((loss_rate+2j*np.pi*frequency)/diffusivity)
    return amplitude*np.exp(1j*phase-propagation*np.asarray(distance))
def diffusivity_from_slopes(amplitude_slope,phase_slope,frequency):
    """Return alpha=|pi*f/(mA*mPhi)|, mm²/s; slopes mm^-1 and rad/mm."""
    return np.abs(np.pi*np.asarray(frequency)/(amplitude_slope*phase_slope))
def fit_forward(distance,coefficients):
    """Predict log-amplitude and phase; coefficient final axis slope,intercept."""
    c=np.asarray(coefficients)
    return c[...,0,None]*np.asarray(distance)+c[...,1,None]
