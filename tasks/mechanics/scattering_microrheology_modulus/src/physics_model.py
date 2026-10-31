"""Siegert/Gaussian displacement and generalized Stokes-Einstein operators.
BSD3 adaptation of DLSuR (2018 developers); see task LICENSE and provenance.
"""
import numpy as np
from scipy.special import gamma
from .preprocessing import N,ANGLE_DEGREES,WAVELENGTH_NM,RADIUS_NM,TEMPERATURE_K,KB
def scattering_vector(n=N,angle=ANGLE_DEGREES,wavelength=WAVELENGTH_NM):
    """Magnitude q in nm^-1; angle in degrees, wavelength in nm."""
    return 4*np.pi*n*np.sin(np.deg2rad(angle)/2)/wavelength
def stretched_exponential(t,amplitude,rate,exponent):
    """Correlation model with microsecond lags; rate has us^-exponent units."""
    return amplitude*np.exp(-rate*t**exponent)
def correlation_forward(msd,intercept,q=None):
    """Return dimensionless g2-1 from MSD in nm^2 and Siegert intercept."""
    if q is None:q=scattering_vector()
    return intercept*np.exp(-q*q*msd/3)
def msd_from_field(g1,q=None):
    """Return three-dimensional MSD in nm^2 from real field correlation."""
    if q is None:q=scattering_vector()
    return -6*np.log(g1)/q**2
def modulus(t,msd,alpha,radius=RADIUS_NM,temperature=TEMPERATURE_K):
    """Return omega (s^-1), storage and loss (Pa), all shape (N,)."""
    amplitude=KB*temperature*1e27/(np.pi*radius*msd*gamma(1+alpha))
    return 1e6/t,amplitude*np.cos(np.pi*alpha/2),amplitude*np.sin(np.pi*alpha/2)
