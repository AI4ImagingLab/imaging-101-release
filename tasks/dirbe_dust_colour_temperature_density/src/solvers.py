"""Explicit PTS positive-flux single-temperature greybody fitting, AGPL-3.0.
Astronomical Observatory, Ghent University. No PTS runtime dependency.
"""
import numpy as np
from scipy.optimize import curve_fit
from .physics_model import flux,forward,MASS_SCALE
def fit_pixel(wavelengths,intensity,uncertainty,temperature_init=20.,mass_init=.01,beta=2.):
    """Weighted LM fit, preserving original Jy scaling and parameterization.
    Return temperature K, surface density kg/m^2. Nonpositive data -> zero pair.
    """
    if not np.all(np.asarray(intensity)>0):return np.array([0.,0.])
    def model(w,t,m):return flux(w,t,m,beta)
    p,_=curve_fit(model,wavelengths,np.asarray(intensity)*1e6,p0=(temperature_init,mass_init),sigma=np.asarray(uncertainty)*1e6,absolute_sigma=False,maxfev=5000)
    return np.array([p[0],p[1]*MASS_SCALE])
def reconstruct(data,beta=2.):
    """Fit all32x48 native-pixel samples independently, return maps and prediction."""
    d=data['intensity'];maps=np.empty((2,)+d.shape[1:])
    for y,x in np.ndindex(d.shape[1:]):maps[:,y,x]=fit_pixel(data['wavelengths'],d[:,y,x],data['uncertainty'][:,y,x],beta=beta)
    return maps,forward(data['wavelengths'],maps[0],maps[1],beta)
