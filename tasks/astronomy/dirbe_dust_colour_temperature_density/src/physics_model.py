"""PTS greybody equations, AGPL-3.0, Astronomical Observatory Ghent University.
Constants intentionally preserve the selected source revision, including old SI values.
"""
import numpy as np
C=2.99792458e8
H=6.62606957e-34
K=1.3806488e-23
PC=3.08567758e16
MSUN=1.9891e30
MASS_SCALE=MSUN/PC**2
KAPPA350=.192
BETA=2.
def planck(wavelength,temperature):
    """B_nu in W m^-2 Hz^-1 sr^-1, wavelength micrometres, temperature K."""
    nu=C/(np.asarray(wavelength)*1e-6)
    return 2*H*nu**3/C**2/(np.exp(H*nu/(K*temperature))-1)
def flux(wavelength,temperature,mass,beta=2.):
    """Original one-pc, one-steradian numerical parameterization, Jy.
    mass is solar-mass coefficient; physical surface density is mass*MASS_SCALE.
    """
    nu=C/(np.asarray(wavelength)*1e-6);nu350=C/350e-6
    opacity=KAPPA350*(nu/nu350)**beta
    return mass*MSUN*opacity*planck(wavelength,temperature)/PC**2*1e26
def forward(wavelengths,temperature,surface_density,beta=2.):
    """Maps(Y,X) -> band radiance(F,Y,X), MJy/sr; optically thin approximation."""
    return flux(np.asarray(wavelengths)[:,None,None],temperature[None],surface_density[None]/MASS_SCALE,beta)/1e6
def bandpass_prediction(wavelengths,temperature,surface_density,response_wavelength,spectral_response,beta=2.):
    """Diagnostic broad-band quoted radiances, not the selected monochromatic fit.
    DIRBE calibration assumes nu*I_nu constant. Integrate full measured R(nu).
    """
    rw=np.asarray(response_wavelength);nu=C/(rw*1e-6)
    spectrum=forward(rw,temperature,surface_density,beta)
    out=[]
    for nominal,response in zip(wavelengths,spectral_response):
        nu0=C/(nominal*1e-6)
        denominator=np.trapezoid((nu0/nu)*response,nu)
        out.append(np.trapezoid(spectrum*response[:,None,None],nu,axis=0)/denominator)
    return np.array(out)
def jacobian(wavelengths,temperature,surface_density):
    """Analytic per-pixel derivatives wrt temperature K and density kg/m^2."""
    w=np.asarray(wavelengths)[:,None,None];nu=C/(w*1e-6);x=H*nu/(K*temperature[None]);p=forward(wavelengths,temperature,surface_density)
    return np.stack([p*x/temperature[None]/(1-np.exp(-x)),p/surface_density[None]],axis=-1)
