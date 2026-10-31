"""Explicit confocal/roll-off optics and calibrated single-scattering integral.

Smith/Dwork2015 mathematical implementation adapted under GPL-3.0.
All z distances in mm; wavelengths in m; signal is author-linear instrument units.
"""
import numpy as np
N_DEPTH=512;Z_MAX=2.358;DZ=Z_MAX/(N_DEPTH-1);DX=.00502
FOCUS=1.;RAYLEIGH=.29017970000000004
WAVELENGTH=1.325e-6;PIXEL_WAVELENGTH=7.1875e-11;RESOLUTION=1.1055e-10
NOISE_POWER=2.5e-5
def optical_response(z=None,focus=FOCUS,rayleigh=RAYLEIGH):
 """Return confocal and spectral roll-off arrays on optical-depth coordinates."""
 z=np.linspace(0,Z_MAX,N_DEPTH) if z is None else np.asarray(z)
 h=1/(1+((z-focus)/rayleigh)**2)
 ranging_depth=WAVELENGTH**2/(4*PIXEL_WAVELENGTH)*1000
 zeta=np.pi/2*z/ranging_depth;w=RESOLUTION/PIXEL_WAVELENGTH
 f=np.sinc(zeta/np.pi)**2*np.exp(-w*w/(2*np.log(2))*zeta*zeta)
 return h,f
def corrected_signal(signal,mask,focus=FOCUS,rayleigh=RAYLEIGH,noise_power=NOISE_POWER,include_confocal=True,include_falloff=True):
 a=np.asarray(signal)*np.asarray(mask);h,f=optical_response(focus=focus,rayleigh=rayleigh)
 transfer=(h if include_confocal else np.ones_like(h))*(f if include_falloff else np.ones_like(f))
 return a/transfer[:,None]*(a*a/(a*a+noise_power))
def cell_signal(attenuation,dz=DZ,scale=1.):
 """Independent exact Beer-Lambert cell integral, for physics checks.

 Shape(depth,...) nonnegative rates. Cell integral=C/2*(S_front-S_back),
 S_next=S_front*exp(-2*mu*dz); excludes instrument optics/noise.
 """
 mu=np.asarray(attenuation,dtype=float);optical=2*dz*mu
 front=np.exp(-np.concatenate((np.zeros_like(mu[:1]),np.cumsum(optical[:-1],axis=0)),axis=0))
 return .5*scale*front*(-np.expm1(-optical))
def consistency(attenuation,term,dz=DZ):
 """Selected Riemann inverse identity, not independently measured truth error."""
 cumulative=np.cumsum(np.asarray(term)[:-1][::-1],axis=0)[::-1]
 residual=np.asarray(term)[:-1]/(2*dz)-np.asarray(attenuation)[:-1]*cumulative
 denominator=np.linalg.norm(np.asarray(term)[:-1]/(2*dz))
 return float(np.linalg.norm(residual)/denominator) if denominator>0 else float(np.linalg.norm(residual))
