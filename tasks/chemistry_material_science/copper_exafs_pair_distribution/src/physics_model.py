# EXAFS equation and transform adapted from xraylarch, MIT; see task LICENSE.
# Copyright 2010-2026 Matthew Newville and collaborators.
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy import constants
REFF=2.5478
DEGENERACY=12.
KSTEP=.05
NFFT=2048
RSTEP=np.pi/(KSTEP*NFFT)
R=np.linspace(1.8,3.4,801)
INITIAL=np.array([1.,0.,0.,0.])
PARAMETER_SCALES=np.array([1.,1.,.01,.1])
ETOK=1./(1.e20*constants.hbar**2/(2*constants.m_e*constants.e))
def model_grid(data):
    return .05*np.arange(int(1.01+data['k'].max()/.05))
def scattering_splines(data):
    """Exact cubic, zero smoothing FEFF interpolation, including extrapolation."""
    return tuple(UnivariateSpline(data['scattering_k'],data[key],s=0) for key in ('scattering_phase','scattering_amplitude','real_momentum','mean_free_path'))
def forward(p,k,splines):
    """First-shell chi(k), with energy shift and full complex second-cumulant correction."""
    amp,e0,sigma2,dr=p
    en=k*k-e0*ETOK
    if np.min(np.abs(en))<1.e-6:en=np.where(np.abs(en)<1.5e-6,1.e-6,en)
    q=np.sign(en)*np.sqrt(np.abs(en))
    phase,f,rep,lam=(s(q) for s in splines)
    pp=(rep+1j/lam)**2
    momentum=np.sqrt(pp)
    cchi=np.exp(-2*REFF*momentum.imag-2*pp*sigma2+1j*(2*q*REFF+phase+2*momentum*(dr-2*sigma2/REFF)))
    cchi=DEGENERACY*amp*f*cchi/(q*(REFF+dr)**2)
    cchi[0]=2*cchi[1]-cchi[2]
    return cchi.imag
def kaiser_window():
    """Published implementation's beta=4 Kaiser support 1..19 Angstrom^-1."""
    k=KSTEP*np.arange(NFFT)
    arg=np.maximum(0.,1.-((k-10.)/9.)**2)
    return (np.i0(4*np.sqrt(arg))-1)/(np.i0(4)-1)
def transform(chi):
    """Negative-sign FFT; factor dk/sqrt(pi), zero padded to 2048."""
    k=KSTEP*np.arange(len(chi));buf=np.zeros(NFFT,dtype=complex)
    buf[:len(chi)]=chi*kaiser_window()[:len(chi)]*k**2
    return (KSTEP/np.sqrt(np.pi))*np.fft.fft(buf)[:NFFT//2]
def selected_transform(chi):
    lo=int(.01+1.4/RSTEP);hi=int(.01+3./RSTEP)
    return transform(chi)[lo:hi]
def epsilon_r():
    # Unit input weighting is a fit normalization, not an invented measured noise estimate.
    return 1./(2*np.sqrt(np.pi*5/(KSTEP*(17.**5-3.**5))))
def residuals(p,k,chi,splines):
    v=selected_transform(chi-forward(p,k,splines))/epsilon_r()
    return np.stack((v.real,v.imag),axis=-1).ravel()
def pair_density(p,r=R):
    """12-neighbor Gaussian radial density in Angstrom^-1, not a measured truth."""
    if p[2]<=0:raise ValueError('nonpositive fitted pair variance')
    return DEGENERACY*np.exp(-.5*(r-(REFF+p[3]))**2/p[2])/np.sqrt(2*np.pi*p[2])
def rfactor(p,k,chi,splines):
    diff=selected_transform(chi-forward(p,k,splines))
    ref=selected_transform(chi)
    return float(np.vdot(diff,diff).real/np.vdot(ref,ref).real)
