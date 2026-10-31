"""Explicit isotropic two-by-two thin-film transfer matrices.

Adapted mathematics from pyElli 4a8787654477355167df74bbf0b6003e8fc35a4d,
GPL-3.0; M. J. Mueller, S. Chatterjee and F. Dobener (2025).
All interface, propagation and outgoing-wave conventions are executable here.
"""
import numpy as np
from numpy.lib.scimath import arcsin,sqrt

MODEL_ANGLE_DEG=70.
ACTUAL_ANGLE_DEG=70.06
OXIDE_THICKNESS_NM=276.36

def cauchy_index(wavelength,parameters):
    """n=n0+100*n1/lambda_nm**2+1e7*n2/lambda_nm**4; return real n."""
    wavelength=np.asarray(wavelength,dtype=np.float64)
    p=np.asarray(parameters,dtype=np.float64)
    return p[0]+1e2*p[1]/wavelength**2+1e7*p[2]/wavelength**4

def fresnel(n_i,n_t,theta_i,theta_t):
    """Return (r_s,r_p,t_s,t_p), incident/transmitted indices and radian angles."""
    ci,ct=np.cos(theta_i),np.cos(theta_t)
    rs=(n_i*ci-n_t*ct)/(n_i*ci+n_t*ct)
    rp=(n_t*ci-n_i*ct)/(n_t*ci+n_i*ct)
    ts=2*n_i*ci/(n_i*ci+n_t*ct)
    tp=2*n_i*ci/(n_t*ci+n_i*ct)
    return rs,rp,ts,tp

def forward_angle(index,angle):
    """Outgoing passive branch: positive imaginary normal wavevector, then real."""
    value=index*np.cos(angle)
    return np.where(abs(value.imag)>1e-10,value.imag>0,value.real>0)

def reflection(wavelength,parameters,silicon_index,angle_deg=MODEL_ANGLE_DEG,
               oxide_thickness=OXIDE_THICKNESS_NM):
    """Return complex (r_s,r_p) for air/TiO2/SiO2/Si at every wavelength.

    Wavelength and thickness are nm. Index convention is n+i*k. Only the two
    finite films propagate; the ambient and calibrated substrate are half-spaces.
    """
    wavelength=np.asarray(wavelength,dtype=np.float64).reshape(-1)
    p=np.asarray(parameters,dtype=np.float64)
    if p.shape!=(4,) or not np.isfinite(p).all() or not np.isfinite(wavelength).all() or np.any(wavelength<=0):
        raise ValueError('Invalid parameters or wavelength')
    substrate=np.asarray(silicon_index,dtype=np.complex128).reshape(-1)
    if substrate.shape!=wavelength.shape or not np.isfinite(substrate).all():raise ValueError('Invalid substrate calibration')
    # Source material API squares its index before Solver2x2 takes this root.
    indices=sqrt(np.vstack([np.ones_like(wavelength),cauchy_index(wavelength,p),
                            1.452+100*36./wavelength**2,substrate])**2)
    theta=arcsin(indices[0]*np.sin(np.deg2rad(angle_deg))/indices)
    for j in [0,3]:theta[j]=np.where(~forward_angle(indices[j],theta[j]),np.pi-theta[j],theta[j])
    kz=2*np.pi*indices*np.cos(theta)/wavelength
    phase=kz[1:-1]*np.array([p[3],oxide_thickness])[:,None]
    ones=np.ones(len(wavelength))
    rs,rp,ts,tp=fresnel(indices[0],indices[1],theta[0],theta[1])
    ms=np.array([[ones,rs],[rs,ones]],dtype=complex)/ts
    mp=np.array([[ones,rp],[rp,ones]],dtype=complex)/tp
    for j in [1,2]:
        rs,rp,ts,tp=fresnel(indices[j],indices[j+1],theta[j],theta[j+1])
        em=np.exp(-1j*phase[j-1]);ep=np.exp(1j*phase[j-1])
        ms=np.einsum('ij...,jk...->ik...',ms,np.array([[em,rs*em],[rs*ep,ep]],dtype=complex)/ts)
        mp=np.einsum('ij...,jk...->ik...',mp,np.array([[em,rp*em],[rp*ep,ep]],dtype=complex)/tp)
    return ms[1,0]/ms[0,0],mp[1,0]/mp[0,0]

def predict_rho(data,parameters,angle_deg=MODEL_ANGLE_DEG):
    """Return reflected p/s complex amplitude ratio, retaining source handedness."""
    rs,rp=reflection(data['wavelength_nm'],parameters,data['silicon_index'],angle_deg)
    return rp/rs
