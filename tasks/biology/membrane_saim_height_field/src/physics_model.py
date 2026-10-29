"""One-colour TE thin-film interference, with lengths in nm.

Adapted mathematical equations from SAIMFitKit, MIT (c)2021 paszeklab.
Acquisition angles are angles in buffer, not in the silicon substrate.
"""
import numpy as np
N_BUFFER=1.33
OXIDE_NM=1924.9
WAVELENGTH_NM=642.0
N_OXIDE=1.4719
N_SILICON=3.8660+0.017933j

def optical_constants(angles,n_buffer=N_BUFFER,oxide_nm=OXIDE_NM,wavelength_nm=WAVELENGTH_NM,n_oxide=N_OXIDE,n_silicon=N_SILICON):
    """Return (angle,3) real columns Re(rTE),Im(rTE),phase/nm."""
    theta=np.asarray(angles,dtype=np.float64).reshape(-1)
    ox=np.arcsin(n_buffer*np.sin(theta)/n_oxide)
    si=np.arcsin(n_oxide*np.sin(ox)/complex(n_silicon))
    delta=2*np.pi*n_oxide/wavelength_nm*oxide_nm*np.cos(ox)
    p_si=n_silicon*np.cos(si);p_ox=n_oxide*np.cos(ox);p_b=n_buffer*np.cos(theta)
    m11=np.cos(delta);m12=-1j/p_ox*np.sin(delta);m21=-1j*p_ox*np.sin(delta)
    r=((m11+m12*p_si)*p_b-(m21+m11*p_si))/((m11+m12*p_si)*p_b+(m21+m11*p_si))
    return np.column_stack((r.real,r.imag,4*np.pi*n_buffer/wavelength_nm*np.cos(theta)))

def forward(constants,parameters):
    """Evaluate32 predicted ADU for [amplitude ADU,background ADU,height nm]."""
    c,d,k=np.asarray(constants).T;a,b,h=np.asarray(parameters)
    return a*(1+2*c*np.cos(k*h)-2*d*np.sin(k*h)+c*c+d*d)+b

def jacobian(constants,parameters):
    """Analytic signal derivatives with respect to A,B,H (not objective)."""
    c,d,k=np.asarray(constants).T;a,b,h=np.asarray(parameters)
    return np.column_stack((1+2*c*np.cos(k*h)-2*d*np.sin(k*h)+c*c+d*d,np.ones(c.size),-2*a*k*(c*np.sin(k*h)+d*np.cos(k*h))))

def predict_stack(result,angles,**optics):
    """Return(batch,angle,row,column) prediction, preserving entire native grid."""
    constants=optical_constants(angles,**optics)
    p=np.column_stack([result[k].ravel() for k in ['amplitude','background','reconstruction']])
    shape=result['reconstruction'].shape
    return np.array([forward(constants,v) for v in p]).T.reshape((shape[0],constants.shape[0],*shape[1:]))
