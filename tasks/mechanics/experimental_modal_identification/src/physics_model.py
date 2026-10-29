"""Accelerance pole-residue expansion; adapted from SDyPy-EMA, MIT LICENSE."""
import numpy as np
def pole_properties(poles):
    omega=np.abs(poles)*np.sign(poles.imag)
    return omega/(2*np.pi),-poles.real/omega
def modal_design(poles,frequency):
    w=2*np.pi*np.asarray(frequency)[:,None]
    a=-w**2/(1j*w-poles[None,:]);b=-w**2/(1j*w-np.conj(poles)[None,:])
    # Columns multiply Re(A), Im(A), Re/Im lower residual, Re/Im upper residual.
    z=np.column_stack([a+b,1j*(a-b),np.ones(len(w)),1j*np.ones(len(w)),-w[:,0]**2,-1j*w[:,0]**2])
    return np.vstack([z.real,z.imag])
def forward(poles,constants,lower_residual,upper_residual,frequency):
    p=modal_design(poles,frequency)
    coefficients=np.column_stack([constants.real,constants.imag,lower_residual.real,lower_residual.imag,upper_residual.real,upper_residual.imag])
    z=p@coefficients.T;nr=len(frequency)
    return (z[:nr]+1j*z[nr:]).T
