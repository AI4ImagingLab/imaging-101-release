"""Explicit discrete calibrated magnetic-particle imaging acquisition."""
import numpy as np

def forward(system,coefficients):
    """u=S c; coefficients relative to one independent0.4umolFe delta."""
    c=np.asarray(coefficients).ravel();out=np.empty(system.shape[0],dtype=np.result_type(system,c))
    for a in range(0,len(out),64):out[a:a+64]=system[a:a+64]@c
    return out

def adjoint(system,spectrum):
    """Complex Hilbert adjoint S^H, same column order as the calibration."""
    out=np.zeros(system.shape[1],dtype=np.result_type(system,spectrum))
    for a in range(0,system.shape[0],64):out+=system[a:a+64].conjugate().T@spectrum[a:a+64]
    return out

def objective(system,measurement,coefficients,regularization):
    c=np.asarray(coefficients).ravel();residual=forward(system,c)-measurement
    return float(np.vdot(residual,residual).real+regularization*np.vdot(c,c).real)

def gradient_real(system,measurement,coefficients,regularization):
    """Real-variable gradient of ||Sc-u||²+lambda||c||², no1/2 factor."""
    c=np.asarray(coefficients).ravel()
    return 2*(adjoint(system,forward(system,c)-measurement).real+regularization*c)

def relative_residual(system,measurement,coefficients):
    return float(np.linalg.norm(forward(system,coefficients)-measurement)/np.linalg.norm(measurement))
