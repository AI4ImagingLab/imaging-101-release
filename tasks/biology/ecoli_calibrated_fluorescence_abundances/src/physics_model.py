"""Local calibrated linear spectral acquisition and Poisson fidelity."""
import numpy as np

def forward(abundances, spectra):
    """Map (...,13) digital brightness coefficients to (...,24) intensities."""
    return np.asarray(abundances) @ np.asarray(spectra).T

def adjoint(spectral_values, spectra):
    return np.asarray(spectral_values) @ np.asarray(spectra)

def poisson_loss(predicted, measured):
    """Poisson negative log likelihood without constants, summed over entries."""
    p,y=np.asarray(predicted),np.asarray(measured)
    if np.any(p<=0): return float('inf')
    return float(np.sum(p-y*np.log(p)))

def poisson_prox(value, measured, mu):
    """Positive root of mu*(v-value)+1-y/v=0, preserving upstream arithmetic."""
    z=value-1/mu
    return (z+np.sqrt(z*z+4*measured/mu))/2

def relative_observation_error(abundances, measured, spectra):
    """Interior relative residual after one positive global detector-gain fit.

    Normalized abundance loses its single global scale. This explicit alignment
    applies only to the observation diagnostic, never to reference scoring.
    """
    p=forward(abundances[1:-1,1:-1],spectra).ravel()
    y=np.asarray(measured,dtype=float)[1:-1,1:-1].ravel()
    scale=max(0.,float(p@y/(p@p))) if p@p>0 else 0.
    return float(np.linalg.norm(scale*p-y)/np.linalg.norm(y)),scale
