"""Explicit isotropic Smith eigenoperator and radial slope quadrature.

Adapted from Jonathan Dupuy's public-domain dj_brdf (EGSR 2015 branch).
No Fresnel inversion or general BRDF rendering is performed.
"""
import numpy as np
def build_kernel(theta,intensity):
    """89x89 positive operator, preserving original floating azimuth loop."""
    theta=np.asarray(theta); intensity=np.asarray(intensity)
    t=np.arange(89)/89*np.sqrt(np.pi/2)
    slope=np.tan(theta); c=np.cos(theta)
    product=slope[:,None]*slope[None,:]
    integral=np.zeros((89,89)); phi=0.; dphi=np.pi/180
    while phi<2*np.pi:
        integral+=np.maximum(1.,product*np.cos(phi)); phi+=dphi
    integral*=dphi
    return ((np.sqrt(np.pi/2)/89*c**6)*(8*intensity))[:,None]*integral*(t*slope/c**2)[None,:]
def quadrature(pdf):
    """128 left samples in sqrt(angle); return slope, weighted radial mass."""
    t=np.arange(128)/128*np.sqrt(np.pi/2); theta=t*t
    slope=np.tan(theta)
    x=np.sqrt(2*np.arctan(np.sqrt(slope*slope))/np.pi)
    density=np.interp(x,np.linspace(0,1,90),pdf)
    weight=t*density*slope/np.cos(theta)**2*(2*np.sqrt(np.pi/2)/128)
    return slope,weight
def mass(pdf):
    return float(2*np.pi*np.sum(quadrature(pdf)[1]))
def roughness(pdf):
    """Finite-quadrature moments: Beckmann sqrt(2 E[sx^2]), GGX E[abs(sx)].

    These are 128-point finite-domain summaries, not convergent infinite-slope moments.
    """
    slope,w=quadrature(pdf)
    return np.array([np.sqrt(2*np.pi*np.sum(slope**2*w)),4*np.sum(slope*w)])
def eigen_residual(kernel,pdf):
    p=np.asarray(pdf)[:89]; kp=kernel@p
    eigenvalue=float(p@kp/(p@p))
    return float(np.linalg.norm(kp-eigenvalue*p)/np.linalg.norm(kp)),eigenvalue
def parametric_pdf(slope,alpha,family):
    if family=='beckmann': return np.exp(-(slope/alpha)**2)/(np.pi*alpha**2)
    if family=='ggx': return alpha**2/(np.pi*(alpha**2+slope**2)**2)
    raise ValueError('Unknown family')
