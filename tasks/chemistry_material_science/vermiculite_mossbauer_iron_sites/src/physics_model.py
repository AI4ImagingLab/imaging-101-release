"""Explicit conventional-source nuclear transmission integral for powder doublets.

Selected isotropic, zero-Gaussian-width branch of SYNCmoss (MIT, ESRF).
The complex2x2 powder cross-section is scalar: its dispersive phase cancels
in the unpolarized transmitted intensity, leaving exp(-real optical depth).
"""
import numpy as np
NATURAL_FWHM=.098
SCALE=.28

def lorentzian(fwhm,offset):
    """Unit-area line density in inverse scaled velocity units."""
    return fwhm/2/np.pi/(offset**2+(fwhm/2)**2)

def source_quadrature(nodes=64):
    coordinate=np.linspace(-.989,.989,nodes)
    coefficients=np.array([-.00317686,20.7572,-30.7854]) if nodes<56 else np.array([.027181,22.7718,-33.6937])
    offset=sum(coefficients[j-1]*np.log((j+coordinate)/(j-coordinate)) for j in [1,2,3])
    jacobian=sum(coefficients[j-1]*(2*j)/(j*j-coordinate**2) for j in [1,2,3])
    weights=lorentzian(NATURAL_FWHM*SCALE,offset)*jacobian*(coordinate[1]-coordinate[0])
    return offset,weights

def transmission(velocity,parameters,nodes=64):
    """Source parameter layout:8 baseline slots then9 slots per doublet.

    Selected free slots per doublet: effective optical thickness I, isomer
    shift delta, quadrupole half-splitting epsilon, absorber Lorentz FWHM.
    All velocities and line widths are mm/s; source quantities are scaled.
    """
    velocity=np.asarray(velocity);p=np.asarray(parameters)
    offsets,weights=source_quadrature(nodes)
    result=np.zeros_like(velocity)
    # Preserve source energy-node summation order, not a BLAS reduction.
    for offset,weight in zip(offsets,weights):
        energy=SCALE*velocity+offset;depth=np.zeros_like(velocity)
        for start in range(8,len(p),9):
            intensity=abs(p[start]);delta,epsilon=p[start+1:start+3];width=abs(p[start+3])*SCALE
            first=lorentzian(width,energy-(delta-epsilon)*SCALE)
            second=lorentzian(width,energy-(delta+epsilon)*SCALE)
            depth+=np.pi*(NATURAL_FWHM/2*SCALE)*(.5*intensity)*(first+second)
        result+=np.exp(-depth)*weight
    # Add photons in the natural source tails outside the finite quadrature.
    result+=1-np.sum(weights)
    baseline=p[0]+p[3]*p[0]/100*(velocity-p[1])+p[2]*p[0]/10000*(velocity-p[1])**2
    background=p[4]+p[7]*p[4]/100*(velocity-p[5])+p[6]*p[4]/10000*(velocity-p[5])**2
    return baseline*result+background
