"""Explicit GP covariance/derivative operators and cosmic acceleration relation.

Mathematics adapted from GaPP (Marina Seikel2012/2013), GPL3-or-later.
No Gaussian-process or cosmological domain library is imported.
"""
import numpy as np
from numpy.polynomial.polynomial import polyval,polyder

KERNELS=('SquaredExponential','Matern52','Matern72','Matern92')

def covariance_operators(x,y,theta,kernel='SquaredExponential'):
    """Return K,dK/dtheta,K01,K11; input coordinates are dimensionless redshifts.

    theta=(amplitudeH,lengthz); gradient shape(2,Nx,Ny), other shapes(Nx,Ny).
    K01 differentiates second argument; K11 differentiates both arguments.
    """
    amplitude,length=np.asarray(theta,dtype=np.float64)
    delta=np.asarray(x,dtype=np.float64)[:,None]-np.asarray(y,dtype=np.float64)[None,:]
    r=np.abs(delta)/length
    if kernel=='SquaredExponential':
        value=amplitude**2*np.exp(-r*r/2)
        gradient=np.stack([2*value/amplitude,value*r*r/length])
        derivative=delta/length**2*value
        mixed=(1-r*r)/length**2*value
        return value,gradient,derivative,mixed
    coefficients={
        'Matern52':(np.sqrt(5),[1,np.sqrt(5),5/3]),
        'Matern72':(np.sqrt(7),[1,np.sqrt(7),14/5,7*np.sqrt(7)/15]),
        'Matern92':(3.,[1,3.,27/7,18/7,27/35])}
    decay,polynomial=coefficients[kernel]
    base=amplitude**2*np.exp(-decay*r)
    p=polyval(r,polynomial);first=polyval(r,polyder(polynomial));second=polyval(r,polyder(polynomial,2))
    value=base*p;radial=base*(first-decay*p)
    gradient=np.stack([2*value/amplitude,-r/length*radial])
    derivative=-np.sign(delta)/length*radial
    mixed=-base*(second-2*decay*first+decay**2*p)/length**2
    return value,gradient,derivative,mixed

def acceleration_parameter(redshift,hubble,hubble_derivative):
    """q(z)=(1+z)H'(z)/H(z)-1, from the expansion-rate differential equation."""
    return (1+np.asarray(redshift))*np.asarray(hubble_derivative)/np.asarray(hubble)-1

def q_delta_standard_deviation(redshift,hubble,hubble_derivative,point_covariance):
    """Per-point delta-method diagnostic, not a cross-redshift confidence band."""
    z=np.asarray(redshift);h=np.asarray(hubble);hp=np.asarray(hubble_derivative)
    jacobian=np.stack([-(1+z)*hp/h**2,(1+z)/h],axis=-1)
    variance=np.einsum('...i,...ij,...j->...',jacobian,point_covariance,jacobian)
    return np.sqrt(variance)
