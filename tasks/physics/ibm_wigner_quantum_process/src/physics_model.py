"""Single-qubit DROPS basis, calibrated ideal acquisition, and SU(2) algebra.
Adapted from Amit Devra's Wigner-Unknown-Process-Tomography, LGPL-3.0.
"""
import numpy as np

def basis(theta, phi):
    """Real normalized degree 0/1 spherical basis, shape(4,N), order X,Y,Z,I."""
    xyz = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
    return np.vstack([np.sqrt(3/(32*np.pi))*xyz, np.ones_like(theta)/np.sqrt(32*np.pi)])

def quaternion_matrix(q):
    """Map real quaternion(A,B,C,D) to complex SU(2) matrix(2,2)."""
    a,b,c,d = q
    return np.array([[d+1j*c,b+1j*a],[-b+1j*a,d-1j*c]])

def matrix_quaternion(u):
    """Project coefficients onto the real quaternion subspace; no normalization."""
    return np.array([u[0,1].imag,u[0,1].real,u[0,0].imag,u[0,0].real])

def euler_quaternion(x):
    """Author Euler parameterization; radians, output(A,B,C,D)."""
    alpha,beta,gamma=x
    return np.array([-np.sin(beta/2)*np.sin((alpha-gamma)/2),np.sin(beta/2)*np.cos((alpha-gamma)/2),np.cos(beta/2)*np.sin((alpha+gamma)/2),np.cos(beta/2)*np.cos((alpha+gamma)/2)])

def ideal_expectation(q, theta, phi):
    """Ideal two-rank interferometric expectation(2,4,N), order I,X,Y,Z.
    Physical device attenuation/errors remain in measured observations; this is
    an ideal-model diagnostic, not an invented perfect-measurement replacement.
    """
    xyz=np.array([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)])
    v=q[:3]@xyz; d=q[3]
    e=np.empty((2,4,len(theta)),complex)
    e[0,0]=d*d; e[1,0]=1j*d*v
    e[0,1:]=-1j*q[:3,None]*d
    e[1,1:]=q[:3,None]*v
    return e

def overlap_coefficients(f, b, weights):
    """Weighted basis projection with author's squared-norm normalization."""
    return (f*weights) @ b.conj().T / np.sum(weights*np.abs(b)**2,axis=1)

def coefficients_matrix(c):
    """Pauli expansion coefficients(cX,cY,cZ,cI) -> complex matrix."""
    x,y,z,i=c
    return np.array([[i+z,x-1j*y],[x+1j*y,i-z]])
