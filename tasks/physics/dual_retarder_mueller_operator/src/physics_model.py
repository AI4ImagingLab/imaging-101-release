"""Explicit local Mueller optics and intensity-acquisition mathematics.

Adapted from Katsu's MIT implementation; see LICENSE and NOTICE. Matrices act
on column Stokes vectors ordered I,Q,U,V; no truth/normalization is loaded.
"""
import numpy as np
from .preprocessing import ANGULAR_INCREMENT,RETARDANCE,PSG_POLARIZER_ANGLE,PSA_POLARIZER_ANGLE

def linear_polarizer(angle):
    """Ideal polarizer, with physical one-half unpolarized transmission."""
    a=np.asarray(angle,dtype=float);c=np.cos(2*a);s=np.sin(2*a)
    m=np.zeros(a.shape+(4,4),dtype=float)
    m[...,0,0]=1;m[...,0,1]=c;m[...,0,2]=s
    m[...,1,0]=c;m[...,1,1]=c*c;m[...,1,2]=c*s
    m[...,2,0]=s;m[...,2,1]=c*s;m[...,2,2]=s*s
    return m/2

def linear_retarder(angle,retardance):
    """Lossless linear retarder; angle and phase delay are radians."""
    a,r=np.broadcast_arrays(np.asarray(angle,dtype=float),np.asarray(retardance,dtype=float))
    c=np.cos(2*a);s=np.sin(2*a);C=np.cos(r);S=np.sin(r)
    m=np.zeros(a.shape+(4,4),dtype=float);m[...,0,0]=1
    m[...,1,1]=c*c+C*s*s;m[...,1,2]=(1-C)*c*s;m[...,1,3]=-S*s
    m[...,2,1]=m[...,1,2];m[...,2,2]=C*c*c+s*s;m[...,2,3]=c*S
    m[...,3,1]=S*s;m[...,3,2]=-c*S;m[...,3,3]=C
    return m

def acquisition(angles,angular_increment=ANGULAR_INCREMENT,psg_retardance=RETARDANCE,psa_retardance=RETARDANCE):
    """Return PSG, PSA and row-major intensity design matrix (N,16)."""
    theta=np.asarray(angles,dtype=float)
    generator=linear_retarder(theta,psg_retardance)@linear_polarizer(PSG_POLARIZER_ANGLE)
    analyzer=linear_polarizer(PSA_POLARIZER_ANGLE)@linear_retarder(angular_increment*theta,psa_retardance)
    # Measured intensity is row0(PSA) M column0(PSG).
    a=analyzer[:,0,:];g=generator[:,:,0]
    design=(a[:,:,None]*g[:,None,:]).reshape(len(theta),16)
    return generator,analyzer,design

def forward(mueller,angles,angular_increment=ANGULAR_INCREMENT,psg_retardance=RETARDANCE,psa_retardance=RETARDANCE):
    """Dimensionless absolute sample transfer matrix -> relative detector power."""
    _,_,design=acquisition(angles,angular_increment,psg_retardance,psa_retardance)
    return design@np.asarray(mueller,dtype=float).reshape(16)

def adjoint(power,angles):
    """Transpose of the default calibrated intensity operator, shape4×4."""
    return (acquisition(angles)[2].T@np.asarray(power)).reshape(4,4)

def coherency_matrix(mueller):
    """Cloude/Jones coherency diagnostic with the same I,Q,U,V convention."""
    pauli=np.array([[[1,0],[0,1]],[[1,0],[0,-1]],[[0,1],[1,0]],[[0,1j],[-1j,0]]],dtype=complex)
    h=np.zeros((4,4),dtype=complex)
    for i in range(4):
        for j in range(4):h+=np.asarray(mueller)[i,j]*np.kron(pauli[i],pauli[j].conj())/4
    return h
