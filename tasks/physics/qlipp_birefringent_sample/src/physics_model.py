"""Explicit Stokes/Mueller equations; waveorder BSD-3-Clause adaptation."""
import torch
import numpy as np

def apply_matrix(matrix,vector):
    """Matrix indices lead; trailing acquisition/spatial axes broadcast."""
    return torch.einsum('NM...,M...->N...',matrix,vector)

def stokes_forward(retardance,orientation,transmittance,depolarization):
    """Circular incident light, declared detector slow-axis convention."""
    d,t,a,p=retardance,transmittance,orientation,depolarization
    return torch.stack((t,t*p*torch.sin(d)*torch.sin(2*a),-t*p*torch.sin(d)*torch.cos(2*a),t*p*torch.cos(d)))

def mueller_matrix(stokes):
    """Original ADR Mueller formula, with explicit float32 matrix storage."""
    s0,s1,s2,s3=stokes;denom=s1*s1+s2*s2
    M=torch.zeros((4,4)+s0.shape,dtype=torch.float32)
    M[0,0]=s0;M[1,1]=(s0*s2*s2+s1*s1*s3)/denom;M[1,2]=s1*s2*(s3-s0)/denom
    M[1,3]=s1;M[2,1]=M[1,2];M[2,2]=(s0*s1*s1+s2*s2*s3)/denom
    M[2,3]=s2;M[3,1]=-s1;M[3,2]=-s2;M[3,3]=s3
    return M

def inverse_mueller(stokes):
    """Invert each measured-background 4x4 Mueller matrix."""
    M=mueller_matrix(stokes)
    return torch.moveaxis(torch.linalg.inv(torch.moveaxis(M,(0,1),(-2,-1))),(-2,-1),(0,1))

def estimate_parameters(stokes):
    """ADR analytic inverse: radians, radians, relative intensity, unitless."""
    s0,s1,s2,s3=stokes;length=torch.sqrt(s1*s1+s2*s2+s3*s3)
    return (torch.asin(torch.sqrt(s1*s1+s2*s2)/length),
            torch.remainder(torch.atan2(s1,-s2),2*np.pi)/2,s0.clone(),length/s0)
