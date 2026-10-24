"""Explicit quantization and vector/tensor finite differences.

Adapted from Kristian Bredies (2013,2025), tgv_pycuda.
Copyright(C)2013,2025 Kristian Bredies. Apache-2.0, see LICENSE/NOTICE.
"""
import numpy as np

def quantization_intervals(image,bins=6):
    """Author's interval observation, including exact grid-boundary samples."""
    return np.floor(image*bins)/bins,np.ceil(image*bins)/bins

def gradient(u):
    """Forward row/column differences; zero outgoing boundary derivatives."""
    result=np.zeros(u.shape+(2,),dtype=u.dtype)
    result[:-1,...,0]=u[1:]-u[:-1]
    result[:,:-1,...,1]=u[:,1:]-u[:,:-1]
    return result

def divergence(p):
    """Negative adjoint of gradient; unused outer-face entries ignored."""
    out=np.zeros(p.shape[:-1],dtype=p.dtype)
    out[:-1]+=p[:-1,...,0];out[1:]-=p[:-1,...,0]
    out[:,:-1]+=p[:,:-1,...,1];out[:,1:]-=p[:,:-1,...,1]
    return out

def symmetric_gradient(w):
    """Symmetric forward derivative; components xx,yy,xy with xy=.5cross."""
    first=gradient(w[...,0]);second=gradient(w[...,1])
    return np.stack([first[...,0],second[...,1],np.float32(.5)*(first[...,1]+second[...,0])],axis=-1)

def symmetric_divergence(q):
    """Negative adjoint under tensor Frobenius weights (1,1,2)."""
    first=divergence(np.stack([q[...,0],q[...,2]],axis=-1))
    second=divergence(np.stack([q[...,2],q[...,1]],axis=-1))
    return np.stack([first,second],axis=-1)

def vector_norm(p):
    return np.sqrt(np.sum(p*p,axis=(-2,-1)))

def tensor_norm(q):
    return np.sqrt(np.sum(q[...,0]**2+q[...,1]**2+2*q[...,2]**2,axis=-1))
