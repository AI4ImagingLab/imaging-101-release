"""Local finite-difference acquisition and weighted symmetric derivatives.

Adapted from Habring/Holler matpy.py, GPL-3.0-or-later; see LICENSE.
Image coordinates are row,column with unit pixel spacing.
"""
import numpy as np

def acquisition(image, mask):
    """Known binary sampling, same spatial shape and dimensionless scale."""
    return np.where(mask, image, 0.0)

def gradient(x):
    out=np.zeros(x.shape+(2,),dtype=np.float64)
    out[:-1,:,0]=x[1:,:]-x[:-1,:]
    out[:,:-1,1]=x[:,1:]-x[:,:-1]
    return out

def gradient_adjoint(p):
    x=np.zeros(p.shape[:2],dtype=np.float64)
    x[0,:]=p[0,:,0]
    x[-1,:]=-p[-2,:,0]
    x[1:-1,:]=p[1:-1,:,0]-p[:-2,:,0]
    x[:,0]+=p[:,0,1]
    x[:,-1]+=-p[:,-2,1]
    x[:,1:-1]+=p[:,1:-1,1]-p[:,:-2,1]
    return -x

def symgrad(v):
    out=np.zeros(v.shape[:2]+(3,),dtype=np.float64)
    out[1:,:,0]=v[1:,:,0]-v[:-1,:,0]
    out[:,1:,1]=v[:,1:,1]-v[:,:-1,1]
    out[1:,:,2]=v[1:,:,1]-v[:-1,:,1]
    out[:,1:,2]+=v[:,1:,0]-v[:,:-1,0]
    out[:,:,2]*=.5
    return out

def _negative_backward_adjoint(p,axis):
    out=np.zeros_like(p)
    if axis==0:
        out[0,:]=p[1,:]
        out[-1,:]=-p[-1,:]
        out[1:-1,:]=p[2:,:]-p[1:-1,:]
    else:
        out[:,0]=p[:,1]
        out[:,-1]=-p[:,-1]
        out[:,1:-1]=p[:,2:]-p[:,1:-1]
    return out

def symgrad_adjoint(q):
    """Adjoint under tensor inner product weights (1,1,2)."""
    x=np.zeros(q.shape[:2]+(2,),dtype=np.float64)
    x[:,:,0]=_negative_backward_adjoint(q[:,:,0],0)
    x[:,:,0]+=_negative_backward_adjoint(q[:,:,2],1)
    x[:,:,1]=_negative_backward_adjoint(q[:,:,2],0)
    x[:,:,1]+=_negative_backward_adjoint(q[:,:,1],1)
    return -x

def symgrad2(w):
    out=np.zeros(w.shape[:2]+(4,),dtype=np.float64)
    out[1:,:,0]=w[1:,:,0]-w[:-1,:,0]
    out[:,1:,1]=w[:,1:,1]-w[:,:-1,1]
    out[1:,:,2]=2*(w[1:,:,2]-w[:-1,:,2])
    out[:,1:,2]+=w[:,1:,0]-w[:,:-1,0]
    out[:,:,2]/=3.0
    out[1:,:,3]=w[1:,:,1]-w[:-1,:,1]
    out[:,1:,3]+=2*(w[:,1:,2]-w[:,:-1,2])
    out[:,:,3]/=3.0
    return out

def symgrad2_adjoint(r):
    """Adjoint from weights (1,1,3,3) to weights (1,1,2)."""
    x=np.zeros(r.shape[:2]+(3,),dtype=np.float64)
    x[:,:,0]=_negative_backward_adjoint(r[:,:,0],0)
    x[:,:,0]+=_negative_backward_adjoint(r[:,:,2],1)
    x[:,:,1]=_negative_backward_adjoint(r[:,:,1],1)
    x[:,:,1]+=_negative_backward_adjoint(r[:,:,3],0)
    x[:,:,2]=_negative_backward_adjoint(r[:,:,2],0)
    x[:,:,2]+=_negative_backward_adjoint(r[:,:,3],1)
    return -x

