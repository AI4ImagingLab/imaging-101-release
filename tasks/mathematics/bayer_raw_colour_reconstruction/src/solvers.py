"""Direct gradient-corrected inverse, Malvar/He/Cutler2004.

Adapted from Colour Developers, Copyright2015, BSD-3-Clause (LICENSE).
All task-specific kernels, sampling conventions and channel assignment are local.
"""
import numpy as np
from scipy.ndimage import convolve
from .physics_model import PATTERN,cfa_masks

def interpolation_kernels():
    """Return green-at-R/B, horizontal, vertical and opposite-colour5x5 kernels."""
    green=np.array([[0,0,-1,0,0],[0,0,2,0,0],[-1,2,4,2,-1],[0,0,2,0,0],[0,0,-1,0,0]],dtype=np.float64)/8
    horizontal=np.array([[0,0,.5,0,0],[0,-1,0,-1,0],[-1,4,5,4,-1],[0,-1,0,-1,0],[0,0,.5,0,0]],dtype=np.float64)/8
    opposite=np.array([[0,0,-1.5,0,0],[0,2,0,2,0],[-1.5,0,6,0,-1.5],[0,2,0,2,0],[0,0,-1.5,0,0]],dtype=np.float64)/8
    return green,horizontal,horizontal.T,opposite

def reconstruct(mosaic,pattern=PATTERN):
    """Return H,W,3 float64 linear camera RGB, signed and unclipped.

    Five-by-five convolutions use half-sample symmetric reflection at each edge,
    exactly as the chosen original implementation. Measured channels stay fixed.
    """
    a=np.asarray(mosaic,dtype=np.float64)
    if a.ndim!=2 or not np.isfinite(a).all():raise ValueError('Invalid scalar mosaic')
    rm,gm,bm=cfa_masks(a.shape,pattern);kg,kh,kv,kd=interpolation_kernels()
    r=a*rm;g=np.where(rm|bm,convolve(a,kg,mode='reflect'),a*gm);b=a*bm
    h=convolve(a,kh,mode='reflect');v=convolve(a,kv,mode='reflect');d=convolve(a,kd,mode='reflect')
    rr=np.any(rm,axis=1)[:,None];rc=np.any(rm,axis=0)[None,:]
    br=np.any(bm,axis=1)[:,None];bc=np.any(bm,axis=0)[None,:]
    r=np.where(rr&bc,h,r);r=np.where(br&rc,v,r);r=np.where(br&bc,d,r)
    b=np.where(br&rc,h,b);b=np.where(rr&bc,v,b);b=np.where(rr&rc,d,b)
    return np.stack([r,g,b],axis=-1)

def bilinear_comparison(mosaic,pattern=PATTERN):
    """Original3x3 bilinear comparator, including its reflective edge effects."""
    a=np.asarray(mosaic,dtype=np.float64);m=cfa_masks(a.shape,pattern)
    hg=np.array([[0,1,0],[1,4,1],[0,1,0]],dtype=np.float64)/4
    hrb=np.array([[1,2,1],[2,4,2],[1,2,1]],dtype=np.float64)/4
    return np.stack([convolve(a*m[i],hg if i==1 else hrb,mode='reflect') for i in range(3)],axis=-1)
