"""Explicit normal-gradient operators, adapted from GPL-3.0 BiNI.

Copyright (C) 2022 Xu Cao. Local extraction of the validated readable CPU
implementation: orthographic branch only, without rendering/domain libraries.
"""
import numpy as np
from scipy.sparse import csr_matrix,vstack


def move_left(mask):return np.pad(mask,((0,0),(0,1)),constant_values=0)[:,1:]
def move_right(mask):return np.pad(mask,((0,0),(1,0)),constant_values=0)[:,:-1]
def move_up(mask):return np.pad(mask,((0,1),(0,0)),constant_values=0)[1:,:]
def move_down(mask):return np.pad(mask,((1,0),(0,0)),constant_values=0)[:-1,:]


def derivative_matrices(mask,nz_horizontal,nz_vertical,step_size=1):
    """Right/left/up/down weighted one-sided differences; missing rows zero.

    Inputs: mask(H,W), normal-z vectors(n,), positive step in pixel units.
    Each return is a sparse CSR(n,n). Positive up means decreasing row.
    """
    if step_size<=0:raise ValueError('Positive step required')
    count=int(mask.sum());index=np.zeros(mask.shape,dtype=int);index[mask]=np.arange(count)
    identity=lambda m:m
    def make(has,nz,current,neighbor):
        value=nz[has[mask]]
        data=np.stack([-value/step_size,value/step_size],-1).ravel()
        cols=np.stack([index[current(has)],index[neighbor(has)]],-1).ravel()
        ptr=np.concatenate([[0],np.cumsum(has[mask].astype(int)*2)])
        return csr_matrix((data,cols,ptr),shape=(count,count))
    return (make(mask&move_left(mask),nz_horizontal,identity,move_right),
            make(mask&move_right(mask),nz_horizontal,move_left,identity),
            make(mask&move_down(mask),nz_vertical,identity,move_up),
            make(mask&move_up(mask),nz_vertical,move_down,identity))


def system(normal,mask,step_size=1):
    """Return A(4n,n), b(4n,), four operators; z grows into scene."""
    nx=normal[mask,1];ny=normal[mask,0];nz=-normal[mask,2]
    ops=derivative_matrices(mask,nz,nz,step_size)
    right,left,up,down=ops
    return vstack([up,down,right,left]),np.concatenate([-nx,-nx,-ny,-ny]),ops


def weights(depth,operators,k):
    """Original sigmoid gradient-contrast IRLS weights, not residual weights.

    Return stacked weights and horizontal/vertical maps as compact vectors.
    Overflow of exp represents a saturated 0/1 logistic, not missing values.
    The original arithmetic is retained for numerical parity.
    """
    right,left,up,down=operators
    gu,gd,gr,gl=up@depth,down@depth,right@depth,left@depth
    with np.errstate(over='ignore'):
        vertical=1/(1+np.exp(-k*(gd**2-gu**2)))
        horizontal=1/(1+np.exp(-k*(gl**2-gr**2)))
    return np.concatenate([vertical,1-vertical,horizontal,1-horizontal]),horizontal,vertical
