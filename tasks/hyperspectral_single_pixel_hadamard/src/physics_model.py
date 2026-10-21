"""Explicit unnormalized two-dimensional Walsh sensing and its exact inverse.

Walsh ordering follows SPYRIT, Sebastien Crombez and Nicolas Ducros (GPLv3).
Local butterfly arithmetic replaces the dense upstream matrix without changing
the published measurement operator, sequency, C-order axes or normalization.
"""
import numpy as np

def sequency_order(n):
    """Natural Sylvester row indices, Gray-coded then bit-reversed."""
    if n<1 or n&(n-1):raise ValueError('Walsh dimension must be a power of two')
    bits=n.bit_length()-1
    return np.array([int(f'{i^(i>>1):0{bits}b}'[::-1],2) if bits else 0 for i in range(n)],dtype=np.int64)

def walsh_matrix(n):
    """Return (n,n) real +/-1 sequency rows for independent small-matrix use."""
    order=sequency_order(n)
    return np.array([[1.-2.*((int(i)&j).bit_count()%2) for j in range(n)] for i in order])

def _natural_transform(a,axis):
    """In-place binary butterflies on a private copy along the selected axis."""
    out=np.moveaxis(np.asarray(a,dtype=np.float64),axis,0).copy()
    n=out.shape[0];sequency_order(n)
    width=1
    while width<n:
        blocks=out.reshape((n//(2*width),2,width)+out.shape[1:])
        left=blocks[:,0].copy();right=blocks[:,1].copy()
        blocks[:,0]=left+right;blocks[:,1]=left-right
        width*=2
    return np.moveaxis(out,0,axis)

def forward(cube):
    """cube (y,x,channel) -> unnormalized Walsh coefficients, same shape."""
    if np.ndim(cube)!=3 or cube.shape[0]!=cube.shape[1]:raise ValueError('Expected square spatial cube')
    order=sequency_order(cube.shape[0])
    natural=_natural_transform(_natural_transform(cube,0),1)
    return np.take(np.take(natural,order,axis=0),order,axis=1)

def inverse(coefficients):
    """Walsh coefficients -> counts/pixel, dividing by total spatial pixels."""
    if np.ndim(coefficients)!=3 or coefficients.shape[0]!=coefficients.shape[1]:raise ValueError('Expected square coefficient cube')
    n=coefficients.shape[0];undo=np.argsort(sequency_order(n))
    natural=np.take(np.take(coefficients,undo,axis=0),undo,axis=1)
    return _natural_transform(_natural_transform(natural,0),1)/(n*n)
