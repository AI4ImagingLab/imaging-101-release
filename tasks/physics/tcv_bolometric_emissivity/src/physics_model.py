"""Explicit acquisition, anisotropic diffusion and toroidal quadrature.

Adapted from Hamm's pyxu-diffops, MIT license. No domain package imports.
"""
import numpy as np
from scipy import sparse
from .preprocessing import SAMPLING, DZ, DR, R

def forward(matrix,image):
    return matrix @ np.asarray(image).reshape(-1)

def adjoint(matrix,values,shape=(120,41)):
    return (matrix.T @ values).reshape(shape)

def gradient_matrix(shape,sampling=SAMPLING):
    """Forward differences, symmetric exterior boundary; component-major rows."""
    nz,nr=shape
    def d(n):
        x=sparse.diags([-np.ones(n),np.ones(n-1)],[0,1],format='lil')/sampling
        x[-1,-1]=0
        return x.tocsr()
    return sparse.vstack((sparse.kron(d(nz),sparse.eye(nr)),sparse.kron(sparse.eye(nz),d(nr))),format='csr')

def diffusion_tensor(flux,alpha=.01,sampling=SAMPLING):
    """Central differences with symmetric padding; eigenvalues (alpha,1)."""
    p=np.pad(flux,((1,1),(1,1)),mode='symmetric')
    gz=(p[2:,1:-1]-p[:-2,1:-1])/(2*sampling)
    gr=(p[1:-1,2:]-p[1:-1,:-2])/(2*sampling)
    v=np.stack((gz,gr),axis=-1)
    s=v[...,None]*v[...,None,:]
    u,_,_=np.linalg.svd(s,hermitian=True)
    w=(u*np.array([alpha,1.])) @ np.swapaxes(u,-1,-2)
    return np.moveaxis(w,(-2,-1),(0,1))

def diffusion_matrix(tensor,gradient):
    n=tensor.shape[-2]*tensor.shape[-1]
    w=sparse.diags([np.concatenate((tensor[0,0].ravel(),tensor[1,1].ravel())),tensor[0,1].ravel(),tensor[1,0].ravel()],[0,n,-n],format='csr')
    return (gradient.T @ w @ gradient).tocsr()

def loss_gradient(x,matrix,y,laplacian,sigma=.025,regularization=10.):
    """Half squared weighted residual plus half diffusion quadratic."""
    flat=np.asarray(x).ravel()
    residual=matrix@flat-y
    lx=laplacian@flat
    value=np.dot(residual,residual)/(2*sigma*sigma)+regularization*np.dot(flat,lx)/2
    grad=(matrix.T@residual)/(sigma*sigma)+regularization*lx
    return float(value),grad.reshape(np.shape(x))

def integrate_power(images,region=None):
    """Toroidal symmetry: emissivity W/m^3 -> power W."""
    weights=2*np.pi*R[None,:]*DR*DZ
    if region is not None:
        weights=weights*region
    return np.sum(np.asarray(images)*weights,axis=(-2,-1))
