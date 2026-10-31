"""Local focal-contrast surrogate and exact discrete Neumann operators."""
import numpy as np
from scipy.fft import dctn,idctn


def gradient(u):
    """Forward unit-pixel differences with zero terminal component."""
    x=np.zeros_like(u);y=np.zeros_like(u)
    x[:,:-1]=u[:,1:]-u[:,:-1];y[:-1]=u[1:]-u[:-1]
    return x,y


def adjoint(x,y):
    """Exact Euclidean adjoint, including arbitrary unused terminal entries."""
    out=np.zeros_like(x)
    out[:,:-1]-=x[:,:-1];out[:,1:]+=x[:,:-1]
    out[:-1]-=y[:-1];out[1:]+=y[:-1]
    return out


def neumann_eigenvalues(shape,dtype=np.float64):
    """Eigenvalues of K^T K on the native grid with unit pixel pitch."""
    h,w=shape
    a=4*np.sin(np.pi*np.arange(h,dtype=np.float64)/(2*h))**2
    b=4*np.sin(np.pi*np.arange(w,dtype=np.float64)/(2*w))**2
    return (a[:,None]+b[None,:]).astype(dtype)


def solve_neumann(rhs,penalty,eigenvalues=None):
    """Solve (I+penalty K^T K)u=rhs by orthonormal DCT-II/III."""
    if eigenvalues is None:eigenvalues=neumann_eigenvalues(rhs.shape,rhs.dtype)
    transformed=dctn(rhs,type=2,norm='ortho',workers=1)
    transformed/=1+penalty*eigenvalues
    return idctn(transformed,type=2,norm='ortho',workers=1)


def polynomial_value(coefficients,x):
    """Source Horner convention, preserving add-before-multiply operation."""
    out=np.zeros_like(x)
    for c in coefficients[:-1]:out=(out+c)*x
    return out+coefficients[-1]


def objective(coefficients,u,data_fidelity=6.):
    """Negative fitted contrast and isotropic spatial TV, unnormalized sums."""
    x,y=gradient(u.astype(np.float64))
    contrast=polynomial_value(coefficients.astype(np.float64),u.astype(np.float64))
    return {'negative_contrast':float(-contrast.sum()),'tv':float(np.hypot(x,y).sum()),
            'objective':float(-data_fidelity*contrast.sum()+np.hypot(x,y).sum())}
