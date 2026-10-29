"""Explicit wavelength interpolation, diffraction mixing and constrained spectral model."""
import numpy as np
from scipy.linalg import block_diag

EPS=1e-12

def interpolation_matrix(query,grid):
    """Piecewise-linear interpolation with linear endpoint extrapolation, original convention."""
    query=np.asarray(query);grid=np.asarray(grid)
    i=np.clip(np.searchsorted(grid,query)-1,0,len(grid)-2)
    d=grid[i+1]-grid[i]
    m=np.zeros((len(query),len(grid)))
    m[np.arange(len(query)),i]=(grid[i+1]-query)/d
    m[np.arange(len(query)),i+1]=(query-grid[i])/d
    return m

def fourier_basis(n,degree=7):
    """Constant,sin1..sinK,cos1..cosK on inclusive0..n-1 grid; odd degree."""
    if degree%2!=1 or degree<1:raise ValueError('Fourier degree must be positive odd')
    angles=2*np.pi*np.arange(n)/(n-1)
    phases=angles[:,None]*np.arange(1,degree//2+1)[None,:]
    return np.column_stack((np.ones(n),np.sin(phases),np.cos(phases)))

def wavelength_kernel(mapping,n):
    """Linear-splat rows on spectral-bin coordinates; original half-bin upper support."""
    mapping=np.asarray(mapping);hi=np.clip(np.ceil(mapping).astype(int),0,n-1)
    lo=np.clip(np.floor(mapping).astype(int),0,n-1)
    kernel=np.zeros((len(mapping),n));valid=(mapping>=0)&(mapping<=n-.5)
    same=valid&(hi==lo);rows=np.nonzero(same)[0];kernel[rows,hi[same]]=1
    rows=np.nonzero(valid&~same)[0]
    kernel[rows,hi[rows]]=mapping[rows]-lo[rows]
    kernel[rows,lo[rows]]=hi[rows]-mapping[rows]
    return kernel

def predicted_observations(sensitivity,attenuation,illumination,kernel,interpolation):
    """Dimensionless direct RGB sum and dispersed RGB; nuisance attenuation may be signed."""
    spectrum=interpolation@sensitivity
    return illumination@spectrum,kernel@(illumination[:,None]*attenuation[:,None]*spectrum)

def spectral_system(direct,dispersed,illumination,kernel,interpolation,bases,attenuation_basis):
    """Build normalized equality-constrained normal system exactly as published Supplement F (upstream comment labels it E)."""
    modes=[interpolation@b for b in bases]
    inverse=np.linalg.pinv(kernel@np.diag(illumination))
    products=np.array([inverse@dispersed[:,c] for c in range(3)])
    mixed=np.vstack([np.diag(products[c])@attenuation_basis for c in range(3)])
    residual=np.hstack((mixed,-block_diag(*modes)))
    residual/=np.linalg.norm(residual,axis=1)[:,None]+EPS
    constraint=block_diag(*[illumination[None,:]@b for b in modes])
    target=direct.copy()
    norm=np.linalg.norm(np.column_stack((constraint,target)),axis=1)+EPS
    constraint/=norm[:,None];target/=norm
    constraint=np.hstack((np.zeros((3,attenuation_basis.shape[1])),constraint))
    matrix=np.block([[residual.T@residual,constraint.T],[constraint,np.zeros((3,3))]])
    rhs=np.r_[np.zeros(matrix.shape[0]-3),target]
    rownorm=np.linalg.norm(np.column_stack((matrix,rhs)),axis=1)+EPS
    return dict(matrix=matrix/rownorm[:,None],rhs=rhs/rownorm,residual=residual,constraint=constraint,target=target,products=products)
