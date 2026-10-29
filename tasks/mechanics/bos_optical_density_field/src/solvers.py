"""Explicit WLS normal equations and covariance, with active Poisson comparison."""
import numpy as np
from scipy import sparse,linalg
from scipy.sparse.linalg import spsolve
from .physics_model import (calibrate,integration_operators,poisson_operators,perimeter,RHO_AMBIENT,BOUNDARY_SIGMA)

def integrate_wls(x,y,gx,gy,sgx,sgy,*,uniform_weights=False):
    """Return density,sigma,diagnostics; all arrays are2D physical fields."""
    D,B,C=integration_operators(x,y)
    n=x.size;nb=C.shape[1]
    g=np.r_[gx.ravel(),gy.ravel()];var=np.r_[sgx.ravel()**2,sgy.ravel()**2]
    rhs=B@g+C@np.full(nb,RHO_AMBIENT)
    covariance=B@sparse.diags(var)@B.T+C@sparse.diags(np.full(nb,BOUNDARY_SIGMA**2))@C.T
    weights=1/covariance.diagonal()
    if uniform_weights:
        # Homoscedastic-gradient ablation: preserve exact boundary calibration.
        weights[:-nb]=np.mean(weights[:-nb])
    W=sparse.diags(weights,format='csr')
    normal=D.T@W@D;normal_rhs=(D.T@W)@rhs
    density=spsolve(normal,normal_rhs)
    propagated=(D.T@W)@covariance@(D.T@W).T
    inv=linalg.inv(normal.toarray())
    density_covariance=inv@propagated.toarray()@inv.T
    variance=np.diag(density_covariance)
    if not np.isfinite(density).all() or not np.isfinite(variance).all() or np.any(variance<0):raise FloatingPointError('Invalid density/covariance')
    sigma=np.sqrt(variance)
    residual=D@density-rhs
    return density.reshape(x.shape),sigma.reshape(x.shape),dict(D=D,B=B,C=C,rhs=rhs,covariance=covariance,weights=weights,normal=normal,density_covariance=density_covariance,standardized_rms=float(np.sqrt(np.mean(weights*residual**2))))

def integrate_poisson(x,y,gx,gy,sgx,sgy):
    """Direct five-point integration with full Dirichlet perimeter and covariance."""
    A,Hx,Hy,C=poisson_operators(x,y);n=x.size
    rhs=Hx@gx.ravel()+Hy@gy.ravel()+C@np.full(n,RHO_AMBIENT)
    density=spsolve(A,rhs)
    covariance=Hx@sparse.diags(sgx.ravel()**2)@Hx.T+Hy@sparse.diags(sgy.ravel()**2)@Hy.T+C@sparse.diags(np.full(n,BOUNDARY_SIGMA**2))@C.T
    inv=linalg.inv(A.toarray());density_covariance=inv@covariance.toarray()@inv.T
    variance=np.diag(density_covariance)
    if np.any(variance<0) or not np.isfinite(variance).all():raise FloatingPointError('Invalid covariance')
    return density.reshape(x.shape),np.sqrt(variance).reshape(x.shape),dict(A=A,Hx=Hx,Hy=Hy,C=C,rhs=rhs,covariance=covariance,density_covariance=density_covariance)

def reconstruct(observations,method='wls',uniform_weights=False):
    """Compute both whole-field quantities from public observations, no cache."""
    pos=observations['positions'][0];disp=observations['displacement'][0];sigma=observations['displacement_sigma'][0]
    x,y,gx,gy,sgx,sgy=calibrate(pos[...,0],pos[...,1],disp[...,0],disp[...,1],sigma[...,0],sigma[...,1])
    if method=='wls':density,uncertainty,diagnostics=integrate_wls(x,y,gx,gy,sgx,sgy,uniform_weights=uniform_weights)
    elif method=='poisson':density,uncertainty,diagnostics=integrate_poisson(x,y,gx,gy,sgx,sgy)
    else:raise ValueError('Unknown integration method')
    return dict(reconstruction=density[None],density_sigma=uncertainty[None],diagnostics=diagnostics)
