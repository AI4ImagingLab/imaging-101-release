"""Closed-form Gaussian posterior mean with explicit task precision."""
import numpy as np
from scipy.sparse.linalg import splu
from .physics_model import operators,center_stress,interpolation
LAMBDA=1e-6
ASSUMED_NOISE=0.001
def objective_gradient(vector,a,b,traction,lam=LAMBDA,spacing=2.):
    """Unhalved scaled quadratic and analytic gradient; vector uses face order."""
    residual=a@vector-traction
    return float(spacing**2*np.dot(residual,residual)+lam*np.dot(vector,b@vector)),2*(spacing**2*(a.T@residual)+lam*(b@vector))
def reconstruct(traction,lam=LAMBDA,alpha_shear=1000.,alpha_boundary=1000.):
    """Return central stress (3,R,C) and full face/posterior precision diagnostics."""
    rows,columns=traction.shape[-2:];spacing=2.
    a,b,d,e=operators(rows,columns,spacing,alpha_shear,alpha_boundary)
    data=traction.reshape(-1)
    h=(lam*b+spacing**2*(a.T@a)).tocsc()
    factor=splu(h)
    rhs=spacing**2*(a.T@data);vector=factor.solve(rhs)
    objective,gradient=objective_gradient(vector,a,b,data,lam,spacing)
    initial,_=objective_gradient(np.zeros_like(vector),a,b,data,lam,spacing)
    state=dict(a=a,b=b,d=d,e=e,h=h,factor=factor,vector=vector,traction=data,prediction=(a@vector).reshape(2,rows,columns),objective=objective,initial_objective=initial,gradient=gradient,lambda_value=lam,spacing=spacing,rows=rows,columns=columns)
    return center_stress(vector,rows,columns),state
def posterior_diagnostics(state,indices=None,noise=ASSUMED_NOISE,batch_size=32):
    """Compute exact central variances L C L^T diagonals using batched linear solves.

    Return exact variance, source diagonal-only approximation, sampled face C and
    face C diagonal; C=noise²*spacing²*H^-1. No dense full covariance is allocated.
    """
    factor=state["factor"];n=state["h"].shape[0];scale=noise**2*state["spacing"]**2
    diag=np.empty(n)
    for first in range(0,n,batch_size):
        ids=np.arange(first,min(first+batch_size,n));rhs=np.zeros((n,len(ids)));rhs[ids,np.arange(len(ids))]=1.
        solution=factor.solve(rhs)
        diag[ids]=solution[ids,np.arange(len(ids))]*scale
    l=interpolation(state["rows"],state["columns"])
    exact=np.empty(l.shape[0])
    for first in range(0,l.shape[0],batch_size):
        block=l[first:first+batch_size];rhs=block.T.toarray()
        solution=factor.solve(rhs)
        exact[first:first+len(rhs.T)]=np.asarray(block.multiply(solution.T).sum(axis=1)).ravel()*scale
    approx=np.asarray(l.power(2)@diag).reshape(3,state["rows"],state["columns"])
    if indices is None:indices=np.array([0,1,n//2,n-1])
    rhs=np.zeros((n,len(indices)));rhs[indices,np.arange(len(indices))]=1
    sample=factor.solve(rhs)[indices]*scale
    return dict(exact_variance=exact.reshape(approx.shape),diagonal_only_variance=approx,covariance_samples=sample,face_covariance_diagonal=diag)
