"""Explicit regularized Gauss-Newton inversion and spline damping quadrature.
Selected mathematics adapted from pymaginverse (MIT, Frenk Out et al.).
"""
import numpy as np
from scipy.interpolate import BSpline
from scipy.integrate import newton_cotes
from scipy.linalg import cho_factor,cho_solve
from .physics_model import harmonic_basis,temporal_basis,observables,observable_jacobian,DEGREE

SPATIAL_DAMPING=1e-13
TEMPORAL_DAMPING=1e-3
ITERATIONS=5

def spline_overlap(first,second,count,step,derivative):
    """Exactly integrate overlapping cubic cardinal splines over the bound interval."""
    order=2*(3-derivative); weights,_=newton_cotes(order)
    b=BSpline.basis_element(np.arange(5)*step,extrapolate=False).derivative(derivative)
    low=max(first,second,3); high=min(first+3,second+3,count-1)
    total=0.
    for t in range(low,high+1):
        total+=np.dot(weights,b(np.linspace(t-first,t-first+1,order+1)*step)*b(np.linspace(t-second,t-second+1,order+1)*step))*step/order
    return total

def damping_matrices(degree=DEGREE,count=43,step=100.):
    """Dense quadratic forms: integrated Ohmic heating (dipole exempt) and acceleration."""
    q=6371.2/3485.; sw=[]; tw=[]
    for n in range(1,degree+1):
        sw.extend([0. if n==1 else 4*np.pi*q**(2*n+3)*(n+1)*(2*n+1)*(2*n+3)/n]*(2*n+1))
        tw.extend([4*np.pi*q**(2*n+4)*(n+1)**2/(2*n+1)]*(2*n+1))
    matrices=[]
    for derivative,weight in [(0,sw),(2,tw)]:
        a=np.zeros((count,count))
        for i in range(count):
            for j in range(i,min(i+4,count)):
                a[i,j]=a[j,i]=spline_overlap(i,j,count,step,derivative)
        matrices.append(np.kron(a,np.diag(weight)))
    return tuple(matrices)

def reconstruct(data,spatial_damping=SPATIAL_DAMPING,temporal_damping=TEMPORAL_DAMPING,iterations=ITERATIONS):
    """data=(time,coords,value,sigma,kind); return coeff(43,15), diagnostics dict."""
    time,coords,value,sigma,kind=data
    basis=harmonic_basis(coords); temporal=temporal_basis(time)
    s,t=damping_matrices(); penalty=spatial_damping*s+temporal_damping*t
    c=np.zeros((temporal.shape[1],basis.shape[2])); c[:,0]=-30000.
    history=[]; models=[]
    for _ in range(iterations):
        p=observables(c,basis,temporal); residual=value-p[np.arange(len(time)),kind]
        residual[kind==6]=(residual[kind==6]+np.pi)%(2*np.pi)-np.pi
        history.append(np.mean(np.abs(residual/sigma)))
        j=observable_jacobian(basis,p,kind)/sigma[:,None]
        jac=np.einsum('ni,nj->nij',temporal,j).reshape(len(time),-1)
        normal=jac.T@jac+penalty
        rhs=jac.T@(residual/sigma)-penalty@c.ravel()
        c+=cho_solve(cho_factor(normal,lower=False),rhs).reshape(c.shape)
        models.append(c.copy())
    predicted=observables(c,basis,temporal)[np.arange(len(time)),kind]
    return c,{'weighted_mae':np.array(history),'iterations':np.array(models),'predicted':predicted}
