"""Local nonnegative transformed Levenberg-Marquardt network inversion."""
import numpy as np
from scipy.optimize import leastsq
from .physics_model import concentrations,charge

def relative_residual(observed, predicted):
    """Author normalization (observed-predicted)/(observed+predicted)."""
    norm = observed + predicted
    return np.divide(observed-predicted,norm,out=np.zeros_like(norm),where=norm!=0)

def residual(rates, c, q, initial):
    fit = concentrations(c[:,0],initial,rates)
    rc = relative_residual(c[:,1:],fit[:,:5]).T.ravel()
    if q is None:
        return rc
    fitq = charge(concentrations(q[:,0],initial,rates))
    rq = relative_residual(q[:,1],fitq)
    return np.r_[rc,rq]

def reconstruct(c,q,initial):
    """Original lmfit sqrt bound transform with MINPACK leastsq defaults.

    Positive rates x=sqrt(z*z+1)-1. Six uniform starts are an explicit
    reliability extension accompanying the source electron-count correction.
    The original lmfit sets epsfcn=1e-10, ftol=xtol=1.5e-8.
    """
    transform=lambda z: np.sqrt(z*z+1)-1
    candidates=[]
    for value in [.005,.01,.025,.05,.1,.2]:
        z0=np.full(11,np.sqrt((1+value)**2-1))
        candidate=leastsq(lambda z:residual(transform(z),c,q,initial),z0,
            full_output=True, ftol=1.5e-8,xtol=1.5e-8,gtol=0,epsfcn=1e-10,maxfev=24000)
        candidates.append(candidate)
    z,cov,info,message,status=min(candidates,key=lambda item:np.sum(item[2]['fvec']**2))
    rates=transform(z)
    time=np.linspace(0,64,150)
    curves=concentrations(time,initial,rates)
    return {'rates':rates[None], 'reconstruction':curves[None],
            'charge':charge(curves)[None]}, {'nfev':info['nfev'],'status':int(status),
            'message':message,'objective':float(np.sum(info['fvec']**2))}
