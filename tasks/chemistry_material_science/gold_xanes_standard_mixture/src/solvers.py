"""Native example's eleven bounded MINPACK fits and reduced-chi ranking."""
import numpy as np
from scipy.optimize import leastsq
from .physics_model import bounded_weights,forward,residual
MODELS=((0,1),(0,1,2),(0,1,3),(0,1,4),(0,1,5),(0,1,2,3),(0,1,2,4),(0,1,2,5),(0,1,3,4),(0,1,4,5),(0,1,3,5))
def fit_model(observation,standards,model,initial=None):
    if initial is None:
        initial=np.array([.4,.4]+[0.]*(len(model)-3))[:len(model)-1]
        if model==(0,1,4,5):initial=np.full(3,.3)
    u=np.arcsin(2.*np.asarray(initial)-1.)
    result=leastsq(residual,u,args=(model,observation,standards),full_output=True,
                   ftol=1.5e-8,xtol=1.5e-8,gtol=0.,epsfcn=1e-10,factor=100.,maxfev=4000*(len(u)+1))
    x,cov,info,message,ier=result
    if ier not in (1,2,3,4):raise RuntimeError(message)
    weights=bounded_weights(x,model);fit=forward(weights,standards)
    chi=np.sum(((observation-fit)/.001)**2);n=observation.size;k=len(u)
    aic=n*np.log(max(chi,1e-250)/n)+2*k
    bic=n*np.log(max(chi,1e-250)/n)+np.log(n)*k
    return weights,fit,np.array([chi,chi/(n-k),aic,bic,k,info['nfev']])
def solve(observation,standards):
    results=[fit_model(observation,standards,m) for m in MODELS]
    weights,fits,statistics=map(np.array,zip(*results))
    best=int(np.argmin(statistics[:,1]))
    return dict(weights=weights,fits=fits,statistics=statistics,best=best)
