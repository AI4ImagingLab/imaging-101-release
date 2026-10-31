import time
import numpy as np
from scipy.optimize import leastsq
from .physics_model import INITIAL,BOUNDS,hankel,bounded_residuals,residuals,autocovariance
def reconstruct(data,initial=INITIAL):
    """CPU MINPACK LM with the original Bumps bound treatment and settings."""
    start=time.perf_counter();q,H,H0=hankel(data)
    p,cov,info,msg,flag=leastsq(bounded_residuals,np.array(initial,dtype=float),args=(data,q,H,H0),ftol=1.5e-8,xtol=1.5e-8,epsfcn=1e-8,maxfev=600,full_output=True)
    if flag not in (1,2,3,4) or not np.isfinite(p).all():raise RuntimeError(msg)
    p=np.clip(p,BOUNDS[:,0],BOUNDS[:,1])
    return dict(reconstruction=autocovariance(p)[None],parameters=p[None]),dict(seconds=time.perf_counter()-start,nfev=int(info['nfev']),success=True,message=msg,chisqr=float(np.sum(residuals(p,data,q,H,H0)**2)))
