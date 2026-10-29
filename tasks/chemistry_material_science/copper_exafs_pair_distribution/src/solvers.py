import time
import numpy as np
from scipy.optimize import leastsq
from .physics_model import model_grid,scattering_splines,INITIAL,residuals,pair_density,rfactor
def reconstruct(data,initial=INITIAL):
    """Unconstrained MINPACK LM with exactly the upstream scalar residual conventions."""
    start=time.perf_counter();k=model_grid(data);chi=np.interp(k,data['k'],data['chi']);spl=scattering_splines(data)
    p,cov,info,msg,flag=leastsq(residuals,np.array(initial,dtype=float),args=(k,chi,spl),full_output=True,ftol=1e-6,xtol=1e-6,gtol=1e-6,epsfcn=1e-10,maxfev=10000)
    if flag not in (1,2,3,4) or not np.isfinite(p).all():raise RuntimeError(msg)
    result=dict(reconstruction=pair_density(p)[None],parameters=p[None])
    return result,dict(seconds=time.perf_counter()-start,nfev=int(info['nfev']),success=True,message=msg,rfactor=rfactor(p,k,chi,spl))
