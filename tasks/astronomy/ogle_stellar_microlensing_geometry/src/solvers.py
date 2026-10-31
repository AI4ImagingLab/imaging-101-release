"""Variable projection with the original Nelder-Mead search."""
import numpy as np
from scipy.optimize import minimize
from .preprocessing import flux_from_magnitude
from .physics_model import magnification
INITIAL=np.array([5380.,.5,25.])
def fit_fluxes(amplification,flux,error,blend=True):
    design=np.array([amplification,np.ones_like(amplification)]).T if blend else amplification[:,None]
    weights=1./error
    coefficients=np.linalg.lstsq(design*weights[:,None],flux*weights,rcond=-1)[0]
    if not blend:coefficients=np.r_[coefficients,0.]
    model=coefficients[0]*amplification+coefficients[1]
    return coefficients,model
def profile_objective(parameters,time,flux,error,blend=True):
    if parameters[2]<=0:return np.inf
    _,model=fit_fluxes(magnification(time,parameters),flux,error,blend)
    return np.sum(((flux-model)/error)**2)
def reconstruct(data,initial=None,blend=True):
    flux,error=flux_from_magnitude(data['magnitude'],data['magnitude_error'])
    result=minimize(profile_objective,INITIAL if initial is None else initial,args=(data['time'],flux,error,blend),method='Nelder-Mead')
    if not result.success:raise RuntimeError(result.message)
    p=result.x.copy();p[1]=abs(p[1])
    coefficients,model=fit_fluxes(magnification(data['time'],p),flux,error,blend)
    return {'reconstruction':p[None], 'source_blend_flux':coefficients[None]}, {'flux':flux,'error':error,'model':model,'chi2':profile_objective(p,data['time'],flux,error,blend),'nfev':result.nfev}
