"""Bounded trust-region least squares for position, amplitude and local sky."""
import numpy as np
from scipy.optimize import least_squares
from .preprocessing import prepare_psf,initialize
from .physics_model import make_interpolator,residual,model

def reconstruct(images, flight_psf):
    """Return output contract dictionary and freshly fitted image stack (4,17,17)."""
    psf=prepare_psf(flight_psf)
    interpolator=make_interpolator(psf)
    parameters=[]
    fitted=[]
    for patch in images:
        initial=initialize(patch)
        lower=[0,initial[1]-2,initial[2]-2,-np.inf]
        upper=[np.inf,initial[1]+2,initial[2]+2,np.inf]
        result=least_squares(residual,initial,args=(interpolator,patch),bounds=(lower,upper),
                             jac='2-point',diff_step=np.sqrt(np.finfo(float).eps),
                             method='trf',max_nfev=200,xtol=1e-10,ftol=1e-8,gtol=1e-8)
        if not result.success:
            raise RuntimeError(result.message)
        parameters.append(result.x)
        fitted.append(model(result.x,interpolator))
    parameters=np.asarray(parameters)
    return {'positions':parameters[None,:,1:3], 'flux':parameters[None,:,0],
            'background':parameters[None,:,3]},np.asarray(fitted)
