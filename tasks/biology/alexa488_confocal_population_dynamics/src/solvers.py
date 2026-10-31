import numpy as np
from scipy.optimize import least_squares
from .physics_model import correlation_model

def reconstruct(lag_ms, observations, triplet=True):
    """Unweighted physical-model fit; return [N,T,tauD_ms], modeled 200-vector, SSE."""
    if triplet:
        result=least_squares(lambda p: correlation_model(p,lag_ms)-observations,
            [4.,.2,.04],bounds=([1e-6,0.,.002],[1e5,.95,1e3]),
            ftol=1e-13,xtol=1e-13,gtol=1e-13,max_nfev=4000)
        parameters=result.x
    else:
        result=least_squares(lambda p: correlation_model([p[0],0.,p[1]],lag_ms)-observations,
            [4.,.04],bounds=([1e-6,.002],[1e5,1e3]),
            ftol=1e-13,xtol=1e-13,gtol=1e-13,max_nfev=4000)
        parameters=np.array([result.x[0],0.,result.x[1]])
    if not result.success: raise RuntimeError(result.message)
    model=correlation_model(parameters,lag_ms)
    return parameters,model,float(np.sum((model-observations)**2))
