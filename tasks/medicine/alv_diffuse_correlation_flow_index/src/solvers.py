"""Bounded unweighted least-squares inversion of every measured correlation."""
import numpy as np
from scipy.optimize import least_squares
from .preprocessing import prepare
from .physics_model import correlation
def fit_curve(tau,observed):
    # Same source two-parameter LS objective, initialization, and bounds.
    # General bounded TRF is the numerical SciPy port of original lsqcurvefit.
    result=least_squares(lambda x:correlation(tau,*x)-observed,[.48,1000.],
        bounds=([.46,0.],[.54,np.inf]),method='trf',jac='2-point',
        ftol=1e-10,xtol=1e-10,gtol=1e-10,max_nfev=100000)
    if not result.success:raise RuntimeError(result.message)
    return result.x
def reconstruct(data):
    weighted,mask=prepare(data);tau=data['lag_times']
    parameters=np.array([fit_curve(tau[m],row[m]) for row,m in zip(weighted,mask)])
    predicted=np.array([correlation(tau,*x) for x in parameters])
    return {'beta':parameters[:,0][None], 'flow_index':(parameters[:,1]/1e11)[None],
            'predicted_g2':predicted[None]}
