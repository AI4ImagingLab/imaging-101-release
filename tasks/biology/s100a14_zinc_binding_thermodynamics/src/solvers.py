"""Author-published maximum-likelihood path; no Bayesian posterior claimed."""
import numpy as np
from scipy.optimize import least_squares
from .physics_model import predict
def reconstruct(data):
    """Fit 78 heats, excluding injection zero; return scipy OptimizeResult.

    Original default TRF, two-point finite differences, linear loss,
    ftol=xtol=gtol=1e-8; numerical uncertainty convention .1 microcal.
    All parameters unbounded except effective site factor [.5,1.5].
    """
    x0=np.array([0.,1.,1.]+[0.,0.]*6)
    lower=np.full(15,-np.inf); upper=np.full(15,np.inf)
    lower[2]=.5; upper[2]=1.5
    residual=lambda p: ((predict(p,data['injection_volumes'])[:,1:]-data['heats'][:,1:])/.1).ravel()
    return least_squares(residual,x0,bounds=(lower,upper))
def submission(result,data):
    p=result.x
    return {'enthalpy':p[0:1][None], 'association_constant':p[1:2][None],
      'effective_site_factor':p[2:3][None], 'dilution_slope':p[3::2][None],
      'dilution_offset':p[4::2][None], 'predicted_heats':predict(p,data['injection_volumes'])[None]}
