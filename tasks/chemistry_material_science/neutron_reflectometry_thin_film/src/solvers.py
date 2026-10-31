import time
import numpy as np
from scipy.optimize import differential_evolution
from .physics_model import BOUNDS,negative_log_likelihood,profile,forward
def reconstruct(q,r,error,dq,seed=2718):
    """Run the tutorial maximum-likelihood differential-evolution stage on every observation."""
    start=time.perf_counter()
    fit=differential_evolution(negative_log_likelihood,BOUNDS,args=(q,r,error,dq),seed=seed,popsize=15,maxiter=1000,tol=1e-8,polish=True,workers=1)
    if not np.isfinite(fit.fun): raise RuntimeError('nonfinite likelihood')
    return dict(reconstruction=profile(fit.x)[None],parameters=fit.x[None]),dict(seconds=time.perf_counter()-start,nll=float(fit.fun),iterations=int(fit.nit),optimizer_success=bool(fit.success),optimizer_message=str(fit.message))
