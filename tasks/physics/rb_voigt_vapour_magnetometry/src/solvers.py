"""Published differential-evolution family on the identifiable physical gauge.

Normalize Ex=cos(theta), Ey=sin(theta); normalized S0 is invariant to common
gain. The original lmfit bounded-parameter sine mapping and DE settings are
evaluated explicitly with general SciPy. No upstream fitting library is used.
"""
import time
import numpy as np
from scipy.optimize import differential_evolution
from .physics_model import transmission

LOWER=np.array([0.,70.,3900.])
UPPER=np.array([np.pi/2,130.,4500.])

def from_internal(values):
    """Original finite-two-sided lmfit bound transform, theta/Celsius/Gauss."""
    return np.array([lower+(np.sin(value)+1.)*(upper-lower)/2. for value,lower,upper in zip(values,LOWER,UPPER)])

def fit_sweep(detuning,observations,nominal_angle,seed=20261029):
    """Fit all native points with DE best1bin, max1500 objective evaluations."""
    start=time.perf_counter();evaluations=0
    def objective(internal):
        nonlocal evaluations
        evaluations+=1
        if evaluations>1500:raise RuntimeError('Original DE objective budget exhausted without convergence')
        theta,t,b=from_internal(internal)
        prediction=transmission(detuning,t,b,np.cos(theta),np.sin(theta))
        residual=observations-prediction
        return (residual*residual).sum()
    result=differential_evolution(objective,[(-np.pi/2,np.pi/2)]*3,strategy='best1bin',maxiter=1500,popsize=7,tol=1e-7,atol=0.,mutation=(.5,1.),recombination=.7,seed=seed,polish=False,init='latinhypercube',updating='immediate',workers=1)
    if not result.success:raise RuntimeError('Original DE convergence not established: '+str(result.message))
    theta,t,b=from_internal(result.x)
    prediction=transmission(detuning,t,b,np.cos(theta),np.sin(theta))
    return np.array([b*1e-4,t+273.15,theta]),prediction,{'native_success':bool(result.success),'evaluations':int(result.nfev),'generations':int(result.nit),'seconds':time.perf_counter()-start,'rms':float(np.sqrt(np.mean((observations-prediction)**2))),'seed':seed,'nominal_angle':nominal_angle}
