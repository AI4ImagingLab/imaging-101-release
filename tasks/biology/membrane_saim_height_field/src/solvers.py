"""Explicit bound-constrained LM used by the selected SAIM one-colour method.

SAIM equations: MIT(c)2021 paszeklab; LM trust-region rules adapted from
LsqFit.jl0.15.1, MIT(c)2012 John Myles White and contributors. See NOTICE.
No upstream, Julia or domain-package import is needed in this runtime.
"""
import numpy as np
from .physics_model import optical_constants,forward,jacobian
INITIAL=np.array([.5,1.,75.])
LOWER=np.array([0.,0.,0.])
UPPER=np.array([10000.,10000.,200.])

def fit_pixel(constants,signal,initial=INITIAL,max_iter=1000):
    """Return parameters, iterations, convergence and residual sum of squares.

    This retains the selected upstream damping/clipping/acceptance/stopping
    rules; a generic bounded trust-region solver is not substituted.
    """
    x=np.array(initial,dtype=float,copy=True)
    if x.shape!=(3,) or np.any(x<LOWER) or np.any(x>UPPER):raise ValueError('Initial point outside bounds')
    signal=np.asarray(signal,dtype=float)
    residual=forward(constants,x)-signal;sse=float(residual@residual);damping=10.
    converged=False
    for iteration in range(1,max_iter+1):
        j=jacobian(constants,x)
        diagonal=np.maximum(np.sum(j*j,axis=0),1e-6)
        step=np.linalg.solve(j.T@j+damping*np.diag(diagonal),-j.T@residual)
        step=np.minimum(np.maximum(x+step,LOWER),UPPER)-x
        predicted=residual+j@step;pred_sse=float(predicted@predicted)
        trial=forward(constants,x+step)-signal;trial_sse=float(trial@trial)
        denominator=pred_sse-sse
        quality=(trial_sse-sse)/denominator if denominator!=0 else np.nan
        if trial_sse<sse and quality>1e-3:
            x+=step;residual=trial;sse=trial_sse
            if quality>.75:damping=max(.1*damping,1e-16)
        else:damping=min(10*damping,1e16)
        # Upstream tests this iteration's J against its newly retained residual.
        converged=np.linalg.norm(j.T@residual,ord=np.inf)<1e-12 or np.linalg.norm(step)<1e-8*(1e-8+np.linalg.norm(x))
        if converged:break
    return x,iteration,bool(converged),sse

def reconstruct(observations,angles,initial=INITIAL,return_diagnostics=False,**optics):
    """Invert all native pixels; output three(batch,row,column) float64 fields."""
    y=np.asarray(observations)
    if y.ndim!=4 or y.shape[0]!=1 or y.shape[1]!=np.size(angles):raise ValueError('Expected one full angle stack')
    constants=optical_constants(angles,**optics);signals=y[0].reshape(y.shape[1],-1).T
    parameters=np.empty((len(signals),3));iterations=np.empty(len(signals),dtype=np.int64);converged=np.empty(len(signals),dtype=bool);sse=np.empty(len(signals))
    for i,signal in enumerate(signals):parameters[i],iterations[i],converged[i],sse[i]=fit_pixel(constants,signal,initial)
    shape=(1,*y.shape[2:]);out={key:parameters[:,i].reshape(shape) for key,i in [('reconstruction',2),('amplitude',0),('background',1)]}
    if return_diagnostics:return out,dict(iterations=iterations.reshape(shape),converged=converged.reshape(shape),sse=sse.reshape(shape))
    return out
