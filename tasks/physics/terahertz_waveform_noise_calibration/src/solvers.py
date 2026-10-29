"""Published joint full-waveform BFGS inverse, including source stopping state."""
import numpy as np
from scipy.optimize import minimize
from .preprocessing import DT_PS
from .physics_model import initialize,objective,gradient,decode_outputs

def reconstruct(waveforms,dt=DT_PS,noise_model='full',maxiter=10000,gtol=1e-6):
    """Jointly fit mu, all amplitude/delay drifts and three noise amplitudes.

    Source default first-frame gauge is a[0]=1, eta[0]=0. The standard BFGS
    precision-loss false flag is preserved; no diagnostic Newton is submitted.
    noise_model='additive' fixes beta/tau to negligible values for a labeled
    model-sensitivity experiment only. It does not replace the selected method.
    """
    x=np.asarray(waveforms,dtype=np.float64);theta,scales,initial=initialize(x,dt)
    free=np.ones(theta.size,dtype=bool)
    if noise_model=='additive':
        theta[1:3]=-100/scales.logv[1:3];free[1:3]=False
    def expand(p):
        z=theta.copy();z[free]=p;return z
    f=lambda p:objective(expand(p),x,scales)
    g=lambda p:gradient(expand(p),x,scales)[free]
    history=[float(f(theta[free]))]
    callback=lambda p:history.append(float(f(p)))
    result=minimize(f,theta[free],method='BFGS',jac=g,tol=1e-5*x.size,options=dict(maxiter=maxiter,gtol=gtol),callback=callback)
    final=expand(result.x);out=decode_outputs(final,x,scales,dt)
    diagnostic=dict(success=bool(result.success),message=str(result.message),iterations=int(result.nit),nll=float(result.fun),gradient_inf=float(abs(result.jac).max()),gradient_inf_per_observation=float(abs(result.jac).max()/x.size),objective_history=np.array(history),theta=final,initial_theta=theta,initial_gradient=gradient(theta,x,scales),scales=scales,initial=initial,model=noise_model)
    return out,diagnostic
