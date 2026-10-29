"""Explicit Michaelis-Menten forward map and derivatives."""
import numpy as np
def rate(substrate,parameters):
    """S[mM], parameters=[Km(mM),Vmax(micromolar/min)] -> rate."""
    k,v=np.asarray(parameters,dtype=float)
    s=np.asarray(substrate,dtype=float)
    return v*s/(k+s)
def jacobian(substrate,parameters):
    """Derivative of rate, columns Km,Vmax; shape (n,2)."""
    k,v=np.asarray(parameters,dtype=float);s=np.asarray(substrate,dtype=float)
    return np.column_stack((-v*s/(k+s)**2,s/(k+s)))
def profile(substrate,observed,km_grid):
    """Profile residual SSE: eliminate V linearly at each fixed positive Km."""
    a=substrate[None,:]/(np.asarray(km_grid)[:,None]+substrate[None,:])
    v=(a@observed)/np.sum(a*a,axis=1)
    return v,np.sum((a*v[:,None]-observed)**2,axis=1)
