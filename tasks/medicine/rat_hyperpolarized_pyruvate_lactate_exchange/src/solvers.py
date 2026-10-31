"""Published inputless constrained nonlinear least squares, all native pixels.
Copyright 2014-2024 Regents of the University of California; BSD-3-Clause.
"""
import numpy as np
from scipy.optimize import least_squares
from .preprocessing import prepare
from .physics_model import rf_factors,trajectory,residual
def reconstruct(signal,angles,t1p=30.0,t1l=25.0,initial_lactate=1.0):
    """Return two (1,x,y) fields and identified same-solve diagnostic arrays.
    Source API configuration: fixed unused S0P=0; kPL,L0>=0; initial(.002,1).
    TRF defaults except ftol=xtol=gtol=1e-6, two-point numerical Jacobian.
    """
    padded,flips=prepare(signal,angles);ss,zz=rf_factors(flips,25);n=len(padded)
    params=np.zeros((n,2));fit=np.zeros((n,25));forcing=np.zeros_like(fit);objective=np.zeros(n);optimality=np.zeros(n);nfev=np.zeros(n,dtype=np.int32);status=np.zeros(n,dtype=np.int32)
    for i,mxy in enumerate(padded):
        if not mxy.any():continue
        mz=mxy/ss
        r=least_squares(residual,np.array([.002,initial_lactate]),bounds=(np.zeros(2),np.full(2,np.inf)),ftol=1e-6,xtol=1e-6,gtol=1e-6,args=(mz,zz,ss,2.0,t1p,t1l))
        params[i]=r.x;objective[i]=r.cost;optimality[i]=r.optimality;nfev[i]=r.nfev;status[i]=r.status
        state,u=trajectory(r.x,mz[0],zz,2.0,t1p,t1l);fit[i]=state[1]*ss[1];forcing[i]=u*ss[0]
    shape=signal.shape[1:3]
    result={'exchange_rate':params[:,0].reshape((1,)+shape),'initial_lactate':params[:,1].reshape((1,)+shape)}
    diag={'fitted_lactate':fit.reshape(shape+(25,)),'forcing_signal_scaled':forcing.reshape(shape+(25,)),'objective':objective.reshape(shape),'optimality':optimality.reshape(shape),'nfev':nfev.reshape(shape),'status':status.reshape(shape)}
    return result,diag
