"""RF depletion and inputless hybrid ODE. Extracted from UCSF 2024 translation.
Copyright 2014-2024 Regents of the University of California; BSD-3-Clause, LICENSE.
The two uncoupled dummy states preserve the selected source matrix conventions.
"""
import numpy as np
from scipy.linalg import expm
def rf_factors(flips,nt=25):
    """Return mean transverse and cumulative longitudinal factors, (metabolite,time)."""
    flips=np.asarray(flips);nf=flips.shape[1]//nt
    if nf<1 or flips.shape[1]!=nf*nt:raise ValueError('incomplete RF groups')
    longitudinal=np.zeros((flips.shape[0],nt));transverse=np.zeros_like(longitudinal)
    for t in range(nt):
        idx=np.arange(nf)+t*nf;longitudinal[:,t]=np.prod(np.cos(flips[:,idx]),axis=1)
        for n in range(nf):transverse[:,t]=transverse[:,t]+np.sin(flips[:,idx[n]])*np.prod(np.cos(flips[:,idx[:n]]),axis=1)
    return transverse/nf,longitudinal
def trajectory(parameters,pyr_longitudinal,longitudinal,tr=2.0,t1p=30.0,t1l=25.0):
    """Pre-RF 4xT states and T forcing values; final unused forcing is zero.
    Parameters are kPL[s^-1], initial L longitudinal native amplitude.
    Constant inverse/exponential hoisted out of time loop, without altered arithmetic.
    """
    k,l0=np.asarray(parameters,dtype=float);p=np.asarray(pyr_longitudinal);n=p.size
    states=np.zeros((4,n));states[0]=p;states[1,0]=l0;u=np.zeros(n)
    A=np.array([[-1/t1p-k,0,0,0],[k,-1/t1l,0,0],[0,0,-1,0],[0,0,0,-1]])
    inverse=np.linalg.inv(A);propagator=expm(A*tr)
    for t in range(n-1):
        initial=states[:,[t]]*longitudinal[:,[t]]
        decay=np.exp((-1/t1p-k)*tr);rate=1/t1p+k
        u[t]=(p[t+1]-initial[0,0]*decay)*rate/(1-decay)
        equilibrium=-inverse@np.array([[u[t]],[0],[0],[0]])
        states[:,[t+1]]=equilibrium+propagator@(initial-equilibrium)
    return states,u
def residual(parameters,mz,longitudinal,transverse,tr=2.0,t1p=30.0,t1l=25.0):
    """Source sign is measured minus model, in transverse native signal units."""
    state,_=trajectory(parameters,mz[0],longitudinal,tr,t1p,t1l)
    return ((mz-state)*transverse)[1:2].flatten()
