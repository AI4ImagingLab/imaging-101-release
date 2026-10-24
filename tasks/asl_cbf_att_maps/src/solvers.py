"""Voxelwise bounded Buxton fitting with explicit acquisition physics.

Optimization settings and float32 parameter storage follow ASLtk CBFMapping;
Copyright 2024 Antonio Carlos da Silva Senra Filho, MIT (task LICENSE).
"""
import numpy as np
from scipy.optimize import curve_fit
from .physics_model import buxton_signal,LABEL_DURATION,POST_LABEL_DELAY,FLOW_TO_CLINICAL

def reconstruct(magnetization, m0, mask, initial=(1e-5,1000.), lower=(0.,0.), upper=(1.,5000.), partition=.98, blood_t1=1650., efficiency=.85):
    """Fit (flow,ATT) independently at each masked voxel.

    Inputs have documented public batch-first shapes. Return clinical CBF and
    ATT float64 arrays (1,5,35,35). Internal optimizer parameters are rounded
    to float32, matching upstream shared-memory storage, then CBF is converted
    from mL/g/ms to mL/100g/min. Failed optimizer convergence returns two zeros
    exactly as upstream; malformed inputs are rejected by preprocessing.
    No regularizer or neighbor coupling is used. SciPy bounded curve_fit uses
    trust-region reflective least squares with its default stopping settings.
    """
    shape=m0.shape;flow=np.zeros(shape,dtype=np.float32);arrival=np.zeros(shape,dtype=np.float32)
    timing=np.array([LABEL_DURATION,POST_LABEL_DELAY])
    for b,z,y,x in np.argwhere(mask>0):
        m=m0[b,z,y,x]
        def prediction(timing,f,a):
            return buxton_signal(timing[0],timing[1],m,f,a,partition,blood_t1,efficiency)
        try:
            fit,_=curve_fit(prediction,timing,magnetization[b,:,z,y,x],p0=initial,bounds=(lower,upper))
            flow[b,z,y,x]=np.float32(fit[0]);arrival[b,z,y,x]=np.float32(fit[1])
        except RuntimeError:
            pass
    return {'reconstruction':flow.astype(np.float64)*FLOW_TO_CLINICAL,'arterial_transit_time':arrival.astype(np.float64)}

def single_delay_approximation(magnetization, m0, mask, partition=.98, blood_t1=1650., efficiency=.85):
    """Optional clinical-flow comparison, not the shipped standard inverse.

    Uses the last LD/PLD and assumes label has fully arrived and perfusion
    washout is negligible: f = lambda*DM*exp(w/T1b)/
    (2*alpha*T1b*M0*(1-exp(-tau/T1b))). Negative observations are retained.
    """
    tau=LABEL_DURATION[-1];w=POST_LABEL_DELAY[-1]
    out=np.zeros_like(m0,dtype=np.float64);valid=mask>0
    out[valid]=partition*magnetization[:,-1][valid]*np.exp(w/blood_t1)/(2*efficiency*blood_t1*m0[valid]*(1-np.exp(-tau/blood_t1)))*FLOW_TO_CLINICAL
    return out
