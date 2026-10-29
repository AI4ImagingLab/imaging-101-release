"""Local Buxton rectangular-bolus transport/relaxation equations.

Adapted mathematical implementation from ASLtk, copyright 2024 Antonio
Carlos da Silva Senra Filho, MIT; see task LICENSE and private provenance.
Flow in this point model is mL/g/ms. All times are milliseconds.
"""
import math
import numpy as np
LABEL_DURATION=np.array([100.,100.,150.,150.,400.,800.,1800.])
POST_LABEL_DELAY=np.array([170.,270.,370.,520.,670.,1070.,1870.])
FLOW_TO_CLINICAL=6000000.

def buxton_signal(tau, w, m0, flow, att, partition=.98, blood_t1=1650., efficiency=.85):
    """Return float64 magnetization at paired LD/PLD samples (T,).

    tau/w: equal length vectors in ms; m0: scanner magnetization; flow:
    mL/g/ms; att: ms; partition: mL/g; blood_t1: ms; efficiency: unitless.
    Arrival, finite delivery, and post-delivery washout are evaluated explicitly.
    """
    tau=np.asarray(tau); w=np.asarray(w)
    if tau.ndim!=1 or tau.shape!=w.shape:raise ValueError('LD/PLD must be equally sized vectors')
    t1eff=1/((1/blood_t1)+(flow/partition))
    signal=np.zeros(tau.size,dtype=np.float64)
    for i in range(tau.size):
        t=tau[i]+w[i]
        if t<att:continue
        if t<tau[i]+att:
            q=1-math.exp(-(t-att)/t1eff)
            signal[i]=2.*m0*flow*t1eff*efficiency*q*math.exp(-att/blood_t1)
        else:
            q=1-math.exp(-tau[i]/t1eff)
            signal[i]=2.*m0*flow*t1eff*efficiency*q*math.exp(-att/blood_t1)*math.exp(-(t-tau[i]-att)/t1eff)
    return signal

def forward_volume(reconstruction, arterial_transit_time, m0, partition=.98, blood_t1=1650., efficiency=.85):
    """Return (B,7,Z,Y,X) signal from clinical flow, ATT and M0 (B,Z,Y,X).

    The same three physical branches are broadcast over voxels. No masks,
    spatial interpolation, smoothing, gain fitting or normalization is applied.
    """
    f=np.asarray(reconstruction,dtype=np.float64)/FLOW_TO_CLINICAL
    a=np.asarray(arterial_transit_time,dtype=np.float64);m=np.asarray(m0,dtype=np.float64)
    if f.shape!=a.shape or f.shape!=m.shape or f.ndim!=4:raise ValueError('Maps and M0 must have matching (B,Z,Y,X) shapes')
    eff=1/(1/blood_t1+f/partition); out=[]
    for tau,w in zip(LABEL_DURATION,POST_LABEL_DELAY):
        t=tau+w
        duration=np.clip(t-a,0,tau)
        decay=np.maximum(t-a-tau,0)
        out.append(2*m*f*eff*efficiency*(1-np.exp(-duration/eff))*np.exp(-a/blood_t1)*np.exp(-decay/eff))
    return np.stack(out,axis=1)
