"""AFI pulsed steady state and short-TR closed-form inverse.

Inverse mathematics extracted from qmrpy B1AFI, MIT; qMRLab portions
Copyright (c) 2017 NeuroPoly. See NOTICE for original attribution.
"""
import numpy as np
NOMINAL_FLIP_ANGLE_DEG=60.
TR1_MS=20.
TR2_MS=100.
ECHO_TIME_MS=3.53
def afi_ratio(b1):
    """Return S2/S1 in the short-TR AFI model; b1 is actual/nominal angle."""
    n=TR2_MS/TR1_MS;c=np.cos(np.asarray(b1)*NOMINAL_FLIP_ANGLE_DEG*np.pi/180)
    return (1+n*c)/(n+c)
def invert_afi(signal):
    """Return dimensionless B1 ratio from (...,2) magnitude measurements.

    Preserve upstream clipping, r>1→zero-angle and zero-signal NaN behavior.
    The declared valid mask excludes these invalid/noise-dominated cases.
    """
    signal=np.asarray(signal,dtype=np.float64)
    s1=signal[...,0];s2=signal[...,1];n=TR2_MS/TR1_MS
    with np.errstate(divide='ignore',invalid='ignore'):
        r=np.abs(s2/s1)
        c=np.where(r==n,1.,(n*r-1)/(n-r))
    c=np.where(r>1,1.,np.clip(c,-1.,1.))
    b1=np.arccos(c)*180/np.pi/NOMINAL_FLIP_ANGLE_DEG
    return np.where(np.abs(s1)<1e-12,np.nan,b1)
def exact_steady_state(b1,t1_ms,m0=1.,tr1_ms=TR1_MS,tr2_ms=TR2_MS):
    """Return (...,2) exact fully spoiled Bloch steady-state signal.

    Diagnostic only: T1 is not known to the benchmark inverse. Echo scaling
    is absorbed in m0. This permits independent checks of short-TR bias.
    """
    a=np.asarray(b1)*NOMINAL_FLIP_ANGLE_DEG*np.pi/180;c=np.cos(a)
    e1=np.exp(-tr1_ms/np.asarray(t1_ms));e2=np.exp(-tr2_ms/np.asarray(t1_ms))
    den=1-e1*e2*c*c
    z1=m0*((1-e2)+e2*c*(1-e1))/den
    z2=m0*((1-e1)+e1*c*(1-e2))/den
    return np.stack([z1*np.sin(a),z2*np.sin(a)],axis=-1)
