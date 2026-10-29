"""Spoiled gradient echo forward model and linearized inverse.

Mathematics extracted from qmrpy T1VFA (MIT), incorporating qMRLab
Copyright (c) 2017 NeuroPoly. See NOTICE.
"""
import numpy as np
FLIP_ANGLES_DEG=np.array([3.,20.])
REPETITION_TIME_MS=15.
def spgr_signal(m0,t1_ms,b1=1.):
    """Return (...,2) steady-state magnitude signal in acquisition units."""
    angle=np.deg2rad(FLIP_ANGLES_DEG)*np.asarray(b1)[...,None]
    e=np.exp(-REPETITION_TIME_MS/np.asarray(t1_ms))[...,None]
    return np.asarray(m0)[...,None]*np.sin(angle)*(1-e)/(1-e*np.cos(angle))
def linearized_fit(signal,b1=1.):
    """Return T1(ms), M0(arbitrary units) from final-axis flip-angle samples."""
    angle=np.deg2rad(FLIP_ANGLES_DEG)*np.asarray(b1)[...,None]
    x=signal/np.tan(angle);y=signal/np.sin(angle)
    # Ordinary least squares of y=intercept+E*x; two distinct angles.
    xm=x.mean(-1,keepdims=True);ym=y.mean(-1,keepdims=True)
    slope=np.sum((x-xm)*(y-ym),-1)/np.sum((x-xm)**2,-1)
    slope=np.clip(slope,1e-12,1.-1e-12)
    intercept=y.mean(-1)-slope*x.mean(-1)
    return -REPETITION_TIME_MS/np.log(slope),intercept/(1-slope)
