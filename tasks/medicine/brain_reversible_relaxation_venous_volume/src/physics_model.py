"""Static-dephasing long-time ASE model and explicitly qualified diagnostics.
Adapted from streamlined_qbold_tools, MIT, Copyright (c) 2016 fmriphysiology.
"""
import numpy as np
from .preprocessing import TAU
GAMMA=2.675e4  # rad / (second Gauss)
B0=3e4        # Gauss
DELTA_CHI=.264e-6  # dimensionless susceptibility, not .264 ppm numerically
HCT=.34
KAPPA=.03

def design_matrix(tau=TAU,cutoff=.015):
    tau=np.asarray(tau,dtype=np.float64)
    zero=np.flatnonzero(tau==0);long=np.flatnonzero(tau>cutoff)
    if len(zero)!=1 or len(long)<2: raise ValueError('Need one spin echo and at least two long times')
    X=np.column_stack((np.ones(len(long)),-tau[long],np.ones(len(long))))
    return np.vstack(([0.,0.,1.],X)),np.r_[zero,long]

def forward_log(parameters,tau=TAU,cutoff=.015):
    """parameters[...,3] = [DBV fraction,R2prime /s,log spin-echo amplitude]."""
    X,_=design_matrix(tau,cutoff)
    return np.asarray(parameters)@X.T

def oxygen_diagnostics(r2prime,dbv):
    """Exploratory ratios: OEF fraction and dHb g/dL. Singular DBV remains NaN/Inf."""
    r2prime=np.asarray(r2prime,dtype=np.float64);dbv=np.asarray(dbv,dtype=np.float64)
    factor=GAMMA*(4./3.)*np.pi*DELTA_CHI*B0
    with np.errstate(divide='ignore',invalid='ignore'):
        return r2prime/(dbv*factor*HCT),r2prime/(dbv*factor*KAPPA)
