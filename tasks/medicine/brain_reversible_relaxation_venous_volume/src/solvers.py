"""Explicit unweighted linear inverse of the published long-time ASE model.
Adapted from streamlined_qbold_tools, MIT, Copyright (c) 2016 fmriphysiology.
"""
import numpy as np
from .preprocessing import TAU,log_measurements,observation_mask
from .physics_model import design_matrix,forward_log

def fit_signals(signals,tau=TAU,cutoff=.015):
    """Fit every supplied curve; return parameters [...,3], residual RMS [...]."""
    values=np.asarray(signals)
    if values.shape[-1]!=len(tau): raise ValueError('Signal/time length mismatch')
    logs=log_measurements(values)
    X,indices=design_matrix(tau,cutoff)
    Y=logs[...,indices]
    p=np.linalg.lstsq(X,Y.reshape(-1,len(indices)).T,rcond=None)[0].T.reshape(values.shape[:-1]+(3,))
    residual=np.sqrt(np.mean((forward_log(p,tau,cutoff)-Y)**2,axis=-1))
    return p,residual

def reconstruct(signals):
    """Full four-condition reconstruction; retain signed fits inside fixed mask."""
    p,residual=fit_signals(signals)
    mask=observation_mask(signals)
    outputs={'reconstruction':np.where(mask,p[...,1],0.)[None],
             'dbv':np.where(mask,p[...,0],0.)[None]}
    diagnostics={'parameters':p,'log_rmse':residual,'mask':mask}
    return outputs,diagnostics
