"""Observation-only noise diagnostics; never a new truth or bound dataset.

Add independent zero-mean Gaussian head noise in metres to all30×29 existing
measurements. sigma_m is an absolute standard deviation in metres; NumPy
default_rng uses explicit PCG64 seed. Survey indices and existing calibration
are copied unchanged. No true field or hidden forward result is consulted.
"""
import numpy as np

def perturb_heads(raw,sigma_m=.1,seed=124):
    if isinstance(sigma_m,bool) or not isinstance(sigma_m,(int,float)) or not np.isfinite(sigma_m) or sigma_m<0:raise ValueError('Invalid head-noise standard deviation')
    result={k:np.array(v,copy=True) for k,v in raw.items()}
    result['hydraulic_head']+=np.random.default_rng(seed).normal(0,sigma_m,size=result['hydraulic_head'].shape)
    return result
