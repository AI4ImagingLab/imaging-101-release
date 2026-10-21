"""Isotropic minimum-norm current field; explicit whitening, SVD, inverse."""
import numpy as np
from .physics_model import projection
NOISE_STD=1e-5
LAMBDA2=1/9
def reconstruct(data,lambda2=LAMBDA2):
    if lambda2<=0:raise ValueError('lambda2 must be positive')
    g=data['lead_field'];y=data['measurements'];p=projection(g.shape[0])
    a=p@g/NOISE_STD
    u,s,vh=np.linalg.svd(a,full_matrices=False)
    rank=g.shape[0]-1
    trace_scale=float(np.sum(s*s)/rank)
    kernel=(vh.T*(s/(s*s+lambda2*trace_scale)))@u.T@p/NOISE_STD
    current=(kernel@y).reshape(-1,3,y.shape[-1])
    return {'reconstruction':current[None]}, {'kernel':kernel,'singular_values':s/np.sqrt(trace_scale),'trace_scale':trace_scale,'projection':p}
