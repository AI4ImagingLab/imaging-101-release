"""Explicit truncated spectral inverse; original method is noniterative."""
import numpy as np
from .physics_model import heat_matrix
from .preprocessing import profiles,detrend

def truncated_inverse(matrix,anomalies,rank=2):
    u,s,vh=np.linalg.svd(matrix,full_matrices=False)
    return (vh[:rank].T/s[:rank])@(u[:,:rank].T@anomalies)

def reconstruct(data,rank=2,diffusivity=1e-6,return_bounds=False):
    result=[]
    for depth,temperature in profiles(data):
        anomalies,_=detrend(depth,temperature)
        result.append(truncated_inverse(heat_matrix(depth,diffusivity),anomalies,rank))
    result=np.asarray(result)
    return result if return_bounds else result[:,:,1]
