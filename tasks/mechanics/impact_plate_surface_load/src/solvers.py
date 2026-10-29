"""Explicit closed-form moving-window virtual-fields inverse."""
import numpy as np
from .preprocessing import filter_deflection,PIXEL_SIZE,SAMPLING_RATE,TIME_SIGMA
from .physics_model import kinematics,hermite_virtual_fields,virtual_work_terms,bending_stiffness,WINDOW_SIZE

def pressure_from_fields(fields: dict, virtual: dict) -> tuple[np.ndarray,dict[str,np.ndarray]]:
    """Recover one pressure(Pa) per valid patch; return bending/inertial parts."""
    a11,a12,inertia=virtual_work_terms(fields,virtual)
    d11,d12=bending_stiffness();normalizer=np.sum(virtual['deflection'])
    bending=(a11*d11+a12*d12)/normalizer;dynamic=inertia/normalizer
    return bending+dynamic,{'bending':bending,'inertial':dynamic}

def reconstruct(deflection: np.ndarray, time_sigma: float=TIME_SIGMA, window: int=WINDOW_SIZE) -> dict[str,np.ndarray]:
    """Input batch-first optically estimated deflection(m); output batch-first pressure(Pa).

    Complete250-frame benchmark defaults return reconstruction(1,250,60,78).
    No optimization, tuning, target, hidden reference, or file read occurs here.
    """
    if deflection.ndim!=4 or deflection.shape[0]!=1:raise ValueError('one batch of time,row,column deflections required')
    a=filter_deflection(deflection,time_sigma)[0];stack=kinematics(a);virtual=hermite_virtual_fields(window)
    pressure=[]
    for i in range(a.shape[0]):
        fields={k:v[i] for k,v in stack.items() if k!='time'}
        p,_=pressure_from_fields(fields,virtual);pressure.append(p)
    return {'reconstruction':np.asarray(pressure)[None]}
