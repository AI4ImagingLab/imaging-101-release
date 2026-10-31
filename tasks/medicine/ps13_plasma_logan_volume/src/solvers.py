"""Original weighted Logan normal equations, extracted from dynamicpet MIT code."""
import numpy as np
from scipy.linalg import solve as linear_solve
from numpy.linalg import LinAlgError
from .physics_model import transformed_data

def fit_tacs(activity,plasma,frame_start,frame_duration,tstar=30.0,integration_type="trapz",weight_by="frame_duration"):
    """Return slope(N),intercept(N), using original symmetric2x2 solves."""
    x,y,w=transformed_data(activity,plasma,frame_start,frame_duration,tstar,integration_type)
    if weight_by is None:w=np.ones_like(w)
    elif weight_by!="frame_duration":raise ValueError("Unsupported weights")
    matrix_w=np.diag(w);select=frame_start>=tstar;slopes=np.zeros(len(activity));offsets=np.zeros(len(activity))
    for k in range(len(activity)):
        if np.allclose(activity[k,select],plasma[select]):slopes[k]=1.;offsets[k]=0.;continue
        design=x[k];response=y[k,:,None]
        try:coeff=linear_solve(design.T@matrix_w@design,design.T@matrix_w@response,assume_a="sym")
        except LinAlgError:coeff=np.ones((2,1))
        offsets[k]=coeff[0,0];slopes[k]=coeff[1,0]
    return slopes,offsets

def reconstruct(activity,mask,plasma,frame_start,frame_duration,tstar=30.0,integration_type="trapz",weight_by="frame_duration"):
    """Return volume map(z,y,x),nuisance intercept map, float64, zero exterior."""
    v,b=fit_tacs(activity[mask],plasma,frame_start,frame_duration,tstar,integration_type,weight_by)
    volume=np.zeros(mask.shape,np.float64);offset=np.zeros_like(volume);volume[mask]=v;offset[mask]=b
    return volume,offset
