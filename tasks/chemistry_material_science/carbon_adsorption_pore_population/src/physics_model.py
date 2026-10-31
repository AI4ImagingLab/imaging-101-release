"""Explicit calibrated adsorption Fredholm operator and objective."""
import numpy as np
from scipy.interpolate import PchipInterpolator
from .preprocessing import calibration_with_origin
def build_kernel(relative_pressure,kernel_pressure,kernel_loading):
    p,cal=calibration_with_origin(kernel_pressure,kernel_loading)
    q=np.asarray(relative_pressure)
    if np.any(q<p[0]) or np.any(q>p[-1]):raise ValueError('Observation outside calibrated pressure range')
    return PchipInterpolator(p,cal,axis=0,extrapolate=False)(q).T
def forward(weights,kernel):return np.multiply(kernel,np.asarray(weights)[:,None]).sum(axis=0)
def objective(weights,kernel,loading):return float(np.sum((forward(weights,kernel)-loading)**2))
def objective_gradient(weights,kernel,loading):return 2*kernel@(forward(weights,kernel)-loading)
