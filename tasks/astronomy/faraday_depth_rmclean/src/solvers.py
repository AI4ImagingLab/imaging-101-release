"""Explicit selected RM-CLEAN inverse updates; no reconstruction-library imports.

Adapted from RMutils.util_RM.RMcleaner, copyright (c) 2020 CIRADA, MIT; ../LICENSE.
The selected published path disables the optional second masked clean (window=None).
"""
import time
import numpy as np
from .preprocessing import prepare
from .physics_model import dirty_spectrum,rotation_measure_response,forward

PHI=np.arange(-600.,603.,3.)

def hogbom_clean(dirty,phi,response,phi2,width,cutoff,gain=.1,max_iterations=1000):
    """Return restored spectrum, complex components, residual and update count.

    Global absolute-peak Hogbom stage, gain0.1, 3sigma stopping, no masked second stage.
    Each component is convolved with a unit-peak Gaussian, then dirty residual restored.
    """
    residual=dirty.copy();components=np.zeros_like(dirty);restored=np.zeros_like(dirty)
    pad=(len(phi2)-len(phi))//2
    response_peak=int(np.argmax(response))
    iterations=0
    while np.max(abs(residual))>=cutoff and iterations<max_iterations:
        index=int(np.argmax(abs(residual)));component=gain*residual[index]
        components[index]+=component
        shifted=np.roll(response,index+pad-response_peak)[pad:-pad]
        residual-=component*shifted
        restored+=component*np.exp(-4*np.log(2)*((phi-phi[index])/width)**2)
        iterations+=1
    return restored+residual,components,residual,iterations

def reconstruct(frequency,polarization,noise):
    """Reconstruct the full spectrum; returns submission and fresh diagnostic arrays."""
    start=time.perf_counter();data=prepare(frequency,polarization,noise)
    dirty=dirty_spectrum(data['polarization'],data['lambda2'],PHI,data['weights'],data['lambda0'])
    response,phi2,width=rotation_measure_response(data['lambda2'],PHI,data['weights'],data['lambda0'])
    cutoff=3*data['sigma_fdf']
    clean,components,residual,iterations=hogbom_clean(dirty,PHI,response,phi2,width,cutoff)
    model=forward(components,data['lambda2'],PHI,data['lambda0'])
    details=dict(data,dirty=dirty,response=response,phi2=phi2,width=width,residual=residual,
                 model=model,iterations=iterations,cutoff=cutoff,runtime_seconds=time.perf_counter()-start)
    return {'reconstruction':clean[None], 'components':components[None]},details
