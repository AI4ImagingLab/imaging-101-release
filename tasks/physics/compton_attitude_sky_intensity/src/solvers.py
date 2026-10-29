"""Ordinary fixed-background Poisson Richardson-Lucy, all physical bins.

The source algorithm is COSIpy cf0cf6c, Apache-2.0, RichardsonLucy.Mstep
with optional flags disabled. Preserve additive float32 delta and update order.
"""
from dataclasses import dataclass
import time
import numpy as np

@dataclass
class ReconstructionResult:
    reconstruction: np.ndarray
    maps: np.ndarray
    log_likelihood: np.ndarray
    delta: np.ndarray
    seconds: float

def poisson_log_likelihood(events,expectation):
    """Source float32 full-array reduction, omitting fixed log-factorials."""
    values=np.log(expectation)
    values*=events
    return float(np.sum(values)-np.sum(expectation))

def reconstruct(data,iterations=30,initial_intensity=1e-4,stopping_increment=.01,callback=None):
    """Recover all sky pixels; callback(iteration,model,delta,likelihood)."""
    if type(iterations) is not int or iterations<1:raise ValueError('iterations must be a positive integer')
    if not np.isfinite(initial_intensity) or initial_intensity<=0:raise ValueError('initial intensity must be positive')
    start=time.perf_counter();operator=data.operator
    exposure=operator.exposure
    if exposure is None:exposure=operator.compute_exposure()
    if np.any(exposure<=0) or not np.all(np.isfinite(exposure)):raise ValueError('All sky pixels must have positive exposure')
    model=np.full(operator.nsky,initial_intensity,dtype=np.float32)
    mean=operator.forward(model);mean+=data.background
    maps=[];likelihoods=[]
    for iteration in range(1,iterations+1):
        # Ratio is formed only after the preceding expectation/likelihood has
        # been consumed; reusing its buffer changes storage, not arithmetic.
        np.divide(data.events,mean,out=mean)
        transpose=operator.adjoint(mean)
        delta=model*(transpose/exposure-1)
        model+=delta
        np.maximum(model,0,out=model)
        del mean
        mean=operator.forward(model);mean+=data.background
        likelihood=poisson_log_likelihood(data.events,mean)
        maps.append(model.copy());likelihoods.append(likelihood)
        if callback is not None:callback(iteration,model,delta,likelihood)
        if iteration>1:
            increment=likelihoods[-1]-likelihoods[-2]
            if 0<=increment<stopping_increment:break
    return ReconstructionResult(model[None].copy(),np.asarray(maps),np.asarray(likelihoods),delta.copy(),time.perf_counter()-start)
