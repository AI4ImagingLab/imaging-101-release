"""Explicit bound transforms and source-convention MINPACK least squares.

The sine bounds reproduce lmfit 1.3.4 Parameter.setup_bounds (BSD-3-Clause).
No lmfit or pyElli dependency is used by this cleaned implementation.
"""
import time
import numpy as np
from scipy.optimize import leastsq
from .preprocessing import measured_rho
from .physics_model import predict_rho,cauchy_index

LOWER=np.array([-100.,-40000.,-40000.,0.])
UPPER=np.array([100.,40000.,40000.,40000.])

def initial_parameters():
    """Return the author's [n0,n1,n2,thickness_nm] initialization."""
    return np.array([2.236,451.,251.,20.])

def to_internal(parameters):
    """Two-sided bound transform u=arcsin(2*(p-min)/(max-min)-1)."""
    p=np.asarray(parameters,dtype=np.float64)
    if p.shape!=(4,) or not np.isfinite(p).all() or np.any(p<LOWER) or np.any(p>UPPER):raise ValueError('Initial parameters outside declared bounds')
    return np.array([float(np.arcsin(2*(v-lo)/(hi-lo)-1)) for v,lo,hi in zip(p,LOWER,UPPER)])

def from_internal(internal):
    """p=min+(sin(u)+1)*(max-min)/2, using source scalar arithmetic order."""
    return np.array([float(lo+(np.sin(v)+1)*(hi-lo)/2.) for v,lo,hi in zip(internal,LOWER,UPPER)])

def residual(parameters,data):
    """All real observed-minus-model residuals, then all imaginary residuals."""
    difference=measured_rho(data)-predict_rho(data,parameters)
    return np.concatenate((difference.real,difference.imag))

def reconstruct(data,initial=None):
    """Return (output,record); all926 complex observations and all4unknowns.

    The numerical primitive is MINPACK lmdif via scipy.leastsq with the native
    lmfit defaults: ftol=xtol=1.5e-8, gtol=0, epsfcn=1e-10, factor100.
    The source guard allows10000 adjusted evaluations plus2 setup probes,
    hence10002 raw callbacks; this is not a bounded-TRF replacement.
    Evaluation traces are function calls, including finite-difference probes.
    """
    start=time.perf_counter();evaluations=[]
    internal=to_internal(initial_parameters() if initial is None else initial)
    def objective(u):
        if len(evaluations)>=10002:raise RuntimeError('Native function-evaluation budget exceeded')
        p=from_internal(u);r=residual(p,data)
        if not np.isfinite(r).all():raise ValueError('Nonfinite model residual')
        evaluations.append(np.r_[p,float(r@r)])
        return r
    best,covariance,info,message,status=leastsq(objective,internal,full_output=True,
        ftol=1.5e-8,xtol=1.5e-8,gtol=0.,maxfev=20000,epsfcn=1e-10,factor=100)
    if status not in [1,2,3,4]:raise RuntimeError('MINPACK failed: '+message)
    parameters=from_internal(best);r=residual(parameters,data)
    covariance_external=None
    if covariance is not None:
        derivative=(UPPER-LOWER)/2*np.cos(best)
        covariance_external=covariance*derivative[:,None]*derivative[None,:]
        covariance_external*=float(r@r)/(len(r)-4)
    output={'parameters':parameters[None],
            'reconstruction':cauchy_index(data['wavelength_nm'],parameters)[None]}
    record=dict(nfev=int(info['nfev']),success=True,message=message,status=int(status),
        residual=r,seconds=time.perf_counter()-start,evaluations=np.asarray(evaluations),
        covariance=covariance_external,initial=from_internal(internal))
    return output,record
