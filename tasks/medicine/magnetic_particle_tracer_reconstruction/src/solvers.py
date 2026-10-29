"""Extracted explicit regularized Kaczmarz from official MDF reference.
Copyright2015-2016 Tobias Knopp and Contributors; GPL-3.0-or-later.
All scalar updates, complex conjugations and sweep projections are local.
"""
import numpy as np
from .preprocessing import prepare_system,prepare_measurement,GRID

def row_norms(system):
    """Original per-row norm; nativecomplex64 norm precision preserved."""
    energy=np.zeros(system.shape[0],dtype=np.float64)
    for row in range(len(energy)):energy[row]=np.linalg.norm(system[row])
    return energy

def kaczmarz(system,measurement,iterations=3,regularization=None,return_history=False):
    """Deterministic original augmented-system sweeps with real/positive projection.
    lambda defaults .001*||S||F, not its square; initializec/rzero, no shuffle.
    Projection occurs after each complete sweep, retaining augmented residual.
    This finite-iteration method does not claim exact constrained optimality.
    """
    if regularization is None:regularization=float(np.linalg.norm(system,ord='fro')*1e-3)
    if iterations<1 or regularization<0:raise ValueError('iterations>=1 andlambda>=0 required')
    c=np.zeros(system.shape[1],dtype=measurement.dtype);residual=np.zeros(system.shape[0],dtype=c.dtype)
    norms=row_norms(system);root=np.sqrt(regularization);history=[]
    for _ in range(iterations):
        for row in range(system.shape[0]):
            if norms[row]>0:
                update=(measurement[row]-system[row].dot(c)-root*residual[row])/(norms[row]**2+regularization)
                c[:]+=update*system[row].conjugate()
                residual[row]+=root*update
        if np.iscomplexobj(c):c.imag=0
        c=c*(c.real>0)
        if return_history:history.append(c.real.copy())
    if return_history:return c.real.copy(),np.array(history),regularization
    return c.real.copy()

def reconstruct(observations):
    """OpenNPZ/mapping -> freshfloat64(1,z19,y19,x19), releasinglargearrays."""
    cal=observations['calibration'];bgcal=observations['calibration_background'];system=prepare_system(cal,bgcal,copy=False);del cal,bgcal
    fg=observations['foreground'];measurement=fg[0].mean(axis=0);del fg
    bg=observations['background'];measurement-=bg[0].mean(axis=0);del bg
    coefficients=kaczmarz(system,measurement)
    return coefficients.reshape((1,)+GRID)
