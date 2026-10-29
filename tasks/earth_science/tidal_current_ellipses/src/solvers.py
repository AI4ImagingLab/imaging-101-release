"""Explicit complex ordinary least-squares tidal inversion.
Adapted from UTide MIT: selected no-node/raw-phase/no-trend case.
"""
import numpy as np
from .physics_model import harmonic_basis,coefficients_from_rotary
def solve(time_seconds,velocity,frequency_cph,reference_days=None,return_diagnostics=False):
    """Real(2,N) observed cm/s -> four batch-first real output arrays.
    No interpolation, astronomical/nodal correction or trend; finite inputs.
    """
    b,tref=harmonic_basis(time_seconds,frequency_cph,reference_days)
    z=velocity[0]+1j*velocity[1];m=np.linalg.lstsq(b,z,rcond=None)[0]
    c,s,mean=coefficients_from_rotary(m);fit=b@m
    out=dict(cosine=c[None],sine=s[None],mean_current=mean[None],reconstruction=np.array([fit.real,fit.imag])[None])
    return(out,dict(B=b,m=m,reference_days=tref,residual=z-fit))if return_diagnostics else out
