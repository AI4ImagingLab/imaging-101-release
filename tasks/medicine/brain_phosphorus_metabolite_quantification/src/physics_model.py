"""Explicit damped complex-resonance acquisition model and physical Jacobian."""
import numpy as np
from .preprocessing import interleave

def components(parameters,time):
    """Return (16,N) complex128 FIDs for [amplitude,Hz,damping/s,radian,g].
    The benchmark fixes g=0 (Lorentzian); the complete published expression is
    retained and exposed for independent operator tests.
    """
    t=np.asarray(time);out=[]
    for amplitude,frequency,damping,phase,g in np.asarray(parameters):
        out.append(amplitude*np.exp(1j*phase)*np.exp(-damping*(1-g+g*t)*t)
                   *np.exp(1j*2*np.pi*frequency*t))
    return np.asarray(out)

def forward(parameters,time):
    """Sum all resonance components in native complex signal units."""
    return components(parameters,time).sum(axis=0)

def physical_jacobian(parameters,time):
    """Interleaved (2N,64) derivatives in amplitude,Hz,damping,phase order.
    This matches the original g-fixed physical Jacobian, before multiplet and
    common-phase constraints are applied.
    """
    p=np.asarray(parameters);t=np.asarray(time);fid=components(p,t).T
    jac=np.zeros((t.size,p.shape[0]*4),dtype=np.complex128)
    jac[:,0::4]=fid/p[:,0]
    jac[:,1::4]=fid*2j*np.pi*t[:,None]
    jac[:,2::4]=-fid*t[:,None]*(p[:,4]*t[:,None]-p[:,4]+1)
    jac[:,3::4]=fid*1j
    out=np.empty((2*t.size,jac.shape[1]),dtype=np.float64)
    out[::2]=jac.real;out[1::2]=jac.imag
    return out
