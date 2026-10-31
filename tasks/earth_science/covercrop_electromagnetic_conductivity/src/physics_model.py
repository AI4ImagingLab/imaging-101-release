"""McNeill cumulative-sensitivity equations, adapted from EMagPy (GPL-3.0).
Copyright original EMagPy authors; source revision recorded in test/validation.md.
Coil ordering is three VCP, then three HCP; all conductivity is mS/m.
"""
import numpy as np
ORIENTATIONS=('vcp','vcp','vcp','hcp','hcp','hcp')
def cumulative_sensitivity(depths,spacing,orientation):
    """Fraction of homogeneous half-space signal below depth (m)."""
    z=np.asarray(depths)/spacing
    if orientation=='hcp': return 1/np.sqrt(4*z*z+1)
    if orientation=='vcp': return np.sqrt(4*z*z+1)-2*z
    raise ValueError('orientation must be hcp or vcp')
def jacobian(interfaces,spacings):
    z=np.r_[0.,interfaces];J=np.empty((6,len(z)))
    for i,(s,o) in enumerate(zip(spacings,ORIENTATIONS)):
        c=cumulative_sensitivity(z,s,o);J[i,:-1]=c[:-1]-c[1:];J[i,-1]=c[-1]
    return J
def forward(conductivity,interfaces,spacings):
    """Layer conductivities (7,) -> apparent conductivities (6,), mS/m."""
    z=np.r_[0.,interfaces];out=np.empty(6)
    for i,(s,o) in enumerate(zip(spacings,ORIENTATIONS)):
        c=cumulative_sensitivity(z,s,o)
        out[i]=np.sum(conductivity[:-1]*(c[:-1]-c[1:]))+conductivity[-1]*c[-1]
    return out
def adjoint(data,interfaces,spacings):
    return jacobian(interfaces,spacings).T@data
