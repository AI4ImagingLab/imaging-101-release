"""Explicit magnetic dipole forward model (SI). Adapted under MIT."""
import numpy as np

VACUUM_MAGNETIC_PERMEABILITY=1.25663706212e-6
MU0_OVER_4PI=VACUUM_MAGNETIC_PERMEABILITY/(4*np.pi)

def dipole_jacobian(x,y,z,position):
    """Bz per magnetic moment: coordinates um, return T/(A m^2), (N,3)."""
    r=np.column_stack([np.asarray(x).ravel(),np.asarray(y).ravel(),np.broadcast_to(z,np.shape(x)).ravel()])
    r=(r-np.asarray(position))*1e-6
    radius=np.linalg.norm(r,axis=1)
    if np.any(radius==0):raise ValueError('Dipole coincides with sensor')
    a=3*r*r[:,2,None]/radius[:,None]**5
    a[:,2]-=1/radius**3
    return MU0_OVER_4PI*a

def forward(x,y,z,positions,moments):
    """Sum dipole fields, returning shape(x) nT."""
    result=np.zeros(np.size(x))
    for p,m in zip(positions,moments):result+=dipole_jacobian(x,y,z,p)@m*1e9
    return result.reshape(np.shape(x))
