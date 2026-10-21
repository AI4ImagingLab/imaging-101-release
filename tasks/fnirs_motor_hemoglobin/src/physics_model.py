"""Local modified Beer-Lambert acquisition equations, explicit units."""
import numpy as np

def absorption_matrix(extinction, distances, ppf=6.0):
    """E: (1,2,2), d: (1,P) m; return (1,P,2,2) mapping M to OD.
    Extinction is natural-log absorption in m^-1 mM^-1. Concentrations
    are mol/L (M); the factor 1000 converts M to mM.
    """
    e=np.asarray(extinction,dtype=np.float64)
    d=np.asarray(distances,dtype=np.float64)
    f=np.broadcast_to(np.asarray(ppf,dtype=np.float64), (2,))
    if np.any(d<=0) or np.any(f<=0): raise ValueError('Positive pathlength required')
    return e[:,None,:,:] * d[:,:,None,None] * f[None,None,:,None] * 1000.0

def forward(concentration, extinction, distances, ppf=6.0):
    """Concentration (1,P,2,T) M -> dimensionless (1,P,2,T) optical density."""
    return np.einsum('bpij,bpjt->bpit',absorption_matrix(extinction,distances,ppf),concentration)

def adjoint(optical_density, extinction, distances, ppf=6.0):
    """Euclidean transpose of forward, same axis order."""
    return np.einsum('bpij,bpit->bpjt',absorption_matrix(extinction,distances,ppf),optical_density)
