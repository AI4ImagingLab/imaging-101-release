"""Single-site mass action and overflow-cell calorimetry.

Mathematics extracted from Harms/Duvvuri/Wheeler pytc (Unlicense),
commit d9ccde3f04e35a3d821ff37a4ad42e62a048d4ac.
No domain package is used by this implementation.
"""
import numpy as np
from .preprocessing import ACQUISITION
def concentrations(volumes,cell_concentration,syringe_concentration,cell_volume):
    """Return protein/ligand M before and after every overflow injection."""
    fractions=np.array([1.]+[np.prod((1-volumes/cell_volume)[:j+1]) for j in range(len(volumes))])
    return cell_concentration*fractions,syringe_concentration*(1-fractions)
def bound_complex(s,t,k):
    """One-site equilibrium complex M; preserve upstream quadratic convention."""
    b=s+t+1/k
    return (b-np.sqrt(b*b-4*s*t))/2
def association_constants(enthalpy,kref,temperatures_kelvin):
    """K(T), M^-1, for constant enthalpy cal/mol and reference 298.15 K."""
    return kref*np.exp(-enthalpy/1.9872036*(1/temperatures_kelvin-1/298.15))
def predict(parameters,volumes):
    """Return all 6x14 predicted injection heats in microcalories.

    Parameter order: dH,Kref,effective site factor, then slope/intercept
    for each of the six experiments. Slopes have units microcal/M.
    """
    p=np.asarray(parameters); h,kref,factor=p[:3]
    ks=association_constants(h,kref,ACQUISITION[:,0]+273.15)
    output=[]
    for i,a in enumerate(ACQUISITION):
        s,t=concentrations(volumes[i],a[1],a[2],a[3]); s=s*factor
        occupancy=bound_complex(s,t,ks[i])/s
        binding=a[3]*s[1:]*h*np.diff(occupancy)
        output.append(binding+t[1:]*p[3+2*i]+p[4+2*i])
    return np.array(output)
