"""Spherical shell geometry and physical count-rate normalization; GPLv3."""
import numpy as np
KPC_CM=3.0856776e21
MPC_CM=3.0856776e24
ARCMIN_KPC=70.73126033926604
LUMINOSITY_MPC=272.69501551950657
REDSHIFT=.059
COUNT_FACTOR=43.53
ELECTRON_TO_HYDROGEN=1/.8527

def projection_volumes(inner,outer):
    """V[annulus,shell] in input-length cubed; exact shell/cylinder volumes."""
    inner=np.asarray(inner).copy();outer=np.asarray(outer).copy()
    outer[:-1]=inner[1:]
    n=len(inner);v=np.zeros((n,n));constant=4/3*np.pi
    # Preserve the source arithmetic as well as the geometric formula: using
    # algebraically expanded differences loses several ulps in thin shells.
    for i in reversed(range(n)):
        v[i,i]=constant*outer[i]**3*(1-(inner[i]/outer[i])**2)**1.5
        for j in reversed(range(i+1,n)):
            f1=(1-(inner[i]/outer[j])**2)**1.5-(1-(outer[i]/outer[j])**2)**1.5
            f2=(1-(inner[i]/inner[j])**2)**1.5-(1-(outer[i]/inner[j])**2)**1.5
            v[i,j]=constant*(f1*outer[j]**3-f2*inner[j]**3)
    return v

def count_to_hydrogen_emission_measure():
    """count/s -> integral nH^2 dV, preserving the author's actual convention.

    XSPEC norm=1e-14 integral(ne*nH)dV/[4*pi*DA^2*(1+z)^2].
    DL=DA*(1+z)^2, ne/nH=1/.8527; count rate=COUNT_FACTOR*norm.
    Thus integral(nH^2)dV=rate*4*pi*1e14*DL^2/[(1+z)^2*nhc*CF].
    """
    return 4*np.pi*1e14*(LUMINOSITY_MPC*MPC_CM)**2/(1+REDSHIFT)**2/ELECTRON_TO_HYDROGEN/COUNT_FACTOR

def edge_correction(inner,outer,emission):
    """McLaughlin outer-shell correction, author's dimensionless expression."""
    inner=np.asarray(inner);outer=np.asarray(outer);n=len(inner)
    a=inner[-1];b=outer[-1]
    e0=(a+b)*a*b/outer**3
    e1=2*outer/b+np.arccos(outer/b)
    e2=outer/b*np.sqrt(1-outer**2/b**2)
    total=e0*(-1+2/np.pi*(e1-e2))
    mask=inner!=0
    e0=(a+b)*a*b/(inner[mask]+outer[mask])/inner[mask]/outer[mask]
    e1=outer[mask]/inner[mask]*np.arccos(inner[mask]/b)-np.arccos(outer[mask]/b)
    e2=outer[mask]/b*(np.sqrt(1-inner[mask]**2/b**2)-np.sqrt(1-outer[mask]**2/b**2))
    total[mask]=e0*(1-2/np.pi*(e1-e2)/(outer[mask]/inner[mask]-1))
    annular_area_ratio=(outer**2-inner**2)/(outer[-1]**2-inner[-1]**2)
    return total*annular_area_ratio*emission[-1]/np.clip(emission,1e-10,None)

def forward_profile(emissivity,edges):
    """Projected source rate/arcmin2 before contribution beyond final shell."""
    area=np.pi*(edges[1:]**2-edges[:-1]**2)
    volume=projection_volumes(edges[:-1]*ARCMIN_KPC*KPC_CM,edges[1:]*ARCMIN_KPC*KPC_CM)
    return volume@emissivity/count_to_hydrogen_emission_measure()/area
