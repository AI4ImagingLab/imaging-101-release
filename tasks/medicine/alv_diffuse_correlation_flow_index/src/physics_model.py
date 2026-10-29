"""Semi-infinite correlation diffusion: explicit mcgeometry equations.

Derived from Melissa Wu's GPL-2.0 mcgeometry semi_infinite_g2.m.
Coordinates/wavelength mm, optical coefficients mm^-1, tau seconds;
q=alpha*D_B(mm^2/s)*1e9 with alpha fixed to1.
"""
import numpy as np
MU_A=.1/10
MU_SP=10/10
RHO=2.5*10
REFRACTIVE_INDEX=1.4
WAVELENGTH_MM=785*1e-6
def field_correlation(tau,q,mu_a=MU_A,mu_sp=MU_SP,rho=RHO,n=REFRACTIVE_INDEX,wavelength_mm=WAVELENGTH_MM):
    tau=np.asarray(tau,dtype=np.float64)
    reflection=-1.440/n**2+.710/n+.668+.0636*n
    ze=2/3*(1+reflection)/(1-reflection)
    zb=ze/mu_sp;z0=1/(mu_a+mu_sp)
    r1=np.sqrt(rho**2+z0**2);r2=np.sqrt(rho**2+(z0+2*zb)**2)
    k0=2*np.pi*n/wavelength_mm
    msd=6*(q/1e9)*tau
    k=np.sqrt(3*mu_a*mu_sp+mu_sp**2*k0**2*msd)
    zero=np.sqrt(3*mu_a*mu_sp)
    g=(3*mu_sp/(4*np.pi))*(np.exp(-k*r1)/r1-np.exp(-k*r2)/r2)
    g0=(3*mu_sp/(4*np.pi))*(np.exp(-zero*r1)/r1-np.exp(-zero*r2)/r2)
    return np.abs(g/g0)
def correlation(tau,beta,q,**kwargs):
    g1=field_correlation(tau,q,**kwargs)
    return 1+beta*g1**2
