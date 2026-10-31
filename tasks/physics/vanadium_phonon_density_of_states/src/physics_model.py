"""Explicit finite-grid incoherent neutron scattering physics."""
import math
import numpy as np
TEMPERATURE=300.
MASS_AMU=50.94
INCIDENT_ENERGY=120.
KB_MEV=.0862
BETA=1/(TEMPERATURE*KB_MEV)
TARGET_E=np.arange(-15.,80.)
DOS_E=np.arange(80.)

def gamma0(energy,g,beta=BETA,step=1.):
    integrand=np.empty_like(g)
    integrand[1:]=g[1:]/energy[1:]/np.tanh(beta*energy[1:]/2)
    integrand[0]=2*integrand[1]-integrand[2]
    return float(integrand.sum()*step)

def spectral_orders(energy,g,orders=4,beta=BETA,step=1.):
    gamma=gamma0(energy,g,beta,step)
    e=np.r_[-energy[::-1],energy[1:]]
    reflected=np.r_[g[::-1],g[1:]]
    spectrum=np.zeros_like(e);nz=e!=0
    spectrum[nz]=reflected[nz]/(e[nz]*gamma)/(1-np.exp(-e[nz]*beta))
    z=len(energy)-1;spectrum[z]=max(0.,2*spectrum[z+1]-spectrum[z+2])
    values=[spectrum]
    for _ in range(1,orders):
        value=np.convolve(values[0],values[-1],mode='same')*step
        value/=value.sum()*step
        values.append(value)
    return e,np.asarray(values)

def debye_waller(q,energy,g,beta=BETA,step=1.):
    recoil=6.2415097e21*(1.0545716e-34*q/1e-10)**2/(2*MASS_AMU*1.66053873e-27)
    return recoil*gamma0(energy,g,beta,step)

def kinematic_valid(q,energy):
    neutron_mass=1.6749286e-27;charge=1.60217733e-19;hbar=1.054571628e-34
    conversion=np.sqrt(2e-3*charge/neutron_mass)/(hbar/neutron_mass*1e10)
    ki=np.sqrt(INCIDENT_ENERGY)*conversion
    kf=np.sqrt(INCIDENT_ENERGY-energy)*conversion
    qq=q[:,None];kf=kf[None]
    return (ki+kf>qq)&(ki+qq>kf)&(kf+qq>ki)

def forward(g,q,observed_mask):
    extended=np.r_[g,np.zeros(2*len(g))]
    extended=extended/extended.sum()
    ee,A=spectral_orders(np.arange(len(extended),dtype=float),extended)
    dw=debye_waller(q,np.arange(len(extended),dtype=float),extended)
    weights=np.asarray([dw**n*np.exp(-dw)/math.factorial(n) for n in range(1,5)])
    select=(len(extended)-1+TARGET_E).astype(int)
    sp=weights[0,:,None]*A[0,select][None]
    mp=sum(weights[n,:,None]*A[n,select][None] for n in range(1,4))
    kine=kinematic_valid(q,TARGET_E)
    mp=np.where(kine,mp,0.)
    ms=np.broadcast_to(.2*mp.sum(axis=0)/kine.sum(axis=0),mp.shape).copy()
    ms[~kine]=np.nan
    total=sp+mp+ms;total[~observed_mask]=np.nan
    return dict(sp=sp,mp=mp,ms=ms,total=total)

