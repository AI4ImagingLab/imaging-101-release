"""Two-spin powder average and homogeneous intermolecular background.
Extracted and adapted from DeerLab; Copyright 2019-2023 Luis Fabregas,
Stefan Stoll and contributors; MIT (task LICENSE).
"""
import numpy as np
from scipy.special import fresnel
from scipy.interpolate import interp1d
DISTANCE=np.arange(1.5,6,.01)
MU0=1.25663706212e-6;MUB=9.2740100783e-24;G=2.00231930436256;H=6.62607015e-34;NAV=6.02214076e23
D=(MU0/(4*np.pi))*(MUB*G)**2/(H/(2*np.pi))
def powder_kernel(time,distance=DISTANCE,integral=True):
    """Powder-averaged cosine kernel, time in microseconds, distance in nanometres."""
    phi=np.abs(np.asarray(time))[:,None]*(D*1e21/distance**3)[None,:];q=np.sqrt(6*phi/np.pi);s,c=fresnel(q)
    z=np.divide(c*np.cos(phi)+s*np.sin(phi),q,out=np.ones_like(q),where=q!=0)
    if integral:
        weights=np.r_[distance[1]-distance[0],distance[2:]-distance[:-2],distance[-1]-distance[-2]]/2;z*=weights
    return z
def gaussian_basis(distance,mean1,std1,mean2,std2):
    p=np.exp(-.5*((distance[:,None]-np.array([mean1,mean2]))/np.array([std1,std2]))**2)/(np.sqrt(2*np.pi)*np.array([std1,std2]))
    return p/np.trapezoid(p,distance,axis=0).sum()
def signal_basis(time,parameters,distance=DISTANCE):
    """Nonlinear parameters [mod,reftime,conc,mean1,std1,mean2,std2]; return (T,2)."""
    mod,ref,conc,m1,s1,m2,s2=parameters;dt=np.mean(np.diff(time))
    grid=np.arange(time.min()-.3-.048-dt,time.max()+.3+.048+dt,dt)
    bare=interp1d(grid,powder_kernel(grid,distance),axis=0,kind='cubic')(time-ref)
    weights=np.r_[distance[1]-distance[0],distance[2:]-distance[:-2],distance[-1]-distance[-2]]/2
    background=np.exp(-(8*np.pi**2/(9*np.sqrt(3)))*mod*conc*1e-3*NAV*D*np.abs((time-ref)*1e-6))
    kernel=background[:,None]*((1-mod)*weights[None,:]+mod*bare)
    return kernel@gaussian_basis(distance,m1,s1,m2,s2)
