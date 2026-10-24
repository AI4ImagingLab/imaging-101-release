"""Explicit FGBuster spectral laws. GPL-3.0; original FGBuster authors.
Copyright (C)2019 Davide Poletti, Josquin Errard and FGBuster developers.
The data are thermodynamic microkelvin; frequency arguments are GHz.
"""
import numpy as np
H_OVER_K=6.62607015e-34*1e9/1.380649e-23
T_CMB=2.7255
def rj_to_cmb(frequency):
    x=H_OVER_K*np.asarray(frequency)/T_CMB
    return np.expm1(x)**2/(np.exp(x)*x*x)
def mixing_matrix(frequencies,beta_sync=-3.,beta_dust=1.54,dust_temperature=20.):
    """Return (F,3), columns CMB,synchrotron22.8GHz,dust93.5GHz."""
    nu=np.asarray(frequencies,dtype=float);g=rj_to_cmb(nu)
    synch=(nu/22.8)**beta_sync*g/rj_to_cmb(22.8)
    dust=np.expm1(H_OVER_K*93.5/dust_temperature)/np.expm1(H_OVER_K*nu/dust_temperature)*(nu/93.5)**(1+beta_dust)*g/rj_to_cmb(93.5)
    return np.stack([np.ones_like(nu),synch,dust],axis=-1)
def forward(components,mixing):
    """(C,Y,X) maps -> (F,Y,X) calibrated frequency maps, microkelvin."""
    return np.einsum('fc,cyx->fyx',mixing,components)
def adjoint(measurements,mixing):
    return np.einsum('fc,fyx->cyx',mixing,measurements)
