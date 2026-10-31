"""Measured photometry and explicit magnitude convention."""
import numpy as np
ZEROPOINT=22.
def load_data(path="data/raw_data.npz"):
    with np.load(path,allow_pickle=False) as f:return {k:f[k][0] for k in f.files}
def flux_from_magnitude(magnitude,error):
    flux=10.**(.4*(ZEROPOINT-magnitude))
    return flux,error*flux*np.log(10.)*.4
def magnitude_from_flux(flux):
    return ZEROPOINT-2.5*np.log10(flux)
