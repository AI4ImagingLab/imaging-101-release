"""Exact dark/stray correction and wavelength selection from measured spectra."""
from pathlib import Path
import numpy as np
FIT_WINDOW=(310.,320.)
STRAY_WINDOW=(280.,290.)
def load_data(path='data/raw_data.npz'):
    """Return unbatched public arrays as a dictionary, without dtype conversions."""
    with np.load(Path(path),allow_pickle=False) as z:return {k:z[k][0].copy() for k in z.files}
def preprocess(wavelength,spectra,dark):
    """Return wavelength(Nfit),dark/stray-corrected counts(...,Nfit)."""
    stray=np.where((wavelength>=STRAY_WINDOW[0])&(wavelength<=STRAY_WINDOW[1]))
    fit=np.where((wavelength>=FIT_WINDOW[0])&(wavelength<=FIT_WINDOW[1]))
    def one(row):
        y=np.subtract(row,dark)
        y=np.subtract(y,np.average(y[stray]))
        return y[fit]
    if spectra.ndim==1:return wavelength[fit],one(spectra)
    return wavelength[fit],np.stack([one(row) for row in spectra])
