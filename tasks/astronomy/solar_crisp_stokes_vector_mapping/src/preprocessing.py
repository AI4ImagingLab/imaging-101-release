"""Observed CRISP spectra and instrument calibration; no domain imports."""
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def regular_grid(wave,step,extra=5):
    grid_index=np.int32(np.rint(wave/step))
    count=grid_index[-1]-grid_index[0]+1+2*extra
    grid=np.arange(count,dtype='f8')*step-extra*step+wave[0]
    indices=np.array([np.argmin(abs(grid-w)) for w in wave],dtype=np.int32)
    return grid,indices

def load_data(path=None):
    with np.load(path or ROOT/'data/raw_data.npz',allow_pickle=False) as z:
        raw=z['stokes'][0].copy(); wave=z['wavelength'][0].copy()-6302.4931
        cavity=z['cavity_velocity'][0].copy(); psfs=[z['spectral_psf_6301'][0].copy(),z['spectral_psf_6302'][0].copy()]
    fine1,idx1=regular_grid(wave[:17],.005); fine2,idx2=regular_grid(wave[17:],.015)
    indices=np.r_[idx1,idx2+len(fine1)]
    obs=np.zeros((*raw.shape[:2],4,len(fine1)+len(fine2)),dtype='f8'); obs[...,indices]=raw
    sigma=np.full((4,obs.shape[-1]),1e32,dtype='f8'); sigma[:,indices]=.003
    sigma[1:3,indices]/=10; sigma[3,indices]/=3.5
    return {'raw':raw,'observations':obs,'sigma':sigma,'indices':indices,'regions':[fine1+6302.4931,fine2+6302.4931],'psfs':psfs,'cavity':cavity}
