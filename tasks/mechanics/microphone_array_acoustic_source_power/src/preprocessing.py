"""Public NPZ reader. Numerical preprocessing is defined below."""
import numpy as np
def load_data(path='data/raw_data.npz'):
    with np.load(path,allow_pickle=False) as f:
        data={k:f[k].copy() for k in f.files}
    if any(v.shape[0]!=1 or not np.isfinite(v).all() for v in data.values()):
        raise ValueError('Expected finite batch-first measurements, batch size 1')
    return {k:v[0] for k,v in data.items()}

INVALID_CHANNELS=[1,7]
def calibrate(signal,factor):
    valid=np.ones(signal.shape[1],bool);valid[INVALID_CHANNELS]=False
    return signal[:,valid]*factor[None,valid],valid
def cross_spectral_matrix(signal,block_size=128):
    """One-sided Welch bin power: Hann, 50% overlap, no mean removal."""
    window=np.hanning(block_size);step=block_size//2;n=(len(signal)-block_size)//step+1
    spectra=np.array([np.fft.rfft(signal[i*step:i*step+block_size]*window[:,None],axis=0) for i in range(n)])
    return np.einsum('bfm,bfn->fmn',spectra,spectra.conj())*(2/block_size/(window@window)/n)
