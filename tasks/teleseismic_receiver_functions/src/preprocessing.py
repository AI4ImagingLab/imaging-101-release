"""Prepare calibrated three-component seismic observations."""
from pathlib import Path
import numpy as np
from scipy.signal import detrend, iirfilter, zpk2sos, sosfilt
from scipy.signal.windows import hann
SAMPLING_RATE = 20.0
BACK_AZIMUTH = 33.06338963947741
def load_data(path="data/raw_data.npz"):
    """Return float64 ZNE velocity array (3,4801), m/s, from public NPZ."""
    with np.load(Path(path),allow_pickle=False) as f:
        x=f['waveforms']
        if x.shape!=(1,3,4801) or x.dtype!=np.float64 or not np.isfinite(x).all():
            raise ValueError('Expected finite float64 waveforms (1,3,4801)')
        return x[0].copy()
def preprocess(waveforms, low=.05, high=1.0):
    """Linear detrend, 5%-per-side symmetric Hann taper, causal order-4 bandpass."""
    x=detrend(np.asarray(waveforms,dtype=np.float64),type='linear',axis=-1)
    n=x.shape[-1]; w=int(.05*n); sides=hann(2*w+1)
    taper=np.concatenate((sides[:w],np.ones(n-2*w),sides[-w:]))
    z,p,k=iirfilter(4,[low/(SAMPLING_RATE/2),high/(SAMPLING_RATE/2)],btype='band',ftype='butter',output='zpk')
    return sosfilt(zpk2sos(z,p,k),x*taper,axis=-1)
