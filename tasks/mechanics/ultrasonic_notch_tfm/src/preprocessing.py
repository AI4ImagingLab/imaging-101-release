"""Zero-phase bandpass and analytic-signal construction."""
import numpy as np
from scipy.signal import butter, filtfilt
DT=4e-8
SOUND_SPEED=6300.
def load_data(path):
    with np.load(path,allow_pickle=False) as f: return {k:f[k][0] for k in f.files}
def analytic_filter(traces):
    b,a=butter(5,np.array([.5e6,6e6])/(.5/DT),btype='bandpass')
    y=filtfilt(b,a,traces,axis=-1)
    n=y.shape[-1]; h=np.zeros(n); h[0]=1
    if n%2==0: h[1:n//2]=2; h[n//2]=1
    else: h[1:(n+1)//2]=2
    return np.fft.ifft(np.fft.fft(y,axis=-1)*h,axis=-1)
