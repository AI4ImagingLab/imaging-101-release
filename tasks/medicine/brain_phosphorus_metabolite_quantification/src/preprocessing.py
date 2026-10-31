"""Acquisition timing and real/imaginary ordering, without signal rescaling."""
import numpy as np
SPECTRAL_WIDTH_HZ=10000.0
DEADTIME_SECONDS=300e-6
NUCLEAR_FREQUENCY_MHZ=120.0

def time_axis(n, spectral_width=SPECTRAL_WIDTH_HZ, deadtime=DEADTIME_SECONDS):
    """Actual source arange convention, in seconds, beginning at deadtime."""
    dwell=1./spectral_width
    return np.arange(0,dwell*n,dwell)+deadtime

def interleave(fid):
    """Real,imaginary pairs in float64, with no magnitude conversion."""
    value=np.asarray(fid);out=np.empty(value.size*2,dtype=np.float64)
    out[::2]=value.real;out[1::2]=value.imag
    return out
