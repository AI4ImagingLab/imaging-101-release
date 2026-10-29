"""MIT2022 Felipe Gutierrez-Barragan; extracted original FullHist matched filter.

The original10000-bin complete IdentityCoding constructor/decoder was actually
run on all2500 histograms. Unused dense coding matrices are not part of the
local mathematical algorithm. No compressed measurement is claimed here.
"""
import numpy as np
from .physics_model import shifts_to_depth
EPSILON=1e-8
def zero_norm(a):
    """Subtract per-histogram mean and divide by L2 norm+original1e-8 epsilon."""
    a=np.asarray(a,dtype=np.float64);a=a-a.mean(axis=-1,keepdims=True)
    return a/(np.linalg.norm(a,axis=-1,keepdims=True)+EPSILON)
def circular_correlation(signal,template):
    """Correlation at lag s=sum_i template[i]*signal[(i+s)modN]."""
    if signal.shape[-1]!=template.shape[-1]:raise ValueError('Signal/template lengths differ')
    return np.fft.ifft(np.fft.fft(template).conj()*np.fft.fft(signal,axis=-1),axis=-1).real
def matched_peaks(histograms,irf):
    """Original integer maximum-lag estimator; ties follow NumPy argmax first index."""
    corr=circular_correlation(zero_norm(histograms),zero_norm(irf));return np.argmax(corr,axis=-1),corr
def reconstruct(counts,irf):
    """Return float64(50,50) relative delay-depth in mm, int64 peak-bin image.

    Row batches bound FFT memory. Every pixel and every10000-bin window is used.
    No count mask, range gate, clipping, spatial smoothing or sub-bin fitting.
    """
    if counts.shape!=(50,50,10000) or irf.shape!=(10000,):raise ValueError('Incorrect bound shapes')
    peaks=np.empty((50,50),dtype=np.int64)
    for row in range(50):peaks[row],_=matched_peaks(counts[row],irf)
    return shifts_to_depth(peaks),peaks
