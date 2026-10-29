"""Spectral views and clearly named physical residual measurements."""
import numpy as np
import matplotlib.pyplot as plt


def amplitude_metrics(estimate,reference):
    """Standard flattened cosine NCC and reference-dynamic-range NRMSE."""
    x=np.asarray(estimate).ravel();r=np.asarray(reference).ravel()
    if np.ptp(r)<=0 or np.linalg.norm(r)==0:raise ValueError('Degenerate reference')
    norm=np.linalg.norm(x)*np.linalg.norm(r)
    return dict(ncc=float(x@r/norm) if norm else 0.,
                nrmse=float(np.sqrt(np.mean((x-r)**2))/np.ptp(r)))


def observation_metrics(measured,predicted):
    """Keep uncentered L2 and original centered relative-square conventions distinct."""
    r=np.asarray(measured);x=np.asarray(predicted)
    error=float(np.sum(abs(x-r)**2))
    return dict(relative_l2=float(np.sqrt(error/np.sum(abs(r)**2))),
        upstream_centered_relative_squared_error=float(error/np.sum(abs(r-r.mean())**2)),
        residual_sum_squares=error)


def spectrum(fid,spectral_width=10000.):
    """Return exact FFT-bin frequencies (Hz) and complex forward-FFT spectrum."""
    return np.fft.fftshift(np.fft.fftfreq(len(fid),d=1/spectral_width)),np.fft.fftshift(np.fft.fft(fid))
