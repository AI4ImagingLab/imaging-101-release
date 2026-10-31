"""
Universal back-projection reconstruction for photoacoustic tomography.

Implements the Xu & Wang (2005) algorithm for recovering the initial pressure
distribution from boundary acoustic measurements.
"""

import numpy as np
from numpy.fft import fft, ifft


def universal_back_projection(prec, xd, yd, t, z_target, c=1484.0,
                               resolution=500e-6, det_area=(2e-3)**2,
                               nfft=2048):
    """Reconstruct initial pressure distribution via universal back-projection.

    Parameters
    ----------
    prec : np.ndarray, shape (n_time, n_det_x, n_det_y)
        Recorded pressure time-series at each detector.
    xd : np.ndarray, shape (n_det_x,)
        X-coordinates of detector centres in meters.
    yd : np.ndarray, shape (n_det_y,)
        Y-coordinates of detector centres in meters.
    t : np.ndarray, shape (n_time,)
        Time vector in seconds.
    z_target : float
        Z-coordinate of the reconstruction plane in meters.
    c : float
        Sound speed in m/s.
    resolution : float
        Pixel size of reconstructed image in meters.
    det_area : float
        Active area of each detector in m^2.
    nfft : int
        FFT length for ramp filtering.

    Returns
    -------
    pfnorm : np.ndarray, shape (nx, ny, 1)
        Normalised reconstructed image.
    xf : np.ndarray, shape (nx,)
        X-coordinates of reconstruction pixels.
    yf : np.ndarray, shape (ny,)
        Y-coordinates of reconstruction pixels.
    zf : np.ndarray, shape (1,)
        Z-coordinate(s) of reconstruction plane.
    """
    xf = np.arange(xd[0], xd[-1] + resolution, resolution)
    yf = xf.copy()
    zf = np.array([z_target])

    Yf, Xf, Zf = np.meshgrid(yf, xf, zf)
    Zf2 = Zf**2

    fs = 1.0 / (t[1] - t[0])

    # Build frequency vector for ramp filter
    fv = fs / 2 * np.linspace(0, 1, int(nfft / 2 + 1))
    fv2 = -np.flipud(fv)
    fv2 = np.delete(fv2, 0)
    fv2 = np.delete(fv2, -1)
    fv3 = np.concatenate((fv, fv2), 0)
    k = 2 * np.pi * fv3 / c

    pnum = 0
    pden = 0

    for xi in range(len(xd)):
        X2 = (Xf - xd[xi])**2
        for yi in range(len(yd)):
            dist = np.sqrt(X2 + (Yf - yd[yi])**2 + Zf2)
            distind = np.round(dist * (fs / c)).astype(int)

            p = prec[:, xi, yi]
            pf = ifft(-1j * k * fft(p, nfft))
            b = 2 * p - 2 * t * c * pf[0:len(p)]
            b1 = b[distind]

            omega = (det_area / dist**2) * Zf / dist
            pnum = pnum + omega * b1
            pden = pden + omega

    pg = pnum / pden
    pgmax = pg[np.nonzero(np.abs(pg) == np.amax(abs(pg)))]
    pfnorm = np.real(pg / pgmax)

    return pfnorm, xf, yf, zf
