"""Explicit independent-background subtraction for calibrated MPI spectra."""
import numpy as np
ROWS=7116
GRID=(19,19,19)

def load_observations(path):
    """Return an open safe NPZ handle; caller closes it after needed reads.
    Avoid materializing redundant public metadata or copying all600MB arrays.
    """
    return np.load(path,allow_pickle=False)

def prepare_system(calibration,calibration_background,copy=True):
    """Native complex64(1,M,6859),(1,M,362) -> S(M,6859)complex64.
    All independent delta background frames contribute to the rowwise mean.
    Spatial foreground columns already have x-fastest regular-grid order.
    copy=False subtracts in the caller-owned loaded calibration to bound RAM.
    """
    if calibration.dtype!=np.complex64 or calibration_background.dtype!=np.complex64:raise ValueError('Native calibration must be complex64')
    system=calibration[0].copy() if copy else calibration[0]
    system-=calibration_background[0].mean(axis=1)[:,None]
    return system

def prepare_measurement(foreground,background,averages=1000):
    """All selected acquired cycles -> mean foreground-minus-background.
    Inputs(1,1000,M)complex128 unscaled native rFFT amplitudes, unitsa.u.
    Canonical averages1000. Smaller values are labeled notebook diagnostics.
    Background always uses all1000 independent empty-bore cycles.
    """
    if not 1<=averages<=foreground.shape[1]:raise ValueError('Invalid foreground average count')
    return foreground[0,:averages].mean(axis=0)-background[0].mean(axis=0)
