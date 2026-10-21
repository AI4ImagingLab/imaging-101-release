"""Native ADC and independent reference background conventions."""
from pathlib import Path
import numpy as np
def load_data(path):
    with np.load(Path(path),allow_pickle=False) as f:
        return {k:f[k][0].copy() for k in ['fringes','background','chirp','dispersion']}
def subtract_background(fringes,background):
    # Preserve original int64 ADC subtraction: fractional mean ADU is truncated
    # toward zero upon assignment, before the Fourier processing.
    corrected=fringes.astype(np.int64)
    corrected[:]=corrected-background[:,:,None]
    return corrected
