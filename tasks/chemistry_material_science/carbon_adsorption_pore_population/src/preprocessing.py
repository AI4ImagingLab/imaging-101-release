"""Public real adsorption observations and independently supplied kernel."""
import numpy as np
def load_observations(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k][0].copy() for k in z.files}
def calibration_with_origin(pressure,kernel):
    return np.r_[0.,pressure],np.vstack([np.zeros(kernel.shape[1]),kernel])
def width_steps(widths):return np.ediff1d(widths,to_begin=widths[0])
