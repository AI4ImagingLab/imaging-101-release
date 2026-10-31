"""Locally specified bounded nonlinear least squares for hyperfine map imaging."""
import numpy as np
from scipy.optimize import least_squares
from .physics_model import brightness
from .preprocessing import estimate_noise
INITIAL = np.array([5., .5, 3., 1.])
OUTPUT_KEYS = ('excitation_temperature','optical_depth','velocity','dispersion')

def fit_spectrum(frequency, spectrum, noise):
    """Return (four parameters, chi-square); preserve upstream objective and bounds."""
    def residual(parameters):
        return (brightness(frequency, parameters)-spectrum)/noise
    result = least_squares(residual, INITIAL, bounds=([1e-5,0,-np.inf,1e-12],np.inf),
                           method='trf', jac='3-point', x_scale='jac',
                           ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=1000)
    if not result.success:
        raise RuntimeError('Fit did not converge: '+result.message)
    return result.x, float(result.fun@result.fun)

def reconstruct(cube, frequency):
    """Return dict of batch-first (1,3,3) float64 maps and (3,3) chi-square."""
    noise = estimate_noise(cube, frequency)
    parameters = np.empty((4,)+cube.shape[1:], dtype=np.float64)
    chi_square = np.empty(cube.shape[1:])
    for y,x in np.ndindex(cube.shape[1:]):
        parameters[:,y,x], chi_square[y,x] = fit_spectrum(frequency,cube[:,y,x],noise[y,x])
    return {key:parameters[i][None] for i,key in enumerate(OUTPUT_KEYS)}, chi_square
