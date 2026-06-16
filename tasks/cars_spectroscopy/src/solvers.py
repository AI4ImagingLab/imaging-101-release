import numpy as np
from scipy.optimize import least_squares
from src.physics_model import forward_operator


def run_inversion(measured_signal, nu_axis, initial_guesses):
    """
    Fits the measured signal using the forward operator and scipy least_squares.

    Args:
        measured_signal (np.ndarray): The observed data.
        nu_axis (np.ndarray): Wavenumber axis.
        initial_guesses (dict): Initial values for optimization.

    Returns:
        dict: The best fit parameters and the fitted spectrum.
    """
    # Parameter order: [temperature, x_mol, nu_shift, slit_width]
    x0 = np.array([
        initial_guesses['temperature'],
        initial_guesses.get('x_mol', 0.79),
        0.0,    # nu_shift
        0.5,    # slit_width
    ])
    bounds_lower = np.array([300.0,  0.1, -5.0, 0.01])
    bounds_upper = np.array([3500.0, 1.0,  5.0, 5.0])

    def residual_func(x):
        temperature, x_mol, nu_shift, slit_width = x
        params = {
            'nu': nu_axis,
            'temperature': temperature,
            'pressure': initial_guesses.get('pressure', 1.0),
            'x_mol': x_mol,
            'species': 'N2',
            'pump_lw': initial_guesses.get('pump_lw', 1.0),
            'nu_shift': nu_shift,
            'nu_stretch': 1.0,
            'slit_params': [slit_width, 2.0, 0, 0]
        }
        return forward_operator(params) - measured_signal

    result = least_squares(residual_func, x0, bounds=(bounds_lower, bounds_upper),
                           method='trf', max_nfev=100)

    best_vals = result.x
    best_params = {
        'temperature': float(best_vals[0]),
        'x_mol': float(best_vals[1]),
        'nu_shift': float(best_vals[2]),
        'nu_stretch': 1.0,
        'slit_width': float(best_vals[3]),
    }

    # Generate best-fit spectrum
    final_params = {
        'nu': nu_axis,
        'temperature': best_params['temperature'],
        'pressure': initial_guesses.get('pressure', 1.0),
        'x_mol': best_params['x_mol'],
        'species': 'N2',
        'pump_lw': initial_guesses.get('pump_lw', 1.0),
        'nu_shift': best_params['nu_shift'],
        'nu_stretch': best_params['nu_stretch'],
        'slit_params': [best_params['slit_width'], 2.0, 0, 0]
    }
    y_pred = forward_operator(final_params)

    return {
        'best_params': best_params,
        'y_pred': y_pred,
        'cost': float(result.cost),
        'nfev': result.nfev,
        'success': result.success,
    }
