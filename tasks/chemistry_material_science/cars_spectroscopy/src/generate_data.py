"""
Generates synthetic CARS spectroscopy data and saves to data/ directory.

Produces:
    data/raw_data.npz    - noisy measured spectrum and wavenumber axis
    data/ground_truth.npz - clean spectrum and true temperature

All physical parameters (pressure, wavenumber range, slit, noise level, ...)
are read from data/meta_data.json so this script and the rest of the pipeline
stay in sync.
"""
import os
import json
import numpy as np
from src.physics_model import forward_operator
from src.preprocessing import load_and_preprocess_data


def generate_and_save(data_dir='data', seed=None):
    """
    Generate synthetic N2 CARS data and save as .npz files.

    Args:
        data_dir (str): Output directory for data files.
        seed (int, optional): Random seed override. If None, uses the value
            from meta_data.json.
    """
    meta_path = os.path.join(data_dir, 'meta_data.json')
    with open(meta_path, 'r') as f:
        meta = json.load(f)

    nu_lo, nu_hi = meta['wavenumber_range_cm-1']
    n_pts = meta['num_spectral_points']
    nu_axis = np.linspace(nu_lo, nu_hi, n_pts)

    if seed is None:
        seed = meta['random_seed']
    np.random.seed(seed)

    true_params = {
        'nu': nu_axis,
        'temperature': 2400,
        'pressure': meta['pressure_bar'],
        'x_mol': 0.79,
        'species': meta['species'],
        'pump_lw': meta['pump_linewidth_cm-1'],
        'slit_params': [meta['slit_width_cm-1'], meta['slit_shape_factor'], 0, 0],
    }

    # Generate clean ground truth spectrum
    clean_signal = forward_operator(true_params)

    # Add noise to simulate measurement
    noisy_signal, nu_axis_proc = load_and_preprocess_data(
        clean_signal, nu_axis, noise_level=meta['noise_level']
    )

    os.makedirs(data_dir, exist_ok=True)

    # Save raw data (batch-first convention: shape (1, ...))
    np.savez(
        os.path.join(data_dir, 'raw_data.npz'),
        measurements=noisy_signal[np.newaxis, :],     # (1, 200)
        nu_axis=nu_axis[np.newaxis, :],                # (1, 200)
    )

    # Save ground truth
    np.savez(
        os.path.join(data_dir, 'ground_truth.npz'),
        spectrum=clean_signal[np.newaxis, :],           # (1, 200)
        temperature=np.array([true_params['temperature']]),  # (1,)
        x_mol=np.array([true_params['x_mol']]),              # (1,)
    )

    print(f"Data saved to {data_dir}/ at P={true_params['pressure']} bar")
    print(f"  raw_data.npz: measurements {noisy_signal.shape}, nu_axis {nu_axis.shape}")
    print(f"  ground_truth.npz: spectrum {clean_signal.shape}, temperature={true_params['temperature']}K")


if __name__ == '__main__':
    generate_and_save()
