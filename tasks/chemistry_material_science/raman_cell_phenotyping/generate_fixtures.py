"""
Generate evaluation fixtures for raman_cell_phenotyping task.

Run from the task directory:
    cd raman_cell_phenotyping
    python generate_fixtures.py
"""

import os
import sys
import numpy as np

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TASK_DIR)

from src.physics_model import forward, reconstruction_error
from src.preprocessing import crop, despike, denoise_savgol, normalise_minmax
from src.visualization import compute_ncc, compute_nrmse

FIXTURE_DIR = os.path.join(TASK_DIR, "evaluation", "fixtures")
os.makedirs(FIXTURE_DIR, exist_ok=True)


def gen_physics_model():
    """Fixture: physics_model.npz

    Keys: endmembers, abundances, input_observed, output_forward, output_rmse

    Test expectations:
      - forward(endmembers, abundances) shape == (20, 50) and matches output_forward
      - reconstruction_error(input_observed, endmembers, abundances) matches output_rmse
      - residual(predicted, endmembers, abundances) == 0 for perfect prediction
    """
    rng = np.random.default_rng(42)

    K, B = 3, 50   # 3 endmembers, 50 bands
    N = 20          # 20 pixels

    endmembers = rng.random((K, B))  # (K, B)

    # Random abundances that are non-negative and sum to 1
    raw = rng.random((N, K))
    abundances = raw / raw.sum(axis=1, keepdims=True)  # (N, K)

    # Forward model: predicted = abundances @ endmembers => (N, B)
    output_forward = forward(endmembers, abundances)

    # Create observed with some noise for reconstruction_error
    noise = rng.normal(0, 0.01, size=(N, B))
    input_observed = output_forward + noise

    output_rmse = reconstruction_error(input_observed, endmembers, abundances)

    path = os.path.join(FIXTURE_DIR, "physics_model.npz")
    np.savez(path,
             endmembers=endmembers,
             abundances=abundances,
             input_observed=input_observed,
             output_forward=output_forward,
             output_rmse=np.float64(output_rmse))
    print(f"  Saved {path}")
    print(f"    endmembers={endmembers.shape}, abundances={abundances.shape}")
    print(f"    output_forward={output_forward.shape}, rmse={output_rmse:.6f}")


def gen_preprocessing_crop():
    """Fixture: preprocessing_crop.npz

    Keys: input_data, input_axis, output_data, output_axis
    Crop region: (700, 1800)
    """
    rng = np.random.default_rng(123)

    # Spectral axis spanning 400..2200 cm^-1, 200 bands
    input_axis = np.linspace(400, 2200, 200)
    # Data shape: (5, 200) -- 5 spectra
    input_data = rng.random((5, 200))

    output_data, output_axis = crop(input_data, input_axis, (700, 1800))

    path = os.path.join(FIXTURE_DIR, "preprocessing_crop.npz")
    np.savez(path,
             input_data=input_data,
             input_axis=input_axis,
             output_data=output_data,
             output_axis=output_axis)
    print(f"  Saved {path}  output_data={output_data.shape} output_axis={output_axis.shape}")


def gen_preprocessing_despike():
    """Fixture: preprocessing_despike.npz

    Keys: input_data, input_axis, output_data
    Test expects: despike(input_data[np.newaxis], input_axis) result[0] matches output_data
    Also: input_data[50] has an artificial spike that should be reduced
    """
    rng = np.random.default_rng(456)

    B = 100
    input_axis = np.linspace(700, 1800, B)
    # Smooth base spectrum
    input_data = np.sin(np.linspace(0, 4 * np.pi, B)) * 0.5 + 1.0
    input_data += rng.normal(0, 0.02, B)  # small noise

    # Add artificial cosmic-ray spike at index 50
    input_data[50] += 10.0

    # Despike expects (..., B) shape. Test passes input_data[np.newaxis] => (1, B)
    result, _ = despike(input_data[np.newaxis], input_axis)
    output_data = result[0]

    path = os.path.join(FIXTURE_DIR, "preprocessing_despike.npz")
    np.savez(path,
             input_data=input_data,
             input_axis=input_axis,
             output_data=output_data)
    print(f"  Saved {path}  spike reduced: {input_data[50]:.2f} -> {output_data[50]:.2f}")


def gen_preprocessing_savgol():
    """Fixture: preprocessing_savgol.npz

    Keys: input_data, input_axis, output_data
    Test: denoise_savgol(input_data[np.newaxis], input_axis) result[0] matches output_data
    """
    rng = np.random.default_rng(789)

    B = 100
    input_axis = np.linspace(700, 1800, B)
    input_data = np.sin(np.linspace(0, 4 * np.pi, B)) + rng.normal(0, 0.1, B)

    result, _ = denoise_savgol(input_data[np.newaxis], input_axis)
    output_data = result[0]

    path = os.path.join(FIXTURE_DIR, "preprocessing_savgol.npz")
    np.savez(path,
             input_data=input_data,
             input_axis=input_axis,
             output_data=output_data)
    print(f"  Saved {path}  input_data={input_data.shape} output_data={output_data.shape}")


def gen_preprocessing_minmax():
    """Fixture: preprocessing_minmax.npz

    Keys: input_data, output_data
    Test: normalise_minmax(input_data, np.arange(B), pixelwise=False) matches output_data
    """
    rng = np.random.default_rng(101)

    input_data = rng.random((5, 50)) * 10 + 2  # values in [2, 12]

    result, _ = normalise_minmax(input_data, np.arange(50), pixelwise=False)
    output_data = result

    path = os.path.join(FIXTURE_DIR, "preprocessing_minmax.npz")
    np.savez(path,
             input_data=input_data,
             output_data=output_data)
    print(f"  Saved {path}  range: [{output_data.min():.4f}, {output_data.max():.4f}]")


def gen_visualization_metrics():
    """Fixture: visualization_metrics.npz

    Keys: input_estimate, input_reference, output_ncc, output_nrmse
    """
    rng = np.random.default_rng(42)
    input_estimate = rng.random((10, 10))
    input_reference = rng.random((10, 10))

    output_ncc = compute_ncc(input_estimate, input_reference)
    output_nrmse = compute_nrmse(input_estimate, input_reference)

    path = os.path.join(FIXTURE_DIR, "visualization_metrics.npz")
    np.savez(path,
             input_estimate=input_estimate,
             input_reference=input_reference,
             output_ncc=np.float64(output_ncc),
             output_nrmse=np.float64(output_nrmse))
    print(f"  Saved {path}  ncc={output_ncc:.6f} nrmse={output_nrmse:.6f}")


if __name__ == "__main__":
    print("Generating raman_cell_phenotyping fixtures...")
    gen_physics_model()
    gen_preprocessing_crop()
    gen_preprocessing_despike()
    gen_preprocessing_savgol()
    gen_preprocessing_minmax()
    gen_visualization_metrics()
    print("Done!")
