"""
CARS Spectroscopy Inversion Pipeline

Loads synthetic N2 CARS spectral data, runs a nonlinear least-squares
inversion to recover gas temperature, and evaluates the result.
"""
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')

# Ensure task root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.physics_model import forward_operator
from src.preprocessing import load_and_preprocess_data
from src.solvers import run_inversion
from src.visualization import compute_metrics, plot_inversion_result

# ==========================================
# Solver parameters (not in meta_data.json)
# ==========================================
_INVERSION_PARAMS = {
    'initial_temperature': 2000,
    'initial_x_mol': 0.79,
    'pressure': 10.0,
    'pump_lw': 1.0,
    'max_nfev': 100,
}


def main():
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    eval_dir = os.path.join(os.path.dirname(__file__), 'evaluation')
    ref_dir = os.path.join(eval_dir, 'reference_outputs')
    os.makedirs(ref_dir, exist_ok=True)

    # --------------------------------------------------
    # 1. Load data
    # --------------------------------------------------
    print("Loading data...")
    raw = np.load(os.path.join(data_dir, 'raw_data.npz'))
    gt = np.load(os.path.join(data_dir, 'ground_truth.npz'))

    measurements = raw['measurements'][0]  # (200,)
    nu_axis = raw['nu_axis'][0]            # (200,)
    gt_spectrum = gt['spectrum'][0]        # (200,)
    gt_temperature = float(gt['temperature'][0])
    gt_x_mol = float(gt['x_mol'][0])

    true_params = {
        'temperature': gt_temperature,
        'x_mol': gt_x_mol,
    }

    # --------------------------------------------------
    # 2. Preprocessing (already done at generation, but
    #    running through the pipeline for completeness)
    # --------------------------------------------------
    print("Preprocessing...")
    processed_signal, nu_axis_proc = load_and_preprocess_data(
        measurements, nu_axis, noise_level=0.0
    )

    # --------------------------------------------------
    # 3. Run inversion
    # --------------------------------------------------
    print("Running inversion...")
    initial_guesses = {
        'temperature': _INVERSION_PARAMS['initial_temperature'],
        'x_mol': _INVERSION_PARAMS['initial_x_mol'],
        'pressure': _INVERSION_PARAMS['pressure'],
        'pump_lw': _INVERSION_PARAMS['pump_lw'],
    }

    result = run_inversion(processed_signal, nu_axis_proc, initial_guesses)

    # --------------------------------------------------
    # 4. Evaluate
    # --------------------------------------------------
    print("Evaluating results...")
    metrics = compute_metrics(
        processed_signal, result['y_pred'],
        result['best_params'], params_true=true_params
    )

    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # Plot
    plot_inversion_result(
        nu_axis_proc, processed_signal, result['y_pred'],
        result['best_params'],
        y_ground_truth=gt_spectrum,
        params_true=true_params,
        save_path=os.path.join(ref_dir, 'inversion_result.png')
    )

    # --------------------------------------------------
    # 5. Save reference outputs and metrics
    # --------------------------------------------------
    np.savez(
        os.path.join(ref_dir, 'reconstruction.npz'),
        y_pred=result['y_pred'][np.newaxis, :],
        temperature_pred=np.array([result['best_params']['temperature']]),
    )

    metrics_json = {
        "baseline": [
            {
                "method": "scipy least_squares with G-matrix CARS forward model",
                "ncc_vs_ref": metrics.get('ncc', 0.0),
                "nrmse_vs_ref": metrics.get('nrmse', 1.0),
                "temperature_error_K": metrics.get('temperature_error_K', 0.0),
            }
        ],
        "ncc_boundary": round(0.9 * metrics.get('ncc', 0.0), 6),
        "nrmse_boundary": round(1.1 * metrics.get('nrmse', 1.0), 6),
        "temperature_error_K_boundary": 50.0,
    }

    with open(os.path.join(eval_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics_json, f, indent=2)

    print(f"\nReference outputs saved to {ref_dir}/")
    print(f"Metrics saved to {os.path.join(eval_dir, 'metrics.json')}")
    print("OPTIMIZATION_FINISHED_SUCCESSFULLY")


if __name__ == '__main__':
    main()
