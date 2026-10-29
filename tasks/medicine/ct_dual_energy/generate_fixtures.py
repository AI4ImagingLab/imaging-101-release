"""Generate test fixtures for all src/ modules."""

import os
import sys
import numpy as np

task_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, task_dir)

from src import physics_model as pm
from src.generate_data import create_phantom
from src.solvers import gauss_newton_decompose, reconstruct_material_maps
from src.visualization import compute_ncc, compute_nrmse

fixture_dir = os.path.join(task_dir, "evaluation", "fixtures")
os.makedirs(fixture_dir, exist_ok=True)


def main():
    rng = np.random.default_rng(123)

    # ---- physics_model fixtures ----
    energies = np.arange(20, 151, dtype=np.float64)
    from src.generate_data import get_attenuation_coefficients
    mus = get_attenuation_coefficients(energies)
    spectra = pm.get_spectra(energies)

    np.savez(os.path.join(fixture_dir, "physics_model_attenuation.npz"),
             input_energies=energies,
             output_mus=mus)

    np.savez(os.path.join(fixture_dir, "physics_model_spectra.npz"),
             input_energies=energies,
             output_spectra=spectra)

    # Small test for polychromatic forward
    small_sinos = rng.uniform(0.5, 5.0, size=(2, 4, 3))  # (nMats, nBins, nAngles)
    counts = pm.polychromatic_forward(small_sinos, spectra, mus, dE=1.0)
    np.savez(os.path.join(fixture_dir, "physics_model_forward.npz"),
             input_material_sinograms=small_sinos,
             input_spectra=spectra,
             input_mus=mus,
             output_counts=counts)

    # Radon transform fixture
    phantom_small = np.zeros((32, 32), dtype=np.float64)
    phantom_small[8:24, 8:24] = 1.0
    theta_small = np.array([0, 45, 90, 135], dtype=np.float64)
    sino_small = pm.radon_transform(phantom_small, theta_small)
    np.savez(os.path.join(fixture_dir, "physics_model_radon.npz"),
             input_image=phantom_small,
             input_theta=theta_small,
             output_sinogram=sino_small)

    # ---- generate_data fixtures ----
    tissue, bone = create_phantom(128)
    np.savez(os.path.join(fixture_dir, "generate_data_phantom.npz"),
             output_tissue=tissue,
             output_bone=bone)

    # ---- solvers fixtures (small problem) ----
    # Create a tiny 2-bin, 2-angle test case
    tissue_tiny = np.array([[0.5, 0.0], [1.0, 0.3]])
    bone_tiny = np.array([[0.0, 0.2], [0.5, 0.0]])
    mat_sinos_tiny = np.stack([tissue_tiny, bone_tiny], axis=0)  # (2, 2, 2)
    counts_tiny = pm.polychromatic_forward(mat_sinos_tiny, spectra, mus, dE=1.0)
    # Add small Poisson noise
    counts_noisy = rng.poisson(np.clip(counts_tiny, 1, None)).astype(np.float64)
    decomp = gauss_newton_decompose(counts_noisy, spectra, mus, n_iters=30,
                                     dE=1.0, eps=1e-6, verbose=False)
    np.savez(os.path.join(fixture_dir, "solvers_decompose.npz"),
             input_sinograms=counts_noisy,
             input_spectra=spectra,
             input_mus=mus,
             param_n_iters=np.array(30),
             output_material_sinograms=decomp,
             config_true_material_sinograms=mat_sinos_tiny)

    # ---- visualization fixtures ----
    a = rng.uniform(0, 1, size=100)
    b = a + rng.normal(0, 0.05, size=100)
    np.savez(os.path.join(fixture_dir, "visualization_metrics.npz"),
             input_a=a,
             input_b=b,
             output_ncc=np.array(compute_ncc(a, b)),
             output_nrmse=np.array(compute_nrmse(a, b)))

    print("Fixtures generated successfully.")


if __name__ == "__main__":
    main()
