"""Generate minimal fixtures for mri_sense.

Tests use reference_outputs and direct src imports, not fixture files.
This script runs basic src functions on small synthetic data and saves
minimal output to evaluation/fixtures/ for reference.
"""
import os
import sys
import pathlib
import numpy as np

TASK_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(TASK_DIR))

from src.physics_model import centered_fft2, centered_ifft2, sos_combine, sense_forward, sense_adjoint

FIXTURES_DIR = TASK_DIR / "evaluation" / "fixtures"


def main():
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)

    # Small synthetic data
    Nx, Ny, Nc = 32, 32, 4
    image = rng.standard_normal((Nx, Ny)) + 1j * rng.standard_normal((Nx, Ny))
    sens = rng.standard_normal((Nx, Ny, Nc)) + 1j * rng.standard_normal((Nx, Ny, Nc))
    mask = np.ones(Nx, dtype=bool)
    mask[::2] = False

    # Test SENSE forward/adjoint
    y = sense_forward(image, sens, mask)
    x_adj = sense_adjoint(y, sens)

    # Test SOS combine
    kspace = rng.standard_normal((Nx, Ny, Nc)) + 1j * rng.standard_normal((Nx, Ny, Nc))
    img_mc = centered_ifft2(kspace)
    sos = sos_combine(img_mc)

    np.savez(
        FIXTURES_DIR / "basic_pipeline.npz",
        image=image,
        sensitivity_maps=sens,
        mask=mask,
        sense_forward=y,
        sense_adjoint=x_adj,
        sos_combine=sos,
    )
    print(f"Saved fixtures to {FIXTURES_DIR / 'basic_pipeline.npz'}")


if __name__ == "__main__":
    main()
