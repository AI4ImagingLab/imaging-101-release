"""Generate minimal fixtures for mri_grappa.

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

from src.physics_model import centered_fft2, centered_ifft2, sos_combine

FIXTURES_DIR = TASK_DIR / "evaluation" / "fixtures"


def main():
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)

    # Small synthetic k-space data
    Nx, Ny, Nc = 32, 32, 4
    kspace = rng.standard_normal((Nx, Ny, Nc)) + 1j * rng.standard_normal((Nx, Ny, Nc))

    # Test FFT roundtrip
    img = centered_ifft2(kspace)
    kspace_rt = centered_fft2(img)

    # Test SOS combine
    sos = sos_combine(img)

    np.savez(
        FIXTURES_DIR / "basic_pipeline.npz",
        kspace=kspace,
        image=img,
        kspace_roundtrip=kspace_rt,
        sos_combine=sos,
    )
    print(f"Saved fixtures to {FIXTURES_DIR / 'basic_pipeline.npz'}")


if __name__ == "__main__":
    main()
