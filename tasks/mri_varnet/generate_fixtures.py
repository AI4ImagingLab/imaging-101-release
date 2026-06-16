"""Generate minimal fixtures for mri_varnet.

Tests use reference_outputs and direct src imports, not fixture files.
This script runs basic preprocessing functions on small synthetic data
and saves minimal output to evaluation/fixtures/ for reference.
"""
import os
import sys
import pathlib
import numpy as np
import torch

TASK_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(TASK_DIR))

from src.preprocessing import EquiSpacedMaskFunc

FIXTURES_DIR = TASK_DIR / "evaluation" / "fixtures"


def main():
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    # Small synthetic k-space
    rng = np.random.default_rng(42)
    Nc, Nx, Ny = 4, 32, 32
    kspace = rng.standard_normal((Nc, Nx, Ny)) + 1j * rng.standard_normal((Nc, Nx, Ny))

    # Test mask generation
    mask_func = EquiSpacedMaskFunc(
        center_fractions=[0.08],
        accelerations=[4],
        seed=42,
    )
    mask, num_low = mask_func(shape=(Nc, Nx, Ny, 1), seed=42)

    np.savez(
        FIXTURES_DIR / "basic_pipeline.npz",
        kspace_real=kspace.real,
        kspace_imag=kspace.imag,
        mask=mask.numpy(),
        num_low_frequencies=np.array(num_low),
    )
    print(f"Saved fixtures to {FIXTURES_DIR / 'basic_pipeline.npz'}")


if __name__ == "__main__":
    main()
