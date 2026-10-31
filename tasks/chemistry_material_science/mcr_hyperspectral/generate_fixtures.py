"""Generate minimal fixtures for mcr_hyperspectral.

Tests import src directly and don't load fixture files.
This script runs basic src functions on small data and saves minimal output
to evaluation/fixtures/ for reference.
"""
import os
import sys
import pathlib
import numpy as np

TASK_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(TASK_DIR))

from src.generate_data import make_spectral_components, make_concentration_maps, generate_hsi
from src.physics_model import forward, residual, mse
from src.preprocessing import estimate_initial_spectra

FIXTURES_DIR = TASK_DIR / "evaluation" / "fixtures"


def main():
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    # Small synthetic data
    n_freq = 50
    n_components = 3
    M, N = 10, 20
    wn = np.linspace(400, 2800, n_freq)

    spectra = make_spectral_components(wn, [1200, 1600, 2000], [300, 500, 300])
    rng = np.random.RandomState(42)
    np.random.seed(42)
    conc = make_concentration_maps(M, N, n_components, rng)
    hsi_clean, hsi_noisy = generate_hsi(conc, spectra, noise_std=250.0, rng=rng)

    # forward model
    C = conc.reshape(-1, n_components)
    D = forward(C, spectra)
    R = residual(C, spectra, hsi_clean)
    err = mse(C, spectra, hsi_clean)

    # initial spectra estimate
    init_spectra = estimate_initial_spectra(hsi_noisy, n_components)

    np.savez(
        FIXTURES_DIR / "basic_pipeline.npz",
        spectra=spectra,
        concentrations=conc,
        hsi_clean=hsi_clean,
        hsi_noisy=hsi_noisy,
        forward_D=D,
        residual_R=R,
        mse=np.array(err),
        initial_spectra=init_spectra,
        wn=wn,
    )
    print(f"Saved fixtures to {FIXTURES_DIR / 'basic_pipeline.npz'}")


if __name__ == "__main__":
    main()
