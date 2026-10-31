#!/usr/bin/env python
"""Generate test fixtures for electron_ptychography.

Creates fixtures in evaluation/fixtures/ that tests load:
  preprocessing/
    - calibration.npz
    - dp_mean.npz
    - dp_mask.npz
  solvers/
    - com_field.npz
    - fourier_integrate.npz
    - cross_correlate_shift.npz
    - electron_wavelength.npz
    - build_probe.npz
"""

import os
import sys
import numpy as np

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TASK_DIR)

DATA_DIR = os.path.join(TASK_DIR, "data")
FIXTURES_DIR = os.path.join(TASK_DIR, "evaluation", "fixtures")

PREPROCESSING_DIR = os.path.join(FIXTURES_DIR, "preprocessing")
SOLVERS_DIR = os.path.join(FIXTURES_DIR, "solvers")

os.makedirs(PREPROCESSING_DIR, exist_ok=True)
os.makedirs(SOLVERS_DIR, exist_ok=True)

from src.preprocessing import (
    load_data,
    load_metadata,
    calibrate_datacube,
    compute_dp_mean,
    compute_bf_mask,
)
from src.solvers import (
    _compute_com_field,
    _fourier_integrate,
    _cross_correlate_shift,
    _electron_wavelength,
    _build_probe,
)


def main():
    print("Generating fixtures for electron_ptychography ...")

    # ---------- Load data ----------
    print("  Loading data...")
    datacube, probe = load_data(DATA_DIR)
    meta = load_metadata(DATA_DIR)
    print(f"  datacube shape: {datacube.shape}, probe shape: {probe.shape}")

    # ========== Preprocessing fixtures ==========

    # --- calibration.npz ---
    radius, center = calibrate_datacube(
        datacube, probe,
        R_pixel_size=meta["R_pixel_size_A"],
        convergence_semiangle=meta["convergence_semiangle_mrad"],
    )
    np.savez(os.path.join(PREPROCESSING_DIR, "calibration.npz"),
             probe_radius=radius,
             probe_center_qx=center[0],
             probe_center_qy=center[1])
    print("  [OK] preprocessing/calibration.npz")

    # --- dp_mean.npz ---
    dp_mean = compute_dp_mean(datacube)
    np.savez(os.path.join(PREPROCESSING_DIR, "dp_mean.npz"),
             dp_mean=dp_mean)
    print("  [OK] preprocessing/dp_mean.npz")

    # --- dp_mask.npz ---
    dp_mask = compute_bf_mask(dp_mean, threshold=0.8)
    np.savez(os.path.join(PREPROCESSING_DIR, "dp_mask.npz"),
             dp_mask=dp_mask)
    print("  [OK] preprocessing/dp_mask.npz")

    # ========== Solvers fixtures ==========

    # --- com_field.npz ---
    # Use a small synthetic datacube for the fixture (4x4x10x10)
    rng = np.random.default_rng(42)
    small_dc = rng.random((4, 4, 10, 10)).astype(np.float64)
    small_mask = np.ones((10, 10), dtype=bool)
    com_x, com_y = _compute_com_field(small_dc, mask=small_mask)
    np.savez(os.path.join(SOLVERS_DIR, "com_field.npz"),
             input_datacube=small_dc,
             input_mask=small_mask,
             output_com_x=com_x,
             output_com_y=com_y)
    print("  [OK] solvers/com_field.npz")

    # --- fourier_integrate.npz ---
    rng2 = np.random.default_rng(7)
    com_x_fi = rng2.standard_normal((8, 8))
    com_y_fi = rng2.standard_normal((8, 8))
    dx, dy = 1.5, 1.5
    phase_fi = _fourier_integrate(com_x_fi, com_y_fi, dx=dx, dy=dy)
    np.savez(os.path.join(SOLVERS_DIR, "fourier_integrate.npz"),
             input_com_x=com_x_fi,
             input_com_y=com_y_fi,
             param_dx=dx,
             param_dy=dy,
             output_phase=phase_fi)
    print("  [OK] solvers/fourier_integrate.npz")

    # --- cross_correlate_shift.npz ---
    rng3 = np.random.default_rng(0)
    img1 = rng3.random((16, 16))
    # Shift img1 by (2, 3) pixels
    img2 = np.roll(np.roll(img1, 2, axis=0), 3, axis=1)
    sx, sy = _cross_correlate_shift(img1, img2)
    np.savez(os.path.join(SOLVERS_DIR, "cross_correlate_shift.npz"),
             input_img1=img1,
             input_img2=img2,
             output_shift_x=sx,
             output_shift_y=sy)
    print("  [OK] solvers/cross_correlate_shift.npz")

    # --- electron_wavelength.npz ---
    energies = np.array([80000.0, 200000.0, 300000.0])
    wavelengths = np.array([_electron_wavelength(e) for e in energies])
    np.savez(os.path.join(SOLVERS_DIR, "electron_wavelength.npz"),
             param_energies=energies,
             output_wavelengths=wavelengths)
    print("  [OK] solvers/electron_wavelength.npz")

    # --- build_probe.npz ---
    # Use the actual vacuum probe from data
    defocus = meta.get("defocus_A", 100.0)
    energy = meta["energy_eV"]
    R_pixel_size = meta["R_pixel_size_A"]
    probe_complex = _build_probe(probe, defocus=defocus, energy=energy,
                                  R_pixel_size=R_pixel_size)
    np.savez(os.path.join(SOLVERS_DIR, "build_probe.npz"),
             input_vacuum_probe=probe,
             param_defocus=defocus,
             param_energy=energy,
             param_R_pixel_size=R_pixel_size,
             output_probe=probe_complex)
    print("  [OK] solvers/build_probe.npz")

    print("Done! All fixtures saved to", FIXTURES_DIR)


if __name__ == "__main__":
    main()
