#!/usr/bin/env python
"""Generate test fixtures for fourier_ptychography.

Creates fixtures in evaluation/fixtures/ that the tests load:
  - compute_pupil_mask.npz
  - compute_kspace_shift.npz
  - fpm_forward_single.npz
  - reconstruction_error.npz
  - fft2c_roundtrip.npz
  - circ_mask.npz
  - output_gaussian2D.npy
"""

import os
import sys
import numpy as np

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TASK_DIR)

FIXTURES_DIR = os.path.join(TASK_DIR, "evaluation", "fixtures")
os.makedirs(FIXTURES_DIR, exist_ok=True)

from src.physics_model import (
    compute_pupil_mask,
    compute_kspace_shift,
    fpm_forward_single,
)
from src.solvers import compute_reconstruction_error
from src.utils import fft2c, ifft2c, circ, gaussian2D


def main():
    print("Generating fixtures for fourier_ptychography ...")

    # --- compute_pupil_mask ---
    Nd, dxp, wavelength, NA = 64, 1.625e-6, 625e-9, 0.1
    pupil = compute_pupil_mask(Nd, dxp, wavelength, NA)
    np.savez(os.path.join(FIXTURES_DIR, "compute_pupil_mask.npz"),
             Nd=Nd, dxp=dxp, wavelength=wavelength, NA=NA,
             output=pupil)
    print("  [OK] compute_pupil_mask.npz")

    # --- compute_kspace_shift ---
    led_pos = np.array([4e-3, -2e-3])
    z_led = 60e-3
    shift = compute_kspace_shift(led_pos, z_led, wavelength, Nd, dxp)
    np.savez(os.path.join(FIXTURES_DIR, "compute_kspace_shift.npz"),
             led_pos=led_pos, z_led=z_led,
             wavelength=wavelength, Nd=Nd, dxp=dxp,
             output=shift)
    print("  [OK] compute_kspace_shift.npz")

    # --- fpm_forward_single ---
    rng = np.random.default_rng(0)
    No = 128
    Nd_fwd = 32
    obj_spectrum = rng.standard_normal((No, No)) + 1j * rng.standard_normal((No, No))
    pupil_fwd = compute_pupil_mask(Nd_fwd, 1.625e-6, 625e-9, 0.1)
    shift_px = np.array([0.0, 0.0])
    lr_image, lr_field = fpm_forward_single(obj_spectrum, pupil_fwd, shift_px, Nd_fwd)
    np.savez(os.path.join(FIXTURES_DIR, "fpm_forward_single.npz"),
             obj_spectrum=obj_spectrum,
             pupil=pupil_fwd,
             output_image=lr_image)
    print("  [OK] fpm_forward_single.npz")

    # --- reconstruction_error ---
    rng2 = np.random.default_rng(42)
    I_meas = rng2.random((4, 16, 16)).astype(np.float32)
    I_est = rng2.random((4, 16, 16)).astype(np.float32)
    err = compute_reconstruction_error(I_meas, I_est)
    np.savez(os.path.join(FIXTURES_DIR, "reconstruction_error.npz"),
             I_meas=I_meas, I_est=I_est,
             output=err)
    print("  [OK] reconstruction_error.npz")

    # --- fft2c_roundtrip ---
    rng3 = np.random.default_rng(7)
    input_field = rng3.standard_normal((16, 16)) + 1j * rng3.standard_normal((16, 16))
    output_fft = fft2c(input_field)
    np.savez(os.path.join(FIXTURES_DIR, "fft2c_roundtrip.npz"),
             input=input_field,
             output=output_fft)
    print("  [OK] fft2c_roundtrip.npz")

    # --- circ_mask ---
    n = 64
    x = np.linspace(-n / 2, n / 2, n)
    X, Y = np.meshgrid(x, x)
    D = 30.0
    mask = circ(X, Y, D)
    np.savez(os.path.join(FIXTURES_DIR, "circ_mask.npz"),
             X=X, Y=Y, D=D,
             output=mask)
    print("  [OK] circ_mask.npz")

    # --- output_gaussian2D ---
    g = gaussian2D(15, 3.0)
    np.save(os.path.join(FIXTURES_DIR, "output_gaussian2D.npy"), g)
    print("  [OK] output_gaussian2D.npy")

    print("Done! All fixtures saved to", FIXTURES_DIR)


if __name__ == "__main__":
    main()
