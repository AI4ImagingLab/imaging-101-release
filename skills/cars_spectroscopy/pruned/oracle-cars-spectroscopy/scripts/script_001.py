"""Reusable submission writer for the CARS temperature-inversion task.

Usage (from run root):
    python ${CLAUDE_SKILL_DIR}/scripts/script_001.py  # imported, not standalone

Provides write_submission(case_id, temperature_K, reconstructed_spectrum, nu_axis).
"""
import os
import numpy as np


def write_submission(case_id, temperature_K, reconstructed_spectrum, nu_axis, out_dir="outputs"):
    os.makedirs(out_dir, exist_ok=True)
    T = float(temperature_K)
    y = np.asarray(reconstructed_spectrum, dtype=np.float64).reshape(-1)
    nu = np.asarray(nu_axis, dtype=np.float64).reshape(-1)
    if y.shape[0] != 200 or nu.shape[0] != 200:
        raise ValueError("reconstructed_spectrum and nu_axis must have length 200")
    if not (np.all(np.isfinite(y)) and np.isfinite(T) and np.all(np.isfinite(nu))):
        raise ValueError("non-finite values in submission")
    path = os.path.join(out_dir, "%s.npz" % case_id)
    np.savez(path,
             estimated_temperature_K=np.array([T], dtype=np.float64),
             reconstructed_spectrum=y,
             nu_axis=nu)
    return path


if __name__ == "__main__":
    print("import write_submission from this module; not a standalone entry point")
