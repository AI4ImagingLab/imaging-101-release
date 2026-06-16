#!/usr/bin/env python
"""Generate per-function test fixtures from the cleaned Shack-Hartmann code.

Fixtures are saved under evaluation/fixtures/<module>/<function>.npz following the
naming convention:
    input_*   – function inputs
    config_*  – scalar settings (rcond, wavelength, n_valid_subaps)
    output_*  – golden outputs produced by the current cleaned src/ functions

All src/ functions here are deterministic, so the fixtures lock their numerical
behaviour and the unit tests compare against them with rtol=1e-10.

Usage
-----
    cd tasks/shack-hartmann
    python generate_fixtures.py
"""

import os
import sys
import json
import numpy as np

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TASK_DIR)

FIX_DIR = os.path.join(TASK_DIR, 'evaluation', 'fixtures')
DATA_DIR = os.path.join(TASK_DIR, 'data')

from src.preprocessing import load_raw_data, load_ground_truth
from src.physics_model import (
    estimate_slopes,
    compute_reconstruction_matrix,
    reconstruct_wavefront,
    compute_ncc,
    compute_nrmse,
)
from src.solvers import reconstruct_all_levels


def _save(module, name, **arrays):
    out_dir = os.path.join(FIX_DIR, module)
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, name), **arrays)
    print(f"  wrote {module}/{name}")


print("Loading data ...")
raw = load_raw_data(os.path.join(DATA_DIR, 'raw_data.npz'))
gt = load_ground_truth(os.path.join(DATA_DIR, 'ground_truth.npz'))
meta = json.load(open(os.path.join(DATA_DIR, 'meta_data.json')))

n_valid_subaps = meta['wavefront_sensor']['n_valid_subaps']
wavelength = meta['wavefront_sensor']['wavelength_wfs_m']
RCOND = 1e-3

response_matrix = raw['response_matrix'].astype(np.float64)
dm_modes = raw['dm_modes'].astype(np.float64)
aperture = raw['aperture'].astype(np.float64)
mask = aperture > 0.5

# ── preprocessing fixtures ────────────────────────────────────────────────────
print("Generating preprocessing fixtures ...")

_save('preprocessing', 'load_raw_data.npz',
      output_keys=np.array(sorted(raw.keys())),
      output_response_matrix_shape=np.array(raw['response_matrix'].shape),
      output_wfs_images_shape=np.array(raw['wfs_images'].shape),
      output_ref_image_shape=np.array(raw['ref_image'].shape),
      output_subap_map_shape=np.array(raw['subap_map'].shape),
      output_dm_modes_shape=np.array(raw['dm_modes'].shape),
      output_aperture_shape=np.array(raw['aperture'].shape),
      output_subap_map_max=np.int64(raw['subap_map'].max()),
      output_response_matrix_first=raw['response_matrix'][:3, :3].astype(np.float64))

_save('preprocessing', 'load_ground_truth.npz',
      output_keys=np.array(sorted(gt.keys())),
      output_wavefront_phases_shape=np.array(gt['wavefront_phases'].shape),
      output_wavefront_phases_first=gt['wavefront_phases'][0, :5].astype(np.float64))

# ── physics_model fixtures ────────────────────────────────────────────────────
print("Generating physics_model fixtures ...")

# estimate_slopes — real level-0 detector image
wfs0 = raw['wfs_images'][0].astype(np.float64)
ref_img = raw['ref_image'].astype(np.float64)
cx = raw['detector_coords_x'].astype(np.float64)
cy = raw['detector_coords_y'].astype(np.float64)
subap_map = raw['subap_map']
slopes0 = estimate_slopes(wfs0, ref_img, cx, cy, subap_map, n_valid_subaps)
_save('physics_model', 'estimate_slopes.npz',
      input_wfs_image=wfs0,
      input_ref_image=ref_img,
      input_detector_coords_x=cx,
      input_detector_coords_y=cy,
      input_subap_map=subap_map,
      config_n_valid_subaps=np.int64(n_valid_subaps),
      output_slopes=slopes0)

# compute_reconstruction_matrix
M = compute_reconstruction_matrix(response_matrix, RCOND)
_save('physics_model', 'compute_reconstruction_matrix.npz',
      input_response_matrix=response_matrix,
      config_rcond=np.float64(RCOND),
      output_matrix=M)

# reconstruct_wavefront
phi0 = reconstruct_wavefront(slopes0, M, dm_modes, wavelength)
_save('physics_model', 'reconstruct_wavefront.npz',
      input_slopes=slopes0,
      input_reconstruction_matrix=M,
      input_dm_modes=dm_modes.astype(np.float32),
      config_wavelength=np.float64(wavelength),
      output_phase=phi0)

# compute_ncc / compute_nrmse — real recon vs ground-truth phase, masked
gt0 = gt['wavefront_phases'][0].astype(np.float64)
ncc0 = compute_ncc(phi0, gt0, mask)
nrmse0 = compute_nrmse(phi0, gt0, mask)
_save('physics_model', 'compute_ncc.npz',
      input_estimate=phi0, input_reference=gt0, input_mask=mask,
      output_ncc=np.float64(ncc0))
_save('physics_model', 'compute_nrmse.npz',
      input_estimate=phi0, input_reference=gt0, input_mask=mask,
      output_nrmse=np.float64(nrmse0))

# ── solvers fixtures ──────────────────────────────────────────────────────────
print("Generating solvers fixtures ...")
result = reconstruct_all_levels(
    raw['wfs_images'].astype(np.float64),
    ref_img, cx, cy, subap_map,
    response_matrix, dm_modes, aperture, wavelength,
    n_valid_subaps=n_valid_subaps, rcond=RCOND,
    ground_truth_phases=gt['wavefront_phases'].astype(np.float64),
)
_save('solvers', 'reconstruct_all_levels.npz',
      output_reconstructed_phases=result['reconstructed_phases'].astype(np.float64),
      output_ncc_per_level=result['ncc_per_level'],
      output_nrmse_per_level=result['nrmse_per_level'])

print("Done.")
