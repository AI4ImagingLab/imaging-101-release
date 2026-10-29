"""Shack-Hartmann wavefront reconstruction pipeline.

Given raw SH-WFS detector images for N wavefront error levels, this script:

  1. Loads raw detector images and ground-truth phase maps.
  2. Extracts WFS slopes via weighted-centroid estimation.
  3. Builds the Tikhonov reconstruction matrix from the calibrated response matrix.
  4. Reconstructs the wavefront phase at each WFE level.
  5. Computes per-level NCC and NRMSE against ground truth, and measures timing.
  6. Saves results and visualisations.

Usage
-----
    cd tasks/shack-hartmann
    python main.py
"""

import os
import json
import numpy as np

os.environ['MPLBACKEND'] = 'Agg'
import matplotlib
matplotlib.use('Agg')

from src.preprocessing import load_raw_data, load_ground_truth
from src.physics_model  import compute_reconstruction_matrix
from src.solvers        import reconstruct_all_levels
from src.visualization  import (
    plot_wavefront_comparison,
    plot_metrics_vs_wfe,
    plot_dm_modes,
    plot_response_singular_values,
    plot_wfs_image,
)

# ── Algorithm hyperparameters ─────────────────────────────────────────────────
_RECON_PARAMS = {
    'rcond': 1e-3,
}

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
OUT_DIR  = os.path.join(os.path.dirname(__file__), 'evaluation', 'reference_outputs')
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    print("=== Shack-Hartmann Wavefront Reconstruction ===\n")

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    raw  = load_raw_data(os.path.join(DATA_DIR, 'raw_data.npz'))
    gt   = load_ground_truth(os.path.join(DATA_DIR, 'ground_truth.npz'))
    meta = json.load(open(os.path.join(DATA_DIR, 'meta_data.json')))

    response_matrix     = raw['response_matrix'].astype(np.float64)
    wfs_images          = raw['wfs_images'].astype(np.float64)
    ref_image           = raw['ref_image'].astype(np.float64)
    detector_coords_x   = raw['detector_coords_x'].astype(np.float64)
    detector_coords_y   = raw['detector_coords_y'].astype(np.float64)
    subap_map           = raw['subap_map']
    dm_modes            = raw['dm_modes'].astype(np.float64)
    aperture            = raw['aperture'].astype(np.float64)
    ground_truth_phases = gt['wavefront_phases'].astype(np.float64)

    N_px          = meta['simulation']['n_pupil_pixels']
    lam_wfs       = meta['wavefront_sensor']['wavelength_wfs_m']
    wfe_levels    = meta['wfe_levels_nm']
    n_valid_subaps = meta['wavefront_sensor']['n_valid_subaps']
    det_shape     = tuple(meta['wavefront_sensor']['det_image_shape'])
    pupil_shape   = (N_px, N_px)

    print(f"Telescope : D = {meta['telescope']['diameter_m']} m")
    print(f"WFS       : {n_valid_subaps} subaps  "
          f"{meta['wavefront_sensor']['n_slopes']} slopes")
    print(f"Detector  : {det_shape[0]}×{det_shape[1]} = {meta['wavefront_sensor']['n_det_pixels']} px")
    print(f"DM        : {meta['deformable_mirror']['n_modes']} modes")
    print(f"WFE levels: {wfe_levels} nm\n")

    # ------------------------------------------------------------------
    # 2. Reconstruct wavefront at each WFE level (with timing)
    # ------------------------------------------------------------------
    print(f"Reconstructing wavefront (Tikhonov rcond={_RECON_PARAMS['rcond']}) ...")
    result = reconstruct_all_levels(
        wfs_images, ref_image,
        detector_coords_x, detector_coords_y, subap_map,
        response_matrix, dm_modes, aperture, lam_wfs,
        n_valid_subaps   = n_valid_subaps,
        rcond            = _RECON_PARAMS['rcond'],
        ground_truth_phases = ground_truth_phases,
    )

    recon_phases  = result['reconstructed_phases']
    ncc_arr       = result['ncc_per_level']
    nrmse_arr     = result['nrmse_per_level']
    recon_time_s  = result['reconstruction_time_s']

    print(f"\n{'WFE (nm)':>10} {'NCC':>10} {'NRMSE':>10}")
    print("-" * 34)
    for i, wfe in enumerate(wfe_levels):
        print(f"{wfe:>10}  {ncc_arr[i]:>9.4f}  {nrmse_arr[i]:>9.4f}")
    print(f"\nReconstruction time (all {len(wfe_levels)} levels): {recon_time_s*1e3:.1f} ms")

    # ------------------------------------------------------------------
    # 3. Visualisations
    # ------------------------------------------------------------------
    plot_wfs_image(
        raw['ref_image'].astype(np.float64), det_shape,
        title='Reference SH-WFS image (flat wavefront)',
        output_path=os.path.join(OUT_DIR, 'wfs_ref_image.png'),
    )
    plot_wavefront_comparison(
        ground_truth_phases, recon_phases, aperture,
        pupil_shape, wfe_levels, ncc_arr, nrmse_arr,
        output_path=os.path.join(OUT_DIR, 'wavefront_comparison.png'),
    )
    plot_metrics_vs_wfe(
        wfe_levels, ncc_arr, nrmse_arr,
        output_path=os.path.join(OUT_DIR, 'metrics_vs_wfe.png'),
    )
    plot_dm_modes(dm_modes, aperture, pupil_shape, n_show=6,
                  output_path=os.path.join(OUT_DIR, 'dm_modes.png'))
    plot_response_singular_values(
        response_matrix, rcond=_RECON_PARAMS['rcond'],
        output_path=os.path.join(OUT_DIR, 'response_singular_values.png'),
    )
    import matplotlib.pyplot as _plt; _plt.close('all')
    print("\nSaved figures to evaluation/reference_outputs/")

    # ------------------------------------------------------------------
    # 4. Save reconstructions
    # ------------------------------------------------------------------
    np.savez(
        os.path.join(OUT_DIR, 'reconstruction.npz'),
        reconstructed_phases    = recon_phases[np.newaxis],
        ground_truth_phases     = ground_truth_phases[np.newaxis].astype(np.float32),
        ncc_per_level           = ncc_arr.astype(np.float32),
        nrmse_per_level         = nrmse_arr.astype(np.float32),
        reconstruction_time_s   = np.float32(recon_time_s),
    )
    print("Saved reconstruction.npz")

    print(f"\n=== Done ===")
    for i, wfe in enumerate(wfe_levels):
        print(f"  WFE={wfe:4d} nm:  NCC={ncc_arr[i]:.4f}  NRMSE={nrmse_arr[i]:.4f}")
    print(f"  Reconstruction time: {recon_time_s*1e3:.1f} ms")

    return ncc_arr, nrmse_arr, recon_time_s


if __name__ == '__main__':
    main()
