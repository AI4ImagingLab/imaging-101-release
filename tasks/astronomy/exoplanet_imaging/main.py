"""KLIP-ADI exoplanet imaging pipeline.

Runs KLIP PSF subtraction on the Beta Pictoris VLT/NACO L'-band dataset,
saves the final detection map, and prints the SNR of Beta Pic b.

Usage
-----
    python main.py

Outputs
-------
evaluation/reference_outputs/klip_result.npz   Detection map for K_klip in K_KLIP_VALUES
evaluation/reference_outputs/klip_result_K10.png
evaluation/reference_outputs/klip_result_K20.png
evaluation/reference_outputs/raw_frame.png
"""

import json
import os
import matplotlib
matplotlib.use('Agg')

import numpy as np

from src.preprocessing import load_raw_data
from src.solvers import klip_adi
from src.visualization import (
    plot_raw_frame,
    plot_klip_result,
    compute_snr,
)

# -----------------------------------------------------------------
# Algorithm parameters
# -----------------------------------------------------------------
K_KLIP_VALUES = [10, 20]      # KL truncation levels
KLIP_METHOD   = 'svd'         # 'svd' | 'pca' | 'eigh'
COMBINE_STAT  = 'mean'        # 'mean' | 'median'
DEVICE        = 'cpu'         # 'cuda' if GPU available

# -----------------------------------------------------------------
# Paths
# -----------------------------------------------------------------
DATA_DIR  = os.path.join(os.path.dirname(__file__), 'data')
EVAL_DIR  = os.path.join(os.path.dirname(__file__), 'evaluation', 'reference_outputs')
os.makedirs(EVAL_DIR, exist_ok=True)


def main():
    # ---------------------------------------------------------------
    # 1. Load data
    # ---------------------------------------------------------------
    cube, angles, psf = load_raw_data(os.path.join(DATA_DIR, 'raw_data.npz'))
    meta = json.load(open(os.path.join(DATA_DIR, 'meta_data.json')))

    N, H, W = cube.shape
    center = tuple(meta['center_px'])           # (cx, cy) = (50, 50)
    iwa    = float(meta['iwa_px'])              # 4 px
    fwhm   = float(meta['fwhm_px'])             # ~4.8 px
    px_sc  = float(meta['pixel_scale_arcsec'])  # 0.02719 arcsec/px
    comp   = meta['known_companion']
    scalebar_px  = 0.25 / px_sc                 # 0.25" = ~9.2 px
    scalebar_lbl = '0.25"'

    print(f"Loaded cube: {cube.shape}, angles: [{angles.min():.1f}, {angles.max():.1f}] deg")

    # ---------------------------------------------------------------
    # 2. Plot raw frame for reference
    # ---------------------------------------------------------------
    plot_raw_frame(
        cube[0], center=center, iwa=iwa,
        vmin=1e2, vmax=1e4, log_scale=True,
        scalebar_length=scalebar_px, scalebar_label=scalebar_lbl,
        title='Beta Pic: raw ADI frame (k=0)',
        output_path=os.path.join(EVAL_DIR, 'raw_frame.png'),
    )
    print("Saved raw_frame.png")

    # ---------------------------------------------------------------
    # 3. Run KLIP-ADI for all K values
    # ---------------------------------------------------------------
    results = klip_adi(
        cube, angles, K_klip=K_KLIP_VALUES,
        iwa=iwa, center=center,
        method=KLIP_METHOD, statistic=COMBINE_STAT,
        device=DEVICE,
    )
    # results: (n_K, H, W)
    print(f"KLIP done. Output shape: {results.shape}")

    # ---------------------------------------------------------------
    # 4. Companion pixel position for Beta Pic b in the derotated image
    # ---------------------------------------------------------------
    # These coordinates are determined empirically from the brightest pixel
    # in the KLIP output and confirmed against the known orbital separation
    # (0.44 arcsec = 16–17 px at this epoch).
    planet_x = float(comp['planet_x_klip'])
    planet_y = float(comp['planet_y_klip'])

    # ---------------------------------------------------------------
    # 5. Plot and compute SNR for each K
    # ---------------------------------------------------------------
    snr_dict = {}
    for i, k in enumerate(K_KLIP_VALUES):
        img = results[i]
        snr = compute_snr(img, planet_x, planet_y, fwhm, exclude_nearest=1)
        snr_dict[k] = snr
        print(f"  K_klip={k:3d}: Beta Pic b SNR = {snr:.1f}")

        plot_klip_result(
            img, center=center, iwa=iwa,
            vmin=-6, vmax=16,
            scalebar_length=scalebar_px, scalebar_label=scalebar_lbl,
            planet_xy=(planet_x, planet_y),
            title=f'Beta Pic: KLIP result (K={k})',
            output_path=os.path.join(EVAL_DIR, f'klip_result_K{k}.png'),
            xlim_half=40, ylim_half=40,
        )
        print(f"  Saved klip_result_K{k}.png")

    # ---------------------------------------------------------------
    # 6. Save reference output for evaluation
    # ---------------------------------------------------------------
    np.savez(
        os.path.join(EVAL_DIR, 'klip_result.npz'),
        **{f'K{k}': results[i][np.newaxis] for i, k in enumerate(K_KLIP_VALUES)},
    )
    print("Saved klip_result.npz")

    print("\nDone.")
    return results, snr_dict


if __name__ == '__main__':
    main()
