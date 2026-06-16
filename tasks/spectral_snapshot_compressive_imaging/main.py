"""PnP-CASSI: Plug-and-Play priors for spectral snapshot compressive imaging.

Pipeline entry point: loads data, runs GAP reconstruction, saves results.
"""

import os
import time
import json
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')

from src.preprocessing import load_meta_data, load_data, build_mask_3d
from src.physics_model import shift_back
from src.solvers import gap_denoise
from src.visualization import (psnr, calculate_ssim, plot_spectral_bands,
                                plot_comparison, plot_measurement)


def main():
    # --- Paths ---
    data_dir = './data'
    eval_dir = './evaluation'
    output_dir = os.path.join(eval_dir, 'reference_outputs')
    checkpoint_path = os.path.join(output_dir, 'deep_denoiser.pth')
    os.makedirs(output_dir, exist_ok=True)

    # --- Load config ---
    meta = load_meta_data(os.path.join(data_dir, 'meta_data.json'))
    r, c, nC, step = meta['r'], meta['c'], meta['nC'], meta['step']
    datname = 'kaist_crop256_01'

    # --- Load data from standard npz interface ---
    meas, mask2d, truth = load_data(
        os.path.join(data_dir, 'raw_data.npz'),
        os.path.join(data_dir, 'ground_truth.npz'),
    )
    mask_3d = build_mask_3d(mask2d, r, c, nC, step)

    # --- Save measurement visualization ---
    plot_measurement(meas, title='{} Measurement'.format(datname),
                     save_path=os.path.join(output_dir, '{}_meas.png'.format(datname)))

    # --- Reconstruct ---
    _GAP_CFG = {
        "lambda": 1,
        "accelerate": True,
        "iter_max": 20,
        "tv_iter_max": 5,
        "sigma": [130, 130, 130, 130, 130, 130, 130, 130],
    }
    solver_cfg = _GAP_CFG
    Phi = mask_3d

    print('Running GAP-HSICNN reconstruction...')
    begin_time = time.time()
    vgap, psnr_all, ssim_all = gap_denoise(
        meas, Phi,
        _lambda=solver_cfg['lambda'],
        accelerate=solver_cfg['accelerate'],
        iter_max=solver_cfg['iter_max'],
        sigma=solver_cfg['sigma'],
        tv_iter_max=solver_cfg['tv_iter_max'],
        X_orig=truth,
        checkpoint_path=checkpoint_path,
    )
    end_time = time.time()
    elapsed = end_time - begin_time

    vrecon = shift_back(vgap, step=1)
    final_psnr = psnr(truth, vrecon)
    final_ssim = calculate_ssim(truth, vrecon)

    print('GAP-HSICNN: PSNR {:.2f} dB, SSIM {:.4f}, time {:.1f}s'.format(
        final_psnr, final_ssim, elapsed))

    # --- Save results ---
    sio.savemat(os.path.join(output_dir, '{}_result.mat'.format(datname)),
                {'img': vrecon})

    # Save metrics
    metrics = {
        'dataset': datname,
        'method': 'GAP-HSICNN',
        'resolution': '{}x{}'.format(r, c),
        'spectral_bands': nC,
        'psnr_db': round(final_psnr, 2),
        'ssim': round(final_ssim, 4),
        'runtime_seconds': round(elapsed, 1),
        'num_iterations': len(psnr_all),
    }
    with open(os.path.join(eval_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

    # --- Visualizations ---
    wl_start = meta['wavelength_start_nm']
    wl_step = meta['wavelength_step_nm']
    vmin, vmax = truth.min(), truth.max()

    plot_spectral_bands(truth, 'Ground Truth - {} Spectral Bands ({}nm - {}nm)'.format(
        nC, wl_start, meta['wavelength_end_nm']),
        wavelength_start=wl_start, wavelength_step=wl_step,
        vmin=vmin, vmax=vmax,
        save_path=os.path.join(output_dir, '{}_groundtruth_all31.png'.format(datname)))

    plot_spectral_bands(vrecon, 'Reconstruction (GAP-HSICNN) - {} Spectral Bands'.format(nC),
        wavelength_start=wl_start, wavelength_step=wl_step,
        vmin=vmin, vmax=vmax,
        save_path=os.path.join(output_dir, '{}_recon_all31.png'.format(datname)))

    plot_comparison(truth, vrecon, final_psnr, final_ssim,
        wavelength_start=wl_start, wavelength_step=wl_step,
        save_path=os.path.join(output_dir, '{}_comparison_all31.png'.format(datname)))

    print('Results saved to {}'.format(output_dir))


if __name__ == '__main__':
    main()
