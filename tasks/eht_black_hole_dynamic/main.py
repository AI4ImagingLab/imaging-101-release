"""
EHT Black Hole Dynamic Reconstruction: StarWarps
==================================================

Reproduces key results from Bouman et al. (2017), arXiv:1711.01357.
Demonstrates that temporal coupling (StarWarps) outperforms static
per-frame reconstruction for time-varying sources.

Usage:
    cd tasks/eht_black_hole_dynamic
    python main.py
"""

import os
import json
import numpy as np

import matplotlib
matplotlib.use('Agg')

TASK_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    print('=' * 60)
    print('EHT Black Hole: Dynamic Imaging with StarWarps')
    print('Bouman et al. (2017), arXiv:1711.01357')
    print('=' * 60)

    from src.preprocessing import load_observation, load_metadata, \
        load_ground_truth, build_per_frame_models    from src.physics_model import gauss_image_covariance
    from src.solvers import StaticPerFrameSolver, StarWarpsSolver
    from src.visualization import compute_video_metrics, print_metrics_table, \
        plot_video_comparison, plot_metrics_over_time, plot_angle_time

    # ── 1. Load data ──────────────────────────────────────────────────
    print('\n[1/6] Loading data ...')
    data_dir = os.path.join(TASK_DIR, 'data')
    obs = load_observation(data_dir)
    meta = load_metadata(data_dir)
    gt = load_ground_truth(data_dir)

    N = meta['N']
    psize = meta['pixel_size_rad']
    n_frames = meta['n_frames']
    total_flux = meta['total_flux']

    print(f'  Image: {N}x{N},  Frames: {n_frames}')
    print(f'  Baselines/frame: {meta["baselines_per_frame"]}')
    print(f'  Total flux: {total_flux} Jy')

    # ── 2. Build per-frame forward models ─────────────────────────────
    print('\n[2/6] Building per-frame forward models ...')
    models = build_per_frame_models(obs, meta)
    print(f'  {len(models)} DFT models built')

    # ── 3. Set up prior ───────────────────────────────────────────────
    print('\n[3/6] Setting up prior ...')
    npixels = N * N

    # Prior mean: super-Gaussian (flat-top) with floor
    # exp(-(r/σ)^6) is flat in the centre and rolls off smoothly at edges.
    # A 5% floor prevents near-zero edge values that would make the
    # covariance singular (Λ ∝ diag(imvec) · BaseCov · diag(imvec)).
    fwhm_uas = 50.0
    n_pow = 6
    uas2rad = np.pi / (180.0 * 3600.0 * 1e6)
    sigma_sg = (fwhm_uas * uas2rad) / (2.0 * (np.log(2.0))**(1.0 / n_pow))

    xlist = np.arange(0, -N, -1) * psize + (psize * N) / 2.0 - psize / 2.0
    ylist = np.arange(0, -N, -1) * psize + (psize * N) / 2.0 - psize / 2.0
    xx, yy = np.meshgrid(xlist, ylist)
    rr = np.sqrt(xx**2 + yy**2)
    sg_blob = np.exp(-(rr / sigma_sg)**n_pow)
    sg_blob = np.maximum(sg_blob, 0.05 * sg_blob.max())  # 5% floor
    prior_mean = (sg_blob / sg_blob.sum() * total_flux).ravel()

    # Prior covariance: power-law Fourier spectrum (StarWarps default)
    prior_cov = gauss_image_covariance(
        N, psize, prior_mean, power_dropoff=2.0, frac=0.5)

    # Process noise: small identity-like covariance allowing intensity change
    variance_img_diff = 1e-7
    process_noise_cov = variance_img_diff * np.eye(npixels)

    print(f'  Prior mean flux: {prior_mean.sum():.2f} Jy')
    print(f'  Prior: super-Gaussian n={n_pow}, FWHM={fwhm_uas:.0f} uas, 5% floor')
    print(f'  Process noise variance: {variance_img_diff:.1e}')

    # ── 4. Reconstruct ────────────────────────────────────────────────
    print('\n[4/6] Reconstructing ...')
    results = {}

    # Static per-frame baseline
    print('  [1/2] Static per-frame (Gaussian MAP) ...')
    static_solver = StaticPerFrameSolver(prior_mean, prior_cov)
    static_frames = static_solver.reconstruct(
        models, obs, N, measurement='vis', num_lin_iters=1)
    results['Static per-frame'] = static_frames

    # StarWarps (match ehtim example parameters)
    print('  [2/2] StarWarps (EM, 30 iterations) ...')
    starwarps_solver = StarWarpsSolver(
        prior_mean=prior_mean,
        prior_cov=prior_cov,
        process_noise_cov=process_noise_cov,
        N=N, psize=psize,
        warp_method='phase',
        measurement='vis',
        n_em_iters=30,
        num_lin_iters=5,
        interior_priors=True,
        motion_basis='affine_no_translation',
        m_step_maxiter=4000,
    )
    sw_result = starwarps_solver.reconstruct(models, obs)
    results['StarWarps'] = sw_result['frames']

    # ── 5. Metrics ────────────────────────────────────────────────────
    print('\n[5/6] Computing metrics ...')
    metrics = {}
    for name, frames in results.items():
        metrics[name] = compute_video_metrics(frames, gt)
    print_metrics_table(metrics)

    # ── 6. Save ───────────────────────────────────────────────────────
    print('[6/6] Saving outputs ...')
    out_dir = os.path.join(TASK_DIR, 'evaluation', 'reference_outputs')
    os.makedirs(out_dir, exist_ok=True)

    # Save reconstructions
    np.save(os.path.join(out_dir, 'static_reconstruction.npy'),
            np.array(results['Static per-frame']))
    np.save(os.path.join(out_dir, 'starwarps_reconstruction.npy'),
            np.array(results['StarWarps']))

    # Save metrics
    # Convert metrics to serializable format
    metrics_save = {}
    for name, m in metrics.items():
        metrics_save[name] = {
            'average': m['average'],
            'per_frame': m['per_frame'],
        }
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics_save, f, indent=2)

    # Save plots
    frame_times = obs['frame_times']
    plot_video_comparison(
        gt, results, frame_times,
        save_path=os.path.join(out_dir, 'video_comparison.png'))
    plot_metrics_over_time(
        metrics, frame_times,
        save_path=os.path.join(out_dir, 'metrics_over_time.png'))
    plot_angle_time(
        gt, results, frame_times, N, psize,
        save_path=os.path.join(out_dir, 'angle_time.png'))

    print(f'  Saved to {out_dir}/')
    print('\nDone!')


if __name__ == '__main__':
    main()
