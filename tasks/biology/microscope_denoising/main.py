"""
Zero-Shot Fluorescence Microscopy Restoration — pipeline entry point.

Pipeline (ZS-DeconvNet, Qiao et al., Nat. Commun. 2024):
  1. Load noisy TIRF Microtubule images and PSF.
  2. Estimate noise parameters (β₁, β₂) and generate recorrupted training pairs.
  3. Jointly train Stage 1 (denoiser) + Stage 2 (deconvolver) U-Nets (30 000 iters).
  4. Run two-stage inference on test frame (frame 01).
  5. Evaluate: Stage 1 — background noise reduction, SNR improvement;
               Stage 2 — PSF residual consistency, sharpness improvement.
  6. Save reference outputs to evaluation/reference_outputs/.

Usage:
  cd tasks/microscope_denoising
  pip install -r requirements.txt
  # First time only: prepare data
  python src/generate_data.py
  # Run full pipeline
  python main.py
"""

import matplotlib
matplotlib.use('Agg')

import os
import sys
import json
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from src.physics_model import estimate_noise_params, psf_convolve
from src.preprocessing import extract_patches, prctile_norm
from src.solvers import train_zs_deconvnet, denoise_image, deconvolve_image
from src.visualization import (
    compute_all_metrics, compute_psf_residual,
    plot_comparison, plot_zoom, plot_training_curve,
    plot_intensity_profile,
)

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(TASK_DIR, 'data')
OUT_DIR = os.path.join(TASK_DIR, 'evaluation', 'reference_outputs')
EVAL_DIR = os.path.join(TASK_DIR, 'evaluation')
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print('\n[1/6] Loading data...')
    raw = np.load(os.path.join(DATA_DIR, 'raw_data.npz'))
    measurements = raw['measurements'].astype(np.float64)   # (55, 502, 502)
    psf = raw['psf'][0].astype(np.float32)                  # (32, 32)
    with open(os.path.join(DATA_DIR, 'meta_data.json')) as f:
        meta = json.load(f)

    bg = float(meta['background_adu'])
    alpha = float(meta['noise_model']['alpha_recorruption'])
    print(f'  Loaded {measurements.shape[0]} frames, '
          f'shape {measurements.shape[1:]}')
    print(f'  PSF shape: {psf.shape}  sum={psf.sum():.6f}')

    # ── 2. Estimate noise parameters ──────────────────────────────────────────
    print('\n[2/6] Estimating noise parameters...')
    beta1, beta2 = estimate_noise_params(measurements[0], bg=bg)
    print(f'  beta1={beta1:.4f}, beta2={beta2:.4f}  (alpha={alpha})')

    # ── 3. Generate training patches ──────────────────────────────────────────
    print('\n[3/6] Generating recorrupted training patches...')
    t0 = time.time()
    y_hat, y_bar = extract_patches(
        measurements,
        patch_size=128,
        n_patches=10000,
        beta1=beta1,
        beta2=beta2,
        alpha=alpha,
        bg=bg,
        seed=42,
    )
    print(f'  Generated {len(y_hat)} pairs in {time.time()-t0:.1f}s')
    print(f'  y_hat range: [{y_hat.min():.3f}, {y_hat.max():.3f}]')

    # ── 4. Joint training: Stage 1 (denoiser) + Stage 2 (deconvolver) ─────────
    print('\n[4/6] Jointly training Stage 1 (denoiser) + Stage 2 (deconvolver)...')
    t0 = time.time()
    model_den, model_dec, loss_history = train_zs_deconvnet(
        y_hat, y_bar,
        psf=psf,
        n_iters=30000,
        batch_size=4,
        lr=5e-4,
        lr_decay_steps=10000,
        lr_decay_factor=0.5,
        mu=0.5,
        hess_weight=0.02,
        device=device,
        verbose=True,
    )
    print(f'  Training done in {(time.time()-t0)/60:.1f} min')
    total_losses = [h[0] for h in loss_history]
    print(f'  Final total loss: {total_losses[-1]:.6f}')

    torch.save(model_den.state_dict(),
               os.path.join(OUT_DIR, 'model_den_weights.pt'))
    torch.save(model_dec.state_dict(),
               os.path.join(OUT_DIR, 'model_dec_weights.pt'))
    np.save(os.path.join(OUT_DIR, 'loss_history.npy'),
            np.array(loss_history))
    print(f'  Saved model weights and loss history.')

    # ── 5. Two-stage inference on test frame ──────────────────────────────────
    print('\n[5/6] Running two-stage inference on test frame (frame 01)...')
    t0 = time.time()
    denoised, deconvolved = deconvolve_image(
        model_den, model_dec, measurements[0],
        patch_size=128,
        overlap=32,
        device=device,
    )
    print(f'  Inference done in {time.time()-t0:.1f}s')
    np.save(os.path.join(OUT_DIR, 'denoised.npy'), denoised)
    np.save(os.path.join(OUT_DIR, 'deconvolved.npy'), deconvolved)

    # ── 6. Evaluate ───────────────────────────────────────────────────────────
    print('\n[6/6] Computing metrics...')
    noisy_frame = measurements[0].astype(np.float32)

    # ── Stage 1 metrics: background noise and SNR (no reference needed) ───────
    def bg_signal_stats(img):
        """Return (bg_std, sig_mean) for darkest/brightest 32×32 block."""
        blocks = sorted(
            [(img[i:i+16, j:j+16].mean(), i, j)
             for i in range(0, img.shape[0]-16, 16)
             for j in range(0, img.shape[1]-16, 16)]
        )
        bg_i, bg_j = blocks[0][1], blocks[0][2]
        sig_i, sig_j = blocks[-1][1], blocks[-1][2]
        bg_std = float(img[bg_i:bg_i+32, bg_j:bg_j+32].std())
        sig_mean = float(img[sig_i:sig_i+32, sig_j:sig_j+32].mean())
        return bg_std, sig_mean

    bg_n_std, sig_n_mean = bg_signal_stats(noisy_frame)
    bg_d_std, sig_d_mean = bg_signal_stats(denoised)
    bg_r_std, sig_r_mean = bg_signal_stats(deconvolved)

    snr_noisy       = sig_n_mean / (bg_n_std + 1e-6)
    snr_denoised    = sig_d_mean / (bg_d_std + 1e-6)
    snr_deconvolved = sig_r_mean / (bg_r_std + 1e-6)

    print(f'  Background noise: noisy={bg_n_std:.2f}  '
          f'denoised={bg_d_std:.2f}  deconvolved={bg_r_std:.2f} ADU')
    print(f'  Noise reduction (Stage 1): {bg_n_std/bg_d_std:.1f}×')
    print(f'  SNR: noisy={snr_noisy:.1f}  denoised={snr_denoised:.1f}  '
          f'deconvolved={snr_deconvolved:.1f}')

    # ── Stage 2 metrics: PSF residual + sharpness ratio ──────────────────────
    psf_residual_noisy = compute_psf_residual(noisy_frame, noisy_frame, psf)
    psf_residual_dec   = compute_psf_residual(deconvolved, noisy_frame, psf)

    # Sharpness: ratio of high-frequency power (Laplacian variance)
    from scipy.ndimage import laplace
    lap_noisy = float(laplace(noisy_frame.astype(np.float64)).var())
    lap_denoised = float(laplace(denoised.astype(np.float64)).var())
    lap_deconvolved = float(laplace(deconvolved.astype(np.float64)).var())

    print(f'  PSF residual (Stage 2 consistency): {psf_residual_dec:.4f}')
    print(f'  Sharpness (Laplacian variance):  '
          f'noisy={lap_noisy:.1f}  denoised={lap_denoised:.1f}  '
          f'deconvolved={lap_deconvolved:.1f}')

    # NCC / NRMSE vs baseline reference outputs
    baseline_ref = np.load(os.path.join(DATA_DIR, 'baseline_reference.npz'))
    ref_denoised    = baseline_ref['denoised'][0].astype(np.float64)
    ref_deconvolved = baseline_ref['deconvolved'][0].astype(np.float64)

    def _ncc(a, b):
        a, b = a.ravel(), b.ravel()
        a, b = a - a.mean(), b - b.mean()
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    def _nrmse(a, b):
        return float(np.sqrt(np.mean((a.ravel() - b.ravel())**2)) /
                     (b.max() - b.min() + 1e-12))

    ncc_denoised    = _ncc(denoised.astype(np.float64), ref_denoised)
    nrmse_denoised  = _nrmse(denoised.astype(np.float64), ref_denoised)
    ncc_dec         = _ncc(deconvolved.astype(np.float64), ref_deconvolved)
    nrmse_dec       = _nrmse(deconvolved.astype(np.float64), ref_deconvolved)
    print(f'  NCC  denoised={ncc_denoised:.4f}  deconvolved={ncc_dec:.4f}')
    print(f'  NRMSE denoised={nrmse_denoised:.4f}  deconvolved={nrmse_dec:.4f}')

    metrics_out = {
        'baseline': [{
            'method': 'ZS-DeconvNet (joint denoising + deconvolution, 30 000 iters)',
            'ncc_vs_ref': round(ncc_denoised, 4),
            'nrmse_vs_ref': round(nrmse_denoised, 4),
            'noise_reduction_factor': round(bg_n_std / (bg_d_std + 1e-6), 2),
            'sharpness_improvement_factor': round(
                lap_deconvolved / (lap_denoised + 1e-6), 2),
        }],
        'ncc_boundary': 0.85,
        'nrmse_boundary': 0.2,
        'noise_reduction_factor_boundary': 5.0,
        'sharpness_improvement_factor_boundary': 3.0,
        'nrmse_definition': (
            'NRMSE computed between agent denoised output and reference denoised output '
            '(evaluation/reference_outputs/denoised.npy), normalised by dynamic range of '
            'the reference. Boundaries are lenient due to stochastic zero-shot training.'
        ),
        'detail': {
            'test_frame': 'measurements[0] (Microtubule frame 01)',
            'stage1_denoising': {
                'background_noise_std_noisy': round(bg_n_std, 4),
                'background_noise_std_denoised': round(bg_d_std, 4),
                'noise_reduction_factor': round(bg_n_std / (bg_d_std + 1e-6), 2),
                'snr_noisy': round(snr_noisy, 1),
                'snr_denoised': round(snr_denoised, 1),
                'snr_improvement_factor': round(snr_denoised / (snr_noisy + 1e-6), 2),
            },
            'stage2_deconvolution': {
                'psf_residual': round(psf_residual_dec, 6),
                'sharpness_noisy': round(lap_noisy, 2),
                'sharpness_denoised': round(lap_denoised, 2),
                'sharpness_deconvolved': round(lap_deconvolved, 2),
                'sharpness_improvement_factor': round(
                    lap_deconvolved / (lap_denoised + 1e-6), 2),
            },
            'training': {
                'n_iters': 30000,
                'n_patches': len(y_hat),
                'patch_size': 128,
                'batch_size': 4,
                'lr_initial': 5e-4,
                'mu': 0.5,
                'hess_weight': 0.02,
            },
        },
    }
    with open(os.path.join(EVAL_DIR, 'metrics.json'), 'w') as f:
        json.dump(metrics_out, f, indent=2)
    print(f'  Saved metrics.json')

    # ── Plots ─────────────────────────────────────────────────────────────────
    print('\nGenerating figures...')
    plot_comparison(
        noisy_frame, denoised, deconvolved,
        titles=['Noisy input (frame 01)',
                'Stage 1: ZS-DeconvNet denoised',
                'Stage 2: ZS-DeconvNet deconvolved'],
        save_path=os.path.join(OUT_DIR, 'comparison.png'),
    )
    plot_zoom(
        noisy_frame, denoised, deconvolved,
        roi=(200, 350, 200, 350),
        save_path=os.path.join(OUT_DIR, 'comparison_zoom.png'),
    )
    plot_training_curve(
        loss_history,
        save_path=os.path.join(OUT_DIR, 'training_curve.png'),
    )
    plot_intensity_profile(
        noisy_frame, denoised, deconvolved,
        row=measurements[0].shape[0] // 2,
        save_path=os.path.join(OUT_DIR, 'intensity_profile.png'),
    )
    print('Done.')


if __name__ == '__main__':
    main()
