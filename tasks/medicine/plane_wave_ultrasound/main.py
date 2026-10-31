"""main.py — Plane wave ultrasound reconstruction pipeline.

Runs Stolt f-k migration with coherent compounding on both phantoms
and saves reference outputs to evaluation/reference_outputs/.
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from src.preprocessing import load_dataset
from src.solvers import coherent_compound
from src.visualization import envelope_bmode, plot_bmode, measure_psf_fwhm, measure_cnr

DATA_DIR   = os.path.join(os.path.dirname(__file__), 'data')
OUT_DIR    = os.path.join(os.path.dirname(__file__), 'evaluation', 'reference_outputs')
NPZ_PATH   = os.path.join(DATA_DIR, 'raw_data.npz')
META_PATH  = os.path.join(DATA_DIR, 'meta_data.json')

os.makedirs(OUT_DIR, exist_ok=True)


def run_dataset(dataset: str, gamma: float = 0.5):
    print(f'\n=== Processing {dataset} phantom ===')
    RF, params = load_dataset(NPZ_PATH, META_PATH, dataset)
    fs          = params['fs']
    pitch       = params['pitch']
    c           = params['c']
    t0          = params['t0']
    angles      = params['TXangle_rad']

    print(f'  RF shape: {RF.shape}  |  {len(angles)} angles  |  fs={fs/1e6:.0f} MHz')
    x, z, compound = coherent_compound(RF, fs, pitch, angles, c=c, t0=t0)
    print(f'  Compound shape: {compound.shape}')

    bmode = envelope_bmode(compound, gamma=gamma)
    return x, z, compound, bmode, params


def main():
    # ---- Fibers phantom (gamma=0.7 per reference examples) -----------------
    x1, z1, compound1, bmode1, _ = run_dataset('fibers', gamma=0.7)

    # ---- Cysts phantom (gamma=0.5 per reference examples) ------------------
    x2, z2, compound2, bmode2, _ = run_dataset('cysts', gamma=0.5)

    # ---- Save compound signals and b-mode images ----------------------------
    np.save(os.path.join(OUT_DIR, 'compound_fibers.npy'), compound1)
    np.save(os.path.join(OUT_DIR, 'compound_cysts.npy'),  compound2)
    np.save(os.path.join(OUT_DIR, 'bmode_fibers.npy'),    bmode1)
    np.save(os.path.join(OUT_DIR, 'bmode_cysts.npy'),     bmode2)
    np.save(os.path.join(OUT_DIR, 'x_fibers.npy'), x1)
    np.save(os.path.join(OUT_DIR, 'z_fibers.npy'), z1)
    np.save(os.path.join(OUT_DIR, 'x_cysts.npy'),  x2)
    np.save(os.path.join(OUT_DIR, 'z_cysts.npy'),  z2)
    print('\nSaved compound signals and b-mode images.')

    # ---- Metrics: PSF FWHM for fibers --------------------------------------
    # Wire targets are approximately every 10 mm from 10 mm to 80 mm depth
    z_targets = [0.01 * k for k in range(1, 9)]   # 0.01 m to 0.08 m
    # Keep only depths within the image
    z_targets = [zt for zt in z_targets if z1.min() <= zt <= z1.max()]
    fwhms = measure_psf_fwhm(bmode1, x1, z1, z_targets)
    print('\nLateral FWHM at wire targets (mm):')
    for zt, fw in zip(z_targets, fwhms):
        print(f'  z = {zt*1e3:.0f} mm  ->  FWHM = {fw:.3f} mm')

    # ---- Metrics: CNR for cysts --------------------------------------------
    # Two cysts visible in the image: left (anechoic) and right (hyperechoic)
    # z coordinates are relative to acquisition start (t0=5e-5 s already handled)
    cyst_centers = [
        (-0.010,  0.020),   # left cyst  (~anechoic)
        ( 0.007,  0.023),   # right cyst (~hyperechoic)
    ]
    # Use linear envelope for CNR (more physically meaningful than power-compressed)
    from scipy.signal import hilbert as _hilbert
    envelope2 = np.abs(_hilbert(np.real(compound2), axis=0))
    cnrs = measure_cnr(envelope2, x2, z2, cyst_centers,
                       cyst_radius=2e-3, shell_inner=3e-3, shell_outer=5e-3)
    print('\nCNR at circular cysts:')
    for i, (c_, (xc, zc)) in enumerate(zip(cnrs, cyst_centers)):
        print(f'  Cyst {i+1}  (x={xc*1e3:.1f}mm, z={zc*1e3:.1f}mm)  ->  CNR = {c_:.3f}')

    # ---- Save metrics.json to evaluation/ ------------------------------------------
    eval_dir = os.path.join(os.path.dirname(__file__), 'evaluation')

    # NCC / NRMSE vs baseline reference
    ref = np.load(os.path.join(DATA_DIR, 'baseline_reference.npz'))
    ref_bf = ref['bmode_fibers'][0].astype(np.float64)
    ref_bc = ref['bmode_cysts'][0].astype(np.float64)

    def _ncc(a, b):
        a, b = a.ravel(), b.ravel()
        a, b = a - a.mean(), b - b.mean()
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    def _nrmse(a, b):
        return float(np.sqrt(np.mean((a.ravel() - b.ravel())**2)) /
                     (b.max() - b.min() + 1e-12))

    ncc_f  = _ncc(bmode1.astype(np.float64), ref_bf)
    nrmse_f = _nrmse(bmode1.astype(np.float64), ref_bf)
    ncc_c  = _ncc(bmode2.astype(np.float64), ref_bc)
    nrmse_c = _nrmse(bmode2.astype(np.float64), ref_bc)
    print(f'\nNCC  fibers={ncc_f:.4f}  cysts={ncc_c:.4f}')
    print(f'NRMSE fibers={nrmse_f:.4f}  cysts={nrmse_c:.4f}')

    metrics = {
        'baseline': [{
            'method': 'Stolt f-k coherent compounding (7 angles)',
            'ncc_fibers_vs_ref': round(ncc_f, 4),
            'nrmse_fibers_vs_ref': round(nrmse_f, 4),
            'ncc_cysts_vs_ref': round(ncc_c, 4),
            'nrmse_cysts_vs_ref': round(nrmse_c, 4),
            'psf_fwhm_mm': {f'z_{int(zt*1e3)}mm': fw for zt, fw in zip(z_targets, fwhms)},
            'cnr': {f'cyst_{i+1}': c_ for i, c_ in enumerate(cnrs)},
        }],
        'ncc_fibers_boundary': 0.9,
        'nrmse_fibers_boundary': 0.1,
        'ncc_cysts_boundary': 0.9,
        'nrmse_cysts_boundary': 0.1,
        'psf_fwhm_mm_mean_boundary': round(
            sum(fwhms) / len(fwhms) * 1.1, 3),
        'cnr_mean_boundary': round(sum(cnrs) / len(cnrs) * 0.9, 3),
    }
    with open(os.path.join(eval_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f'\nSaved metrics.json')

    # ---- Save diagnostic figures -------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    plot_bmode(bmode1, x1, z1,
               title='F-K Migrated Point Targets\n7 Angles Compounded (-1.5° : 0.5° : 1.5°)',
               ax=axes[0])
    plot_bmode(bmode2, x2, z2,
               title='F-K Migrated Circular Cysts\n7 Angles Compounded (-1.5° : 0.5° : 1.5°)',
               ax=axes[1])
    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, 'bmode_comparison.png')
    plt.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f'Saved comparison figure: {fig_path}')

    # ---- Before / after for single angle (fibers) --------------------------
    from src.solvers import fkmig
    from src.preprocessing import load_dataset as ld
    RF, params = ld(NPZ_PATH, META_PATH, 'fibers')
    idx = 3  # centre angle (0°)
    x1b, z1b, migRF_single = fkmig(RF[:, :, idx], params['fs'], params['pitch'],
                                    TXangle=params['TXangle_rad'][idx], c=params['c'])
    from src.visualization import envelope_bmode as eb
    im_before = eb(RF[:, :, idx].astype(complex), gamma=0.7)
    im_after  = eb(migRF_single, gamma=0.7)

    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 6))
    plot_bmode(im_before, x1b, z1b, title='RF Data Before Migration', ax=axes2[0])
    plot_bmode(im_after,  x1b, z1b, title='F-K Migrated (single 0° angle)',  ax=axes2[1])
    plt.tight_layout()
    fig2_path = os.path.join(OUT_DIR, 'before_after_migration.png')
    plt.savefig(fig2_path, dpi=150)
    plt.close(fig2)
    print(f'Saved before/after figure: {fig2_path}')

    print('\nDone.')


if __name__ == '__main__':
    main()
