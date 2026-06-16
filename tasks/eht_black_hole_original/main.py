"""
EHT Black Hole Static Reconstruction: Closure-Only Imaging
============================================================

Reproduces key results from Chael et al. (2018), ApJ 857:23.
Demonstrates that closure-only imaging is robust to station-based
gain errors while traditional visibility imaging fails.

Usage:
    cd tasks/eht_black_hole_original
    python main.py
"""

import os
import json
import numpy as np

import matplotlib
matplotlib.use('Agg')

TASK_DIR = os.path.dirname(os.path.abspath(__file__))


def _make_gaussian_prior(N, pixel_size_rad, fwhm_uas, total_flux):
    """Create a Gaussian prior image (matches ehtim's add_gauss)."""
    uas_to_rad = np.pi / (180 * 3600 * 1e6)
    fwhm_rad = fwhm_uas * uas_to_rad
    sigma = fwhm_rad / (2 * np.sqrt(2 * np.log(2)))

    coords_rad = (np.arange(N) - N / 2) * pixel_size_rad
    xx, yy = np.meshgrid(coords_rad, coords_rad)
    prior = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    prior *= total_flux / prior.sum()
    return prior


def _get_closure_uv_data(obs, triangles, quadrangles):
    """Map closure topology to per-leg UV coordinates."""
    uv = obs['uv_coords']
    sids = obs['station_ids']

    # Build baseline lookup: (s1, s2) → row index
    bl_lookup = {}
    for idx in range(len(sids)):
        s1, s2 = int(sids[idx, 0]), int(sids[idx, 1])
        bl_lookup[(s1, s2)] = idx
        bl_lookup[(s2, s1)] = idx

    # Closure phase UV (V_ij * V_jk * V_ki)
    n_tri = len(triangles)
    cp_u1 = np.zeros((n_tri, 2))
    cp_u2 = np.zeros((n_tri, 2))
    cp_u3 = np.zeros((n_tri, 2))

    for t in range(n_tri):
        i, j, k = triangles[t]
        idx_ij = bl_lookup[(i, j)]
        idx_jk = bl_lookup[(j, k)]
        idx_ki = bl_lookup[(k, i)]

        # If stored as (j,i), negate UV for conjugate
        cp_u1[t] = uv[idx_ij] if sids[idx_ij, 0] == i else -uv[idx_ij]
        cp_u2[t] = uv[idx_jk] if sids[idx_jk, 0] == j else -uv[idx_jk]
        cp_u3[t] = uv[idx_ki] if sids[idx_ki, 0] == k else -uv[idx_ki]

    # Log closure amplitude UV (log|V_ij| + log|V_kl| - log|V_ik| - log|V_jl|)
    n_quad = len(quadrangles)
    lca_u1 = np.zeros((n_quad, 2))
    lca_u2 = np.zeros((n_quad, 2))
    lca_u3 = np.zeros((n_quad, 2))
    lca_u4 = np.zeros((n_quad, 2))

    for q in range(n_quad):
        i, j, k, l = quadrangles[q]
        lca_u1[q] = uv[bl_lookup[(i, j)]]
        lca_u2[q] = uv[bl_lookup[(k, l)]]
        lca_u3[q] = uv[bl_lookup[(i, k)]]
        lca_u4[q] = uv[bl_lookup[(j, l)]]

    return {
        'cp_u1': cp_u1, 'cp_u2': cp_u2, 'cp_u3': cp_u3,
        'lca_u1': lca_u1, 'lca_u2': lca_u2, 'lca_u3': lca_u3, 'lca_u4': lca_u4,
    }


def main():
    print('=' * 60)
    print('EHT Black Hole: Closure-Only Imaging')
    print('Chael et al. (2018), ApJ 857:23')
    print('=' * 60)

    from src.preprocessing import load_observation, load_metadata
    from src.physics_model import ClosureForwardModel
    from src.solvers import ClosureRMLSolver
    from src.visualization import compute_metrics, print_metrics_table

    # ── 1. Load data ────────────────────────────────────────────────────
    print('\n[1/6] Loading data ...')
    data_dir = os.path.join(TASK_DIR, 'data')
    obs = load_observation(data_dir)
    meta = load_metadata(data_dir)

    N = meta['N']
    psize = meta['pixel_size_rad']
    total_flux = meta['total_flux']

    # Per-scan closure data from ehtim (stored in raw_data.npz)
    n_cp = len(obs['cp_values_deg'])
    n_lca = len(obs['lca_values'])

    print(f'  Image: {N}x{N},  Baselines: {len(obs["uv_coords"])}')
    print(f'  Closure phases: {n_cp},  Log closure amps: {n_lca}')

    # ── 2. Build model ──────────────────────────────────────────────────
    print('\n[2/6] Building forward model ...')
    model = ClosureForwardModel(
        uv_coords=obs['uv_coords'], N=N, pixel_size_rad=psize,
        triangles=np.zeros((0, 3), dtype=int),
        quadrangles=np.zeros((0, 4), dtype=int),
    )
    print(f'  {model}')

    # ── 3. Create prior ─────────────────────────────────────────────────
    print('\n[3/6] Creating Gaussian prior ...')
    prior = _make_gaussian_prior(N, psize, fwhm_uas=20.0, total_flux=total_flux)

    gt = np.load(os.path.join(TASK_DIR, 'data', 'ground_truth.npz'))['image']

    # ── 4. Reconstruct ──────────────────────────────────────────────────
    print('\n[4/6] Reconstructing ...')
    reg = {'gs': 1, 'simple': 10}

    # Observation data for calibrated visibilities (using ehtim-derived closure data)
    obs_data_cal = {
        'vis_obs': obs['vis_cal'],
        'sigma_vis': obs['sigma_vis'],
        'cp_values_deg': obs['cp_values_deg'],
        'cp_sigmas_deg': obs['cp_sigmas_deg'],
        'cp_u1': obs['cp_u1'], 'cp_u2': obs['cp_u2'], 'cp_u3': obs['cp_u3'],
        'lca_values': obs['lca_values'],
        'lca_sigmas': obs['lca_sigmas'],
        'lca_u1': obs['lca_u1'], 'lca_u2': obs['lca_u2'],
        'lca_u3': obs['lca_u3'], 'lca_u4': obs['lca_u4'],
    }

    # Observation data for corrupted visibilities
    obs_data_corrupt = {
        'vis_obs': obs['vis_corrupt'],
        'sigma_vis': obs['sigma_vis'],
        'cp_values_deg': obs['cp_corrupt_values_deg'],
        'cp_sigmas_deg': obs['cp_corrupt_sigmas_deg'],
        'cp_u1': obs['cp_u1'], 'cp_u2': obs['cp_u2'], 'cp_u3': obs['cp_u3'],
        'lca_values': obs['lca_corrupt_values'],
        'lca_sigmas': obs['lca_corrupt_sigmas'],
        'lca_u1': obs['lca_u1'], 'lca_u2': obs['lca_u2'],
        'lca_u3': obs['lca_u3'], 'lca_u4': obs['lca_u4'],
    }

    results = {}

    print('  [1/4] Vis RML (calibrated) ...')
    solver = ClosureRMLSolver(data_terms={'vis': 100}, reg_terms=reg, prior=prior)
    results['Vis RML (cal)'] = solver.reconstruct(model, obs_data_cal, x0=prior)

    print('  [2/4] Vis RML (corrupted) ...')
    solver = ClosureRMLSolver(data_terms={'vis': 100}, reg_terms=reg, prior=prior)
    results['Vis RML (corrupt)'] = solver.reconstruct(model, obs_data_corrupt, x0=prior)

    print('  [3/4] Closure-only (calibrated) ...')
    solver = ClosureRMLSolver(
        data_terms={'cphase': 50, 'logcamp': 50}, reg_terms=reg, prior=prior)
    results['Closure-only (cal)'] = solver.reconstruct(model, obs_data_cal, x0=prior)

    print('  [4/4] Closure-only (corrupted) ...')
    solver = ClosureRMLSolver(
        data_terms={'cphase': 50, 'logcamp': 50}, reg_terms=reg, prior=prior)
    results['Closure-only (corrupt)'] = solver.reconstruct(model, obs_data_corrupt, x0=prior)

    # ── 5. Metrics ──────────────────────────────────────────────────────
    print('\n[5/6] Computing metrics ...')
    metrics = {name: compute_metrics(img, gt) for name, img in results.items()}
    print_metrics_table(metrics)

    # ── 6. Save ─────────────────────────────────────────────────────────
    print('\n[6/6] Saving outputs ...')
    out_dir = os.path.join(TASK_DIR, 'evaluation', 'reference_outputs')
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, 'reconstruction.npy'), results['Closure-only (corrupt)'])

    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f'  Saved to {out_dir}/')
    print('\nDone!')


if __name__ == '__main__':
    main()
