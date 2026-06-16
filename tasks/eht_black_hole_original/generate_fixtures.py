#!/usr/bin/env python
"""
Generate per-function test fixtures from the cleaned code.
Saves fixtures in evaluation/fixtures/<module>/<function>.npz
"""

import os, sys, json
import numpy as np

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TASK_DIR)

FIX_DIR = os.path.join(TASK_DIR, 'evaluation', 'fixtures')

from src.preprocessing import (
    load_observation, load_metadata, find_triangles, find_quadrangles,
    compute_closure_phases, compute_log_closure_amplitudes,
    closure_phase_sigma, closure_amplitude_sigma, prepare_data,
)
from src.physics_model import ClosureForwardModel, _ftmatrix
from src.solvers import GullSkillingRegularizer, SimpleEntropyRegularizer, TVRegularizer
from src.visualization import compute_metrics

DATA_DIR = os.path.join(TASK_DIR, 'data')

print('Loading data ...')
obs = load_observation(DATA_DIR)
meta = load_metadata(DATA_DIR)
N = meta['N']
psize = meta['pixel_size_rad']

# ── Preprocessing fixtures ──────────────────────────────────────────────
print('Generating preprocessing fixtures ...')
fix_pre = os.path.join(FIX_DIR, 'preprocessing')
os.makedirs(fix_pre, exist_ok=True)

# load_observation
np.savez(os.path.join(fix_pre, 'load_observation.npz'),
         output_vis_cal_shape=np.array(obs['vis_cal'].shape),
         output_uv_coords_shape=np.array(obs['uv_coords'].shape),
         output_vis_cal_first=obs['vis_cal'][:5],
         output_uv_first=obs['uv_coords'][:5])

# load_metadata
with open(os.path.join(fix_pre, 'load_metadata.json'), 'w') as f:
    json.dump(meta, f, indent=2)

# find_triangles
triangles = find_triangles(obs['station_ids'], meta['n_stations'])
np.savez(os.path.join(fix_pre, 'find_triangles.npz'),
         input_station_ids=obs['station_ids'],
         input_n_stations=np.array(meta['n_stations']),
         output_triangles=triangles)

# find_quadrangles
quadrangles = find_quadrangles(obs['station_ids'], meta['n_stations'])
np.savez(os.path.join(fix_pre, 'find_quadrangles.npz'),
         input_station_ids=obs['station_ids'],
         input_n_stations=np.array(meta['n_stations']),
         output_quadrangles=quadrangles)

# compute_closure_phases
vis = obs['vis_corrupt']
cphase = compute_closure_phases(vis, obs['station_ids'], triangles)
np.savez(os.path.join(fix_pre, 'compute_closure_phases.npz'),
         input_vis=vis,
         input_station_ids=obs['station_ids'],
         input_triangles=triangles,
         output_cphase=cphase)

# compute_log_closure_amplitudes
logcamp = compute_log_closure_amplitudes(vis, obs['station_ids'], quadrangles)
np.savez(os.path.join(fix_pre, 'compute_log_closure_amplitudes.npz'),
         input_vis=vis,
         input_station_ids=obs['station_ids'],
         input_quadrangles=quadrangles,
         output_logcamp=logcamp)

# closure_phase_sigma
sigma_cp = closure_phase_sigma(obs['sigma_vis'], vis, obs['station_ids'], triangles)
np.savez(os.path.join(fix_pre, 'closure_phase_sigma.npz'),
         input_sigma_vis=obs['sigma_vis'],
         input_vis=vis,
         input_station_ids=obs['station_ids'],
         input_triangles=triangles,
         output_sigma_cp=sigma_cp)

# closure_amplitude_sigma
sigma_lca = closure_amplitude_sigma(obs['sigma_vis'], vis, obs['station_ids'], quadrangles)
np.savez(os.path.join(fix_pre, 'closure_amplitude_sigma.npz'),
         input_sigma_vis=obs['sigma_vis'],
         input_vis=vis,
         input_station_ids=obs['station_ids'],
         input_quadrangles=quadrangles,
         output_sigma_lca=sigma_lca)

# ── Physics model fixtures ──────────────────────────────────────────────
print('Generating physics model fixtures ...')
fix_pm = os.path.join(FIX_DIR, 'physics_model')
os.makedirs(fix_pm, exist_ok=True)

model = ClosureForwardModel(
    uv_coords=obs['uv_coords'], N=N, pixel_size_rad=psize,
    triangles=triangles, quadrangles=quadrangles,
    station_ids=obs['station_ids'],
)

gt = np.load(os.path.join(TASK_DIR, 'evaluation', 'reference_outputs', 'ground_truth.npy'))
gt_jy = np.load(os.path.join(TASK_DIR, 'evaluation', 'reference_outputs', 'ground_truth_jy.npy'))

# forward
vis_model = model.forward(gt_jy)
np.savez(os.path.join(fix_pm, 'forward_unit.npz'),
         param_uv_coords=obs['uv_coords'][:10],
         param_N=np.array(N),
         param_pixel_size_rad=np.array(psize),
         input_image=gt_jy,
         output_vis=vis_model)

# adjoint
adjoint_img = model.adjoint(vis_model)
np.savez(os.path.join(fix_pm, 'adjoint.npz'),
         input_vis=vis_model,
         output_image=adjoint_img)

# dirty image
dirty = model.dirty_image(obs['vis_cal'])
np.savez(os.path.join(fix_pm, 'dirty_image.npz'),
         input_vis=obs['vis_cal'],
         output_image=dirty)

# psf
psf = model.psf()
np.savez(os.path.join(fix_pm, 'psf.npz'),
         output_psf=psf)

# ── Solver/regularizer fixtures ─────────────────────────────────────────
print('Generating solver fixtures ...')
fix_sol = os.path.join(FIX_DIR, 'solvers')
os.makedirs(fix_sol, exist_ok=True)

prior = np.load(os.path.join(TASK_DIR, 'evaluation', 'reference_outputs', 'prior_image.npy'))

# TV regularizer
tv_reg = TVRegularizer(epsilon=1e-6)
tv_val, tv_grad = tv_reg.value_and_grad(gt_jy)
np.savez(os.path.join(fix_sol, 'tv_regularizer.npz'),
         input_image=gt_jy,
         output_val=np.array(tv_val),
         output_grad=tv_grad)

# ── Visualization fixtures ──────────────────────────────────────────────
print('Generating visualization fixtures ...')
fix_vis = os.path.join(FIX_DIR, 'visualization')
os.makedirs(fix_vis, exist_ok=True)

metrics = compute_metrics(dirty, gt)
np.savez(os.path.join(fix_vis, 'compute_metrics.npz'),
         input_estimate=dirty,
         input_reference=gt,
         output_nrmse=np.array(metrics['nrmse']),
         output_ncc=np.array(metrics['ncc']))

print('Done! All fixtures generated.')
