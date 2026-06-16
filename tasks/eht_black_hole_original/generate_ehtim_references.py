#!/usr/bin/env python
"""
Generate Reference Outputs from ehtim
======================================

This script runs the ehtim pipeline (mirroring validate_with_ehtim.ipynb)
and saves all intermediate + final outputs for parity testing.

This is a BUILD TOOL, not part of the final task. It requires ehtim.

Outputs:
  data/raw_data.npz         — observation data (calibrated + corrupted)
  data/meta_data             — imaging parameters as JSON
  evaluation/fixtures/       — per-function parity fixtures
  evaluation/reference_outputs/ — final images and metrics

Usage:
  cd tasks/eht_black_hole_original
  python generate_ehtim_references.py
"""

import os, sys, io, json, tempfile, warnings
import numpy as np

warnings.filterwarnings('ignore')

import ehtim
import ehtim.const_def as ehc
from ehtim.observing import obs_helpers as obsh
from ehtim.imaging import imager_utils

# Add task root to path
TASK_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TASK_DIR)
from src.generate_data import make_ring_image

# ── Directories ──────────────────────────────────────────────────────────
DATA_DIR = os.path.join(TASK_DIR, 'data')
FIX_DIR = os.path.join(TASK_DIR, 'evaluation', 'fixtures')
REF_DIR = os.path.join(TASK_DIR, 'evaluation', 'reference_outputs')

for d in [DATA_DIR,
          os.path.join(FIX_DIR, 'preprocessing'),
          os.path.join(FIX_DIR, 'physics_model'),
          os.path.join(FIX_DIR, 'solvers'),
          os.path.join(FIX_DIR, 'visualization'),
          REF_DIR]:
    os.makedirs(d, exist_ok=True)


# ═════════════════════════════════════════════════════════════════════════
# 1. Ground truth image + EHT array
# ═════════════════════════════════════════════════════════════════════════
print('=' * 60)
print('Step 1: Ground truth image + EHT array')
print('=' * 60)

N = 64
total_flux = 0.6  # Jy
pixel_size_uas = 2.0
psize = pixel_size_uas * ehc.RADPERUAS

gt = make_ring_image(N=N)
gt_jy = gt * total_flux  # Jy/pixel

ra_hours = 12.513728717168174
dec_deg = 12.39112323919932

im_gt = ehtim.image.Image(gt_jy, psize, ra_hours, dec_deg,
                          rf=230e9, source='M87', mjd=57849)
print(f'  Image: {N}x{N}, pixel={pixel_size_uas} uas, flux={im_gt.total_flux():.3f} Jy')

# EHT 2017 array
arr_txt = """ALMA   2225061.164  -5440057.370  -2481681.150    90
APEX   2225039.530  -5441197.630  -2479303.360  3500
JCMT  -5464584.680  -2493001.170   2150653.980  6000
SMA   -5464523.400  -2493147.080   2150611.750  4900
SMT   -1828796.200  -5054406.800   3427865.200  5000
LMT    -768713.964  -5988541.798   2063275.947   600
PV     5088967.900   -301681.600   3825015.800  1400
SPT          0.010         0.010  -6359609.700  5000
"""
tmpf = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
tmpf.write(arr_txt); tmpf.close()
eht = ehtim.array.load_txt(tmpf.name)
os.unlink(tmpf.name)

station_names = [t['site'] for t in eht.tarr]
print(f'  Array: {len(station_names)} stations: {station_names}')


# ═════════════════════════════════════════════════════════════════════════
# 2. Simulate observations (calibrated + corrupted)
# ═════════════════════════════════════════════════════════════════════════
print('\n' + '=' * 60)
print('Step 2: Simulate observations')
print('=' * 60)

tint, tadv, bw = 10, 600, 2e9
tstart, tstop = 0.0, 6.0

obs_cal = im_gt.observe(eht, tint=tint, tadv=tadv, tstart=tstart, tstop=tstop,
                        bw=bw, add_th_noise=True,
                        ampcal=True, phasecal=True,
                        ttype='direct', seed=42, verbose=False)
obs_cal = obs_cal.add_fractional_noise(0.01)

obs_corrupt = im_gt.observe(eht, tint=tint, tadv=tadv, tstart=tstart, tstop=tstop,
                            bw=bw, add_th_noise=True,
                            ampcal=False, phasecal=False,
                            gainp=0.2,
                            ttype='direct', seed=42, verbose=False)
obs_corrupt = obs_corrupt.add_fractional_noise(0.01)

# Also generate corrupted obs on same baselines for gain-invariance test
obs_corrupt_same = im_gt.observe_same(obs_cal,
                                      add_th_noise=True,
                                      ampcal=False, phasecal=False,
                                      gainp=0.2,
                                      ttype='direct', seed=42, verbose=False)

n_vis = len(obs_cal.data)
print(f'  Calibrated obs:  {n_vis} visibilities')
print(f'  Corrupted obs:   {len(obs_corrupt.data)} visibilities')
print(f'  Resolution:      {obs_cal.res() / ehc.RADPERUAS:.1f} uas')


# ═════════════════════════════════════════════════════════════════════════
# 3. Extract observation data arrays
# ═════════════════════════════════════════════════════════════════════════
print('\n' + '=' * 60)
print('Step 3: Extract observation data arrays')
print('=' * 60)

vis_cal = obs_cal.data['vis']
vis_corrupt = obs_corrupt.data['vis']
sigma_vis = obs_cal.data['sigma']
u = obs_cal.data['u']
v = obs_cal.data['v']
uv_coords = np.column_stack([u, v])
t1_names = obs_cal.data['t1']
t2_names = obs_cal.data['t2']

# Map station names to integer IDs
all_stations = sorted(set(t1_names) | set(t2_names))
name_to_id = {name: i for i, name in enumerate(all_stations)}
station_ids = np.column_stack([
    np.array([name_to_id[n] for n in t1_names]),
    np.array([name_to_id[n] for n in t2_names])
])

print(f'  UV coords shape: {uv_coords.shape}')
print(f'  Stations: {all_stations}')
print(f'  Noise std range: [{sigma_vis.min():.4e}, {sigma_vis.max():.4e}]')


# ═════════════════════════════════════════════════════════════════════════
# 4. Compute closure quantities with ehtim
# ═════════════════════════════════════════════════════════════════════════
print('\n' + '=' * 60)
print('Step 4: Closure quantities')
print('=' * 60)

# Closure phases
cp_data = obs_cal.c_phases(mode='all', count='min', vtype='vis')
cp_corrupt_data = obs_corrupt.c_phases(mode='all', count='min', vtype='vis')

# Closure amplitudes
lca_data = obs_cal.c_amplitudes(mode='all', count='min', vtype='vis', ctype='logcamp')
lca_corrupt_data = obs_corrupt.c_amplitudes(mode='all', count='min', vtype='vis', ctype='logcamp')

print(f'  Closure phases:  {len(cp_data)} triangles')
print(f'  Log closure amps: {len(lca_data)} quadrangles')

# Extract closure phase data
cp_values_deg = cp_data['cphase']
cp_sigmas_deg = cp_data['sigmacp']
cp_t1 = cp_data['t1']
cp_t2 = cp_data['t2']
cp_t3 = cp_data['t3']
cp_u1 = np.column_stack([cp_data['u1'], cp_data['v1']])
cp_u2 = np.column_stack([cp_data['u2'], cp_data['v2']])
cp_u3 = np.column_stack([cp_data['u3'], cp_data['v3']])

# Extract log closure amplitude data
lca_values = lca_data['camp']
lca_sigmas = lca_data['sigmaca']
lca_t1 = lca_data['t1']
lca_t2 = lca_data['t2']
lca_t3 = lca_data['t3']
lca_t4 = lca_data['t4']
lca_u1 = np.column_stack([lca_data['u1'], lca_data['v1']])
lca_u2 = np.column_stack([lca_data['u2'], lca_data['v2']])
lca_u3 = np.column_stack([lca_data['u3'], lca_data['v3']])
lca_u4 = np.column_stack([lca_data['u4'], lca_data['v4']])


# ═════════════════════════════════════════════════════════════════════════
# 5. DFT matrix and forward model (using ehtim's ftmatrix)
# ═════════════════════════════════════════════════════════════════════════
print('\n' + '=' * 60)
print('Step 5: DFT matrix (ehtim ftmatrix)')
print('=' * 60)

im0 = ehtim.image.make_empty(N, im_gt.fovx(), ra_hours, dec_deg,
                             rf=230e9, source='M87')
im0 = im0.add_gauss(total_flux,
                     (20 * ehc.RADPERUAS, 20 * ehc.RADPERUAS, 0, 0, 0))

# Full DFT matrix for all baselines
A_full = obsh.ftmatrix(psize, N, N, uv_coords, pulse=im_gt.pulse)
print(f'  DFT matrix shape: {A_full.shape}')

# Model visibilities from ground truth
gt_vec = gt_jy.flatten()
vis_model = A_full @ gt_vec
print(f'  Model vis residual vs data: {np.mean(np.abs(vis_model - vis_cal)):.4e}')

# Closure phase Amatrices
A_cp1 = obsh.ftmatrix(psize, N, N, cp_u1, pulse=im_gt.pulse)
A_cp2 = obsh.ftmatrix(psize, N, N, cp_u2, pulse=im_gt.pulse)
A_cp3 = obsh.ftmatrix(psize, N, N, cp_u3, pulse=im_gt.pulse)
A_cp = (A_cp1, A_cp2, A_cp3)

# Log closure amplitude Amatrices
A_lca1 = obsh.ftmatrix(psize, N, N, lca_u1, pulse=im_gt.pulse)
A_lca2 = obsh.ftmatrix(psize, N, N, lca_u2, pulse=im_gt.pulse)
A_lca3 = obsh.ftmatrix(psize, N, N, lca_u3, pulse=im_gt.pulse)
A_lca4 = obsh.ftmatrix(psize, N, N, lca_u4, pulse=im_gt.pulse)
A_lca = (A_lca1, A_lca2, A_lca3, A_lca4)


# ═════════════════════════════════════════════════════════════════════════
# 6. Chi-squared and gradients (ehtim reference values)
# ═════════════════════════════════════════════════════════════════════════
print('\n' + '=' * 60)
print('Step 6: Chi-squared and gradients')
print('=' * 60)

# Use the ground truth image for parity testing
test_img = gt_jy.copy()
test_vec = test_img.flatten()

# Also test with a perturbed image (not all zeros, not the answer)
np.random.seed(123)
perturbed_img = test_img + 0.01 * np.random.randn(N, N)
perturbed_img = np.maximum(perturbed_img, 1e-10)
perturbed_vec = perturbed_img.flatten()

# Closure phase chi-squared
chisq_cp_gt = imager_utils.chisq_cphase(test_vec, A_cp, cp_values_deg, cp_sigmas_deg)
grad_cp_gt = imager_utils.chisqgrad_cphase(test_vec, A_cp, cp_values_deg, cp_sigmas_deg)
chisq_cp_pert = imager_utils.chisq_cphase(perturbed_vec, A_cp, cp_values_deg, cp_sigmas_deg)
grad_cp_pert = imager_utils.chisqgrad_cphase(perturbed_vec, A_cp, cp_values_deg, cp_sigmas_deg)

print(f'  chisq_cphase (gt):    {chisq_cp_gt:.6f}')
print(f'  chisq_cphase (pert):  {chisq_cp_pert:.6f}')

# Log closure amplitude chi-squared
chisq_lca_gt = imager_utils.chisq_logcamp(test_vec, A_lca, lca_values, lca_sigmas)
grad_lca_gt = imager_utils.chisqgrad_logcamp(test_vec, A_lca, lca_values, lca_sigmas)
chisq_lca_pert = imager_utils.chisq_logcamp(perturbed_vec, A_lca, lca_values, lca_sigmas)
grad_lca_pert = imager_utils.chisqgrad_logcamp(perturbed_vec, A_lca, lca_values, lca_sigmas)

print(f'  chisq_logcamp (gt):   {chisq_lca_gt:.6f}')
print(f'  chisq_logcamp (pert): {chisq_lca_pert:.6f}')

# Visibility chi-squared (for comparison)
chisq_vis = imager_utils.chisq_vis(test_vec, A_full, vis_cal, sigma_vis)
grad_vis = imager_utils.chisqgrad_vis(test_vec, A_full, vis_cal, sigma_vis)
print(f'  chisq_vis (gt):       {chisq_vis:.6f}')

# Regularizers
prior_vec = im0.imvec
gs_val = imager_utils.sgs(test_vec, prior_vec, total_flux, norm_reg=False)
gs_grad = imager_utils.sgsgrad(test_vec, prior_vec, total_flux, norm_reg=False)
simple_val = imager_utils.ssimple(test_vec, prior_vec, total_flux, norm_reg=False)
simple_grad = imager_utils.ssimplegrad(test_vec, prior_vec, total_flux, norm_reg=False)
print(f'  sgs (gt):             {gs_val:.6f}')
print(f'  ssimple (gt):         {simple_val:.6f}')


# ═════════════════════════════════════════════════════════════════════════
# 7. Image reconstruction with ehtim
# ═════════════════════════════════════════════════════════════════════════
print('\n' + '=' * 60)
print('Step 7: Image reconstruction')
print('=' * 60)

old_stdout = sys.stdout

def run_ehtim_imaging(obs, data_term, reg_term, label, niter=3, maxit=300):
    global old_stdout
    imgr = ehtim.imager.Imager(obs, im0, prior_im=im0, flux=total_flux,
                               data_term=data_term, reg_term=reg_term,
                               maxit=maxit, ttype='direct')
    sys.stdout = io.StringIO()
    out = imgr.make_image_I(niter=niter, blur_frac=0, show_updates=False)
    sys.stdout = old_stdout
    return out

reg = {'gs': 1, 'simple': 10}

# Calibrated data
print('  Calibrated data:')
print('    [1/3] Vis RML ...')
out_vis_cal = run_ehtim_imaging(obs_cal,
    data_term={'vis': 100}, reg_term=reg, label='Vis')

print('    [2/3] Amp + CPhase ...')
out_amp_cal = run_ehtim_imaging(obs_cal,
    data_term={'amp': 100, 'cphase': 50}, reg_term=reg, label='Amp+CP')

print('    [3/3] Closure-only ...')
out_clo_cal = run_ehtim_imaging(obs_cal,
    data_term={'cphase': 50, 'logcamp': 50}, reg_term=reg, label='Closure')

# Corrupted data
print('  Corrupted data:')
print('    [1/3] Vis RML ...')
out_vis_cor = run_ehtim_imaging(obs_corrupt,
    data_term={'vis': 100}, reg_term=reg, label='Vis')

print('    [2/3] Amp + CPhase ...')
out_amp_cor = run_ehtim_imaging(obs_corrupt,
    data_term={'amp': 100, 'cphase': 50}, reg_term=reg, label='Amp+CP')

print('    [3/3] Closure-only ...')
out_clo_cor = run_ehtim_imaging(obs_corrupt,
    data_term={'cphase': 50, 'logcamp': 50}, reg_term=reg, label='Closure')


# ═════════════════════════════════════════════════════════════════════════
# 8. Compute metrics
# ═════════════════════════════════════════════════════════════════════════
print('\n' + '=' * 60)
print('Step 8: Metrics')
print('=' * 60)

def compute_metrics(est, ref):
    est_n = est * (ref.sum() / (est.sum() + 1e-30))
    nrmse = float(np.sqrt(np.mean((est_n - ref)**2))
                  / (np.sqrt(np.mean(ref**2)) + 1e-30))
    ncc = float(np.sum(est_n * ref)
               / (np.sqrt(np.sum(est_n**2)) * np.sqrt(np.sum(ref**2)) + 1e-30))
    return {'nrmse': round(nrmse, 4), 'ncc': round(ncc, 4)}

results = {
    'Vis RML (cal)': out_vis_cal.imarr(),
    'Amp+CP (cal)': out_amp_cal.imarr(),
    'Closure-only (cal)': out_clo_cal.imarr(),
    'Vis RML (corrupt)': out_vis_cor.imarr(),
    'Amp+CP (corrupt)': out_amp_cor.imarr(),
    'Closure-only (corrupt)': out_clo_cor.imarr(),
}

metrics = {}
for name, img in results.items():
    metrics[name] = compute_metrics(img, gt)
    print(f'  {name:30s}  NRMSE={metrics[name]["nrmse"]:.4f}  NCC={metrics[name]["ncc"]:.4f}')


# ═════════════════════════════════════════════════════════════════════════
# 9. Save everything
# ═════════════════════════════════════════════════════════════════════════
print('\n' + '=' * 60)
print('Step 9: Saving outputs')
print('=' * 60)

# -- data/ --
noise_std_scalar = float(np.median(sigma_vis))
np.savez(os.path.join(DATA_DIR, 'raw_data.npz'),
         vis_cal=vis_cal,
         vis_corrupt=vis_corrupt,
         uv_coords=uv_coords,
         sigma_vis=sigma_vis,
         station_ids=station_ids,
         # Per-scan closure data from ehtim (for solver)
         cp_values_deg=cp_values_deg,
         cp_sigmas_deg=cp_sigmas_deg,
         cp_u1=cp_u1, cp_u2=cp_u2, cp_u3=cp_u3,
         lca_values=lca_values,
         lca_sigmas=lca_sigmas,
         lca_u1=lca_u1, lca_u2=lca_u2, lca_u3=lca_u3, lca_u4=lca_u4,
         # Corrupted closure data
         cp_corrupt_values_deg=obs_corrupt.c_phases(mode='all', count='min', vtype='vis')['cphase'],
         cp_corrupt_sigmas_deg=obs_corrupt.c_phases(mode='all', count='min', vtype='vis')['sigmacp'],
         lca_corrupt_values=obs_corrupt.c_amplitudes(mode='all', count='min', vtype='vis', ctype='logcamp')['camp'],
         lca_corrupt_sigmas=obs_corrupt.c_amplitudes(mode='all', count='min', vtype='vis', ctype='logcamp')['sigmaca'])
print(f'  Saved data/raw_data.npz')

meta = {
    'N': N,
    'pixel_size_uas': pixel_size_uas,
    'pixel_size_rad': float(psize),
    'total_flux': total_flux,
    'noise_std': noise_std_scalar,
    'freq_ghz': 230.0,
    'n_baselines': n_vis,
    'n_stations': len(all_stations),
    'station_names': all_stations,
    'gain_amp_error': 0.2,
    'gain_phase_error_deg': 30.0,
    'ra_hours': ra_hours,
    'dec_deg': dec_deg,
}
with open(os.path.join(DATA_DIR, 'meta_data'), 'w') as f:
    json.dump(meta, f, indent=2)
print(f'  Saved data/meta_data')

# -- evaluation/fixtures/preprocessing/ --
# Closure phase data from ehtim
np.savez(os.path.join(FIX_DIR, 'preprocessing', 'closure_phases.npz'),
         cp_values_deg=cp_values_deg,
         cp_sigmas_deg=cp_sigmas_deg,
         cp_t1=cp_t1, cp_t2=cp_t2, cp_t3=cp_t3,
         cp_u1=cp_u1, cp_u2=cp_u2, cp_u3=cp_u3,
         # Also the corrupted closure phases for gain-invariance
         cp_corrupt_values_deg=obs_corrupt.c_phases(mode='all', count='min', vtype='vis')['cphase'])

np.savez(os.path.join(FIX_DIR, 'preprocessing', 'closure_amplitudes.npz'),
         lca_values=lca_values,
         lca_sigmas=lca_sigmas,
         lca_t1=lca_t1, lca_t2=lca_t2, lca_t3=lca_t3, lca_t4=lca_t4,
         lca_u1=lca_u1, lca_u2=lca_u2, lca_u3=lca_u3, lca_u4=lca_u4)

print(f'  Saved fixtures/preprocessing/')

# -- evaluation/fixtures/physics_model/ --
# DFT matrix (save a few rows to keep file size reasonable)
n_rows_save = min(50, A_full.shape[0])
np.savez(os.path.join(FIX_DIR, 'physics_model', 'ftmatrix.npz'),
         A_rows=A_full[:n_rows_save],
         uv_rows=uv_coords[:n_rows_save],
         pixel_size_rad=np.array(psize),
         N=np.array(N))

# Forward model
np.savez(os.path.join(FIX_DIR, 'physics_model', 'forward.npz'),
         input_image=gt_jy,
         output_vis=vis_model)

# Closure phase chi-squared
np.savez(os.path.join(FIX_DIR, 'physics_model', 'chisq_cphase.npz'),
         input_image_gt=test_img,
         input_image_pert=perturbed_img,
         cp_values_deg=cp_values_deg,
         cp_sigmas_deg=cp_sigmas_deg,
         cp_u1=cp_u1, cp_u2=cp_u2, cp_u3=cp_u3,
         output_chisq_gt=np.array(chisq_cp_gt),
         output_grad_gt=grad_cp_gt,
         output_chisq_pert=np.array(chisq_cp_pert),
         output_grad_pert=grad_cp_pert)

# Log closure amplitude chi-squared
np.savez(os.path.join(FIX_DIR, 'physics_model', 'chisq_logcamp.npz'),
         input_image_gt=test_img,
         input_image_pert=perturbed_img,
         lca_values=lca_values,
         lca_sigmas=lca_sigmas,
         lca_u1=lca_u1, lca_u2=lca_u2, lca_u3=lca_u3, lca_u4=lca_u4,
         output_chisq_gt=np.array(chisq_lca_gt),
         output_grad_gt=grad_lca_gt,
         output_chisq_pert=np.array(chisq_lca_pert),
         output_grad_pert=grad_lca_pert)

# Visibility chi-squared
np.savez(os.path.join(FIX_DIR, 'physics_model', 'chisq_vis.npz'),
         input_image=test_img,
         output_chisq=np.array(chisq_vis),
         output_grad=grad_vis)

# Regularizers
np.savez(os.path.join(FIX_DIR, 'solvers', 'regularizers.npz'),
         input_image=test_img,
         prior_image=im0.imvec.reshape(N, N),
         gs_val=np.array(gs_val),
         gs_grad=gs_grad,
         simple_val=np.array(simple_val),
         simple_grad=simple_grad,
         total_flux=np.array(total_flux))

print(f'  Saved fixtures/physics_model/')
print(f'  Saved fixtures/solvers/')

# -- evaluation/reference_outputs/ --
np.save(os.path.join(REF_DIR, 'ground_truth.npy'), gt)
np.save(os.path.join(REF_DIR, 'ground_truth_jy.npy'), gt_jy)
np.save(os.path.join(REF_DIR, 'prior_image.npy'), im0.imvec.reshape(N, N))

for name, img in results.items():
    fname = name.lower().replace(' ', '_').replace('+', '_').replace('(', '').replace(')', '')
    np.save(os.path.join(REF_DIR, f'{fname}.npy'), img)

with open(os.path.join(REF_DIR, 'metrics.json'), 'w') as f:
    json.dump(metrics, f, indent=2)

print(f'  Saved reference_outputs/')

# Also save the full A matrices for parity testing
np.savez(os.path.join(FIX_DIR, 'physics_model', 'amatrices.npz'),
         A_full=A_full,
         A_cp1=A_cp1, A_cp2=A_cp2, A_cp3=A_cp3,
         A_lca1=A_lca1, A_lca2=A_lca2, A_lca3=A_lca3, A_lca4=A_lca4,
         uv_coords=uv_coords,
         cp_u1=cp_u1, cp_u2=cp_u2, cp_u3=cp_u3,
         lca_u1=lca_u1, lca_u2=lca_u2, lca_u3=lca_u3, lca_u4=lca_u4)
print(f'  Saved fixtures/physics_model/amatrices.npz')

# -- Gain invariance test data --
cp_same_data = obs_corrupt_same.c_phases(mode='all', count='min', vtype='vis')
lca_same_data = obs_corrupt_same.c_amplitudes(mode='all', count='min', vtype='vis', ctype='logcamp')
np.savez(os.path.join(FIX_DIR, 'preprocessing', 'gain_invariance.npz'),
         amp_cal=np.abs(obs_cal.data['vis']),
         amp_corrupt=np.abs(obs_corrupt_same.data['vis']),
         cp_cal_deg=cp_values_deg,
         cp_corrupt_deg=cp_same_data['cphase'],
         lca_cal=lca_values,
         lca_corrupt=lca_same_data['camp'])

print(f'  Saved fixtures/preprocessing/gain_invariance.npz')

# Visibility chi-squared fixture for the vis RML solver
np.savez(os.path.join(FIX_DIR, 'physics_model', 'visibility.npz'),
         vis_cal=vis_cal,
         vis_corrupt=vis_corrupt,
         sigma_vis=sigma_vis,
         uv_coords=uv_coords)

print('\n' + '=' * 60)
print('DONE! All reference outputs generated.')
print('=' * 60)
