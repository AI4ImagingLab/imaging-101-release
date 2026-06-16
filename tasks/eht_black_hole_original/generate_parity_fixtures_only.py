#!/usr/bin/env python
"""
Generate ONLY the missing parity-test fixtures from ehtim.
Skips the expensive imaging reconstruction step.

Missing fixtures:
  - fixtures/physics_model/ftmatrix.npz       (DFT matrix rows)
  - fixtures/physics_model/amatrices.npz      (full A matrices)
  - fixtures/physics_model/chisq_cphase.npz   (closure phase chi2 + grad)
  - fixtures/physics_model/chisq_logcamp.npz  (log closure amp chi2 + grad)
  - fixtures/physics_model/forward.npz        (forward model reference)
  - fixtures/solvers/regularizers.npz         (GS + Simple entropy from ehtim)
"""

import os, sys, json
import numpy as np

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TASK_DIR)

FIX_DIR = os.path.join(TASK_DIR, 'evaluation', 'fixtures')
REF_DIR = os.path.join(TASK_DIR, 'evaluation', 'reference_outputs')
DATA_DIR = os.path.join(TASK_DIR, 'data')

# ── Load existing data ──────────────────────────────────────────────────
print("Loading observation data...")
raw = np.load(os.path.join(DATA_DIR, 'raw_data.npz'))
with open(os.path.join(DATA_DIR, 'meta_data')) as f:
    meta = json.load(f)

N = meta['N']
psize = meta['pixel_size_rad']
total_flux = meta['total_flux']

uv_coords = raw['uv_coords']
vis_cal = raw['vis_cal']
sigma_vis = raw['sigma_vis']

# Closure phase data
cp_values_deg = raw['cp_values_deg']
cp_sigmas_deg = raw['cp_sigmas_deg']
cp_u1, cp_u2, cp_u3 = raw['cp_u1'], raw['cp_u2'], raw['cp_u3']

# Log closure amplitude data
lca_values = raw['lca_values']
lca_sigmas = raw['lca_sigmas']
lca_u1, lca_u2, lca_u3, lca_u4 = raw['lca_u1'], raw['lca_u2'], raw['lca_u3'], raw['lca_u4']

# Ground truth images
gt_jy = np.load(os.path.join(REF_DIR, 'ground_truth_jy.npy'))
gt = np.load(os.path.join(REF_DIR, 'ground_truth.npy'))

# ── Use ehtim to generate reference values ──────────────────────────────
print("Importing ehtim...")
import ehtim
import ehtim.const_def as ehc
from ehtim.observing import obs_helpers as obsh
from ehtim.imaging import imager_utils

# Build ehtim Image object (needed for pulse function)
ra_hours = meta.get('ra_hours', 12.513728717168174)
dec_deg = meta.get('dec_deg', 12.39112323919932)
im_gt = ehtim.image.Image(gt_jy, psize, ra_hours, dec_deg,
                          rf=230e9, source='M87', mjd=57849)

# Build a Gaussian prior image (matching generate_ehtim_references.py)
im0 = ehtim.image.make_empty(N, im_gt.fovx(), ra_hours, dec_deg,
                             rf=230e9, source='M87')
im0 = im0.add_gauss(total_flux,
                     (20 * ehc.RADPERUAS, 20 * ehc.RADPERUAS, 0, 0, 0))

# ── 1. DFT matrix fixtures ─────────────────────────────────────────────
print("Generating DFT matrix fixtures...")
fix_pm = os.path.join(FIX_DIR, 'physics_model')
os.makedirs(fix_pm, exist_ok=True)

# Full DFT matrix
A_full = obsh.ftmatrix(psize, N, N, uv_coords, pulse=im_gt.pulse)
print(f"  A_full shape: {A_full.shape}")

# Forward model visibilities
vis_model = A_full @ gt_jy.flatten()

# Save ftmatrix (first 50 rows for parity test)
n_rows_save = min(50, A_full.shape[0])
np.savez(os.path.join(fix_pm, 'ftmatrix.npz'),
         A_rows=A_full[:n_rows_save],
         uv_rows=uv_coords[:n_rows_save],
         pixel_size_rad=np.array(psize),
         N=np.array(N))
print("  Saved ftmatrix.npz")

# Save forward model reference
np.savez(os.path.join(fix_pm, 'forward.npz'),
         input_image=gt_jy,
         output_vis=vis_model)
print("  Saved forward.npz")

# ── 2. Closure chi-squared fixtures ─────────────────────────────────────
print("Computing closure phase chi-squared...")

# Build per-uv A matrices
A_cp1 = obsh.ftmatrix(psize, N, N, cp_u1, pulse=im_gt.pulse)
A_cp2 = obsh.ftmatrix(psize, N, N, cp_u2, pulse=im_gt.pulse)
A_cp3 = obsh.ftmatrix(psize, N, N, cp_u3, pulse=im_gt.pulse)
A_cp = (A_cp1, A_cp2, A_cp3)

A_lca1 = obsh.ftmatrix(psize, N, N, lca_u1, pulse=im_gt.pulse)
A_lca2 = obsh.ftmatrix(psize, N, N, lca_u2, pulse=im_gt.pulse)
A_lca3 = obsh.ftmatrix(psize, N, N, lca_u3, pulse=im_gt.pulse)
A_lca4 = obsh.ftmatrix(psize, N, N, lca_u4, pulse=im_gt.pulse)
A_lca = (A_lca1, A_lca2, A_lca3, A_lca4)

# Save amatrices
np.savez(os.path.join(fix_pm, 'amatrices.npz'),
         A_full=A_full,
         A_cp1=A_cp1, A_cp2=A_cp2, A_cp3=A_cp3,
         A_lca1=A_lca1, A_lca2=A_lca2, A_lca3=A_lca3, A_lca4=A_lca4,
         uv_coords=uv_coords,
         cp_u1=cp_u1, cp_u2=cp_u2, cp_u3=cp_u3,
         lca_u1=lca_u1, lca_u2=lca_u2, lca_u3=lca_u3, lca_u4=lca_u4)
print("  Saved amatrices.npz")

# Test images
test_vec = gt_jy.flatten()
np.random.seed(123)
perturbed_img = gt_jy + 0.01 * np.random.randn(N, N)
perturbed_img = np.maximum(perturbed_img, 1e-10)
perturbed_vec = perturbed_img.flatten()

# Closure phase chi-squared (ehtim reference)
chisq_cp_gt = imager_utils.chisq_cphase(test_vec, A_cp, cp_values_deg, cp_sigmas_deg)
grad_cp_gt = imager_utils.chisqgrad_cphase(test_vec, A_cp, cp_values_deg, cp_sigmas_deg)
chisq_cp_pert = imager_utils.chisq_cphase(perturbed_vec, A_cp, cp_values_deg, cp_sigmas_deg)
grad_cp_pert = imager_utils.chisqgrad_cphase(perturbed_vec, A_cp, cp_values_deg, cp_sigmas_deg)
print(f"  chisq_cphase (gt): {chisq_cp_gt:.6f}")
print(f"  chisq_cphase (pert): {chisq_cp_pert:.6f}")

np.savez(os.path.join(fix_pm, 'chisq_cphase.npz'),
         input_image_gt=gt_jy,
         input_image_pert=perturbed_img,
         cp_values_deg=cp_values_deg,
         cp_sigmas_deg=cp_sigmas_deg,
         cp_u1=cp_u1, cp_u2=cp_u2, cp_u3=cp_u3,
         output_chisq_gt=np.array(chisq_cp_gt),
         output_grad_gt=grad_cp_gt,
         output_chisq_pert=np.array(chisq_cp_pert),
         output_grad_pert=grad_cp_pert)
print("  Saved chisq_cphase.npz")

# Log closure amplitude chi-squared (ehtim reference)
print("Computing log closure amplitude chi-squared...")
chisq_lca_gt = imager_utils.chisq_logcamp(test_vec, A_lca, lca_values, lca_sigmas)
grad_lca_gt = imager_utils.chisqgrad_logcamp(test_vec, A_lca, lca_values, lca_sigmas)
chisq_lca_pert = imager_utils.chisq_logcamp(perturbed_vec, A_lca, lca_values, lca_sigmas)
grad_lca_pert = imager_utils.chisqgrad_logcamp(perturbed_vec, A_lca, lca_values, lca_sigmas)
print(f"  chisq_logcamp (gt): {chisq_lca_gt:.6f}")
print(f"  chisq_logcamp (pert): {chisq_lca_pert:.6f}")

np.savez(os.path.join(fix_pm, 'chisq_logcamp.npz'),
         input_image_gt=gt_jy,
         input_image_pert=perturbed_img,
         lca_values=lca_values,
         lca_sigmas=lca_sigmas,
         lca_u1=lca_u1, lca_u2=lca_u2, lca_u3=lca_u3, lca_u4=lca_u4,
         output_chisq_gt=np.array(chisq_lca_gt),
         output_grad_gt=grad_lca_gt,
         output_chisq_pert=np.array(chisq_lca_pert),
         output_grad_pert=grad_lca_pert)
print("  Saved chisq_logcamp.npz")

# ── 3. Regularizer fixtures ────────────────────────────────────────────
print("Computing regularizer reference values...")
fix_sol = os.path.join(FIX_DIR, 'solvers')
os.makedirs(fix_sol, exist_ok=True)

prior_vec = im0.imvec
prior_img = prior_vec.reshape(N, N)

# GS entropy (ehtim reference, norm_reg=False)
gs_val = imager_utils.sgs(test_vec, prior_vec, total_flux, norm_reg=False)
gs_grad = imager_utils.sgsgrad(test_vec, prior_vec, total_flux, norm_reg=False)
print(f"  sgs (gt): {gs_val:.6f}")

# Simple entropy (ehtim reference, norm_reg=False)
simple_val = imager_utils.ssimple(test_vec, prior_vec, total_flux, norm_reg=False)
simple_grad = imager_utils.ssimplegrad(test_vec, prior_vec, total_flux, norm_reg=False)
print(f"  ssimple (gt): {simple_val:.6f}")

np.savez(os.path.join(fix_sol, 'regularizers.npz'),
         input_image=gt_jy,
         prior_image=prior_img,
         gs_val=np.array(gs_val),
         gs_grad=gs_grad,
         simple_val=np.array(simple_val),
         simple_grad=simple_grad,
         total_flux=np.array(total_flux))
print("  Saved regularizers.npz")

# Also save the prior image for reference
np.save(os.path.join(REF_DIR, 'prior_image.npy'), prior_img)
print("  Saved prior_image.npy")

print("\n" + "="*60)
print("All parity fixtures generated successfully!")
print("="*60)
