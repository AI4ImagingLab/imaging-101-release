"""
End-to-end synthetic data generation for the shapelet source reconstruction task.

Generates all intermediate and final outputs, saves them for evaluation and notebooks.
"""

import numpy as np
import json
import os

from .preprocessing import load_and_prepare_galaxy, decompose_shapelets, reconstruct_from_shapelets
from .physics_model import (
    make_grid, image2array, array2image, simulate_image,
    add_poisson_noise, add_background_noise, shapelet_function,
)
from .solvers import build_response_matrix, linear_solve, reduced_residuals


# Default parameters matching the original notebook
DEFAULT_PARAMS = {
    'background_rms': 10.0,
    'exp_time': 100.0,
    'numPix': 64,
    'deltaPix': 0.05,
    'high_res_factor': 5,
    'fwhm_lensing': 0.1,
    'fwhm_deconv': 0.25,
    'n_max_decomp': 150,
    'beta_decomp': 10,
    'n_max_recon': 20,
    'beta_recon_lensing': 0.15,
    'beta_recon_deconv': 0.3,
    'source_x': 0.2,
    'source_y': 0.0,
    'beta_model_lensing': 0.06,
    'beta_model_deconv': 0.12,
    'coeff_scale_deconv': 5.0,
    'theta_E': 0.5,
    'gamma': 2.0,
    'e1': 0.0,
    'e2': 0.0,
    'gamma1': 0.0,
    'gamma2': 0.0,
}


def generate_all_data(image_path, output_dir, params=None, seed=42):
    """Generate all synthetic data and save to output_dir.

    :param image_path: path to source galaxy image
    :param output_dir: directory to save outputs
    :param params: parameter dict (uses DEFAULT_PARAMS if None)
    :param seed: random seed for reproducibility
    :return: metrics dict
    """
    np.random.seed(seed)
    p = {**DEFAULT_PARAMS, **(params or {})}
    os.makedirs(output_dir, exist_ok=True)

    # ---- Stage 1: Preprocessing and shapelet decomposition ----
    print("Stage 1: Preprocessing and shapelet decomposition...")
    ngc_square, ngc_conv, ngc_resized, numPix_resized = load_and_prepare_galaxy(image_path)
    coeff_ngc = decompose_shapelets(ngc_resized, p['n_max_decomp'], p['beta_decomp'])
    image_reconstructed = reconstruct_from_shapelets(coeff_ngc, p['n_max_decomp'],
                                                      p['beta_decomp'], numPix_resized)

    # ---- Stage 2: Lensing simulation ----
    print("Stage 2: Lensing simulation...")
    kwargs_spemd = {'theta_E': p['theta_E'], 'gamma': p['gamma'],
                    'center_x': 0, 'center_y': 0, 'e1': p['e1'], 'e2': p['e2']}
    kwargs_shear = {'gamma1': p['gamma1'], 'gamma2': p['gamma2']}

    coeff_lens = coeff_ngc / p['deltaPix']**2
    kwargs_source_lens = {'n_max': p['n_max_decomp'], 'beta': p['beta_model_lensing'],
                          'amp': coeff_lens, 'center_x': p['source_x'], 'center_y': p['source_y']}

    numPix = p['numPix']
    deltaPix = p['deltaPix']
    hrf = p['high_res_factor']
    numPix_hr = numPix * hrf
    deltaPix_hr = deltaPix / hrf

    image_hr_nolens = simulate_image(numPix_hr, deltaPix_hr, 1, fwhm=0,
                                      kwargs_source=kwargs_source_lens, apply_lens=False, apply_psf=False)
    image_hr_lensed = simulate_image(numPix_hr, deltaPix_hr, 1, fwhm=0,
                                      kwargs_source=kwargs_source_lens, kwargs_spemd=kwargs_spemd,
                                      kwargs_shear=kwargs_shear, apply_lens=True, apply_psf=False)
    image_hr_conv = simulate_image(numPix_hr, deltaPix_hr, 1, fwhm=p['fwhm_lensing'],
                                    kwargs_source=kwargs_source_lens, kwargs_spemd=kwargs_spemd,
                                    kwargs_shear=kwargs_shear, apply_lens=True, apply_psf=True)
    image_no_noise = simulate_image(numPix, deltaPix, hrf, fwhm=p['fwhm_lensing'],
                                     kwargs_source=kwargs_source_lens, kwargs_spemd=kwargs_spemd,
                                     kwargs_shear=kwargs_shear, apply_lens=True, apply_psf=True)
    image_real = image_no_noise + add_poisson_noise(image_no_noise, p['exp_time']) \
                 + add_background_noise(image_no_noise, p['background_rms'])

    # ---- Stage 3: Source reconstruction ----
    print("Stage 3: Source reconstruction (lensed)...")
    A_lens = build_response_matrix(numPix, deltaPix, hrf, p['fwhm_lensing'],
                                    p['n_max_recon'], p['beta_recon_lensing'],
                                    p['source_x'], p['source_y'],
                                    kwargs_spemd=kwargs_spemd, kwargs_shear=kwargs_shear, apply_lens=True)
    param_lens, model_lens = linear_solve(A_lens, image_real, p['background_rms'], p['exp_time'])
    residuals_lens = reduced_residuals(model_lens, image_real, p['background_rms'], p['exp_time'])

    x_hr, y_hr = make_grid(numPix_hr, deltaPix_hr)
    source_recon = shapelet_function(x_hr, y_hr, param_lens, p['n_max_recon'],
                                      p['beta_recon_lensing'], p['source_x'], p['source_y'])
    source_recon_2d = array2image(source_recon) * deltaPix_hr**2

    # ---- Stage 4: Deconvolution ----
    print("Stage 4: Deconvolution (no lensing)...")
    coeff_dc = coeff_ngc * p['coeff_scale_deconv'] / deltaPix**2
    kwargs_source_dc = {'n_max': p['n_max_decomp'], 'beta': p['beta_model_deconv'],
                        'amp': coeff_dc, 'center_x': 0.0, 'center_y': 0.0}

    image_hr_dc = simulate_image(numPix_hr, deltaPix_hr, 1, fwhm=0,
                                  kwargs_source=kwargs_source_dc, apply_lens=False, apply_psf=False)
    image_hr_conv_dc = simulate_image(numPix_hr, deltaPix_hr, 1, fwhm=p['fwhm_deconv'],
                                       kwargs_source=kwargs_source_dc, apply_lens=False, apply_psf=True)
    image_no_noise_dc = simulate_image(numPix, deltaPix, hrf, fwhm=p['fwhm_deconv'],
                                        kwargs_source=kwargs_source_dc, apply_lens=False, apply_psf=True)
    image_real_dc = image_no_noise_dc + add_poisson_noise(image_no_noise_dc, p['exp_time']) \
                    + add_background_noise(image_no_noise_dc, p['background_rms'])

    A_dc = build_response_matrix(numPix, deltaPix, hrf, p['fwhm_deconv'],
                                  p['n_max_recon'], p['beta_recon_deconv'], 0.0, 0.0,
                                  apply_lens=False)
    param_dc, model_dc = linear_solve(A_dc, image_real_dc, p['background_rms'], p['exp_time'])
    residuals_dc = reduced_residuals(model_dc, image_real_dc, p['background_rms'], p['exp_time'])

    source_deconv = shapelet_function(x_hr, y_hr, param_dc, p['n_max_recon'],
                                       p['beta_recon_deconv'], 0.0, 0.0)
    source_deconv_2d = array2image(source_deconv) * deltaPix_hr**2

    # ---- Compute metrics ----
    chi2_lens = np.sum(residuals_lens**2) / residuals_lens.size
    chi2_dc = np.sum(residuals_dc**2) / residuals_dc.size
    metrics = {
        'chi2_reduced_lensing': float(chi2_lens),
        'chi2_reduced_deconv': float(chi2_dc),
        'num_shapelet_coeffs_decomp': int(len(coeff_ngc)),
        'num_shapelet_coeffs_recon': int(len(param_lens)),
        'source_recon_residual_rms': float(np.std(source_recon_2d - image_hr_nolens)),
        'deconv_residual_rms': float(np.std(source_deconv_2d - image_hr_dc)),
    }

    # ---- Save outputs ----
    print("Saving outputs...")
    np.savez(os.path.join(output_dir, 'raw_data.npz'),
             coeff_ngc=coeff_ngc,
             ngc_square=ngc_square,
             ngc_conv=ngc_conv,
             ngc_resized=ngc_resized,
             image_reconstructed=image_reconstructed)

    np.savez(os.path.join(output_dir, 'lensing_outputs.npz'),
             image_hr_nolens=image_hr_nolens,
             image_hr_lensed=image_hr_lensed,
             image_hr_conv=image_hr_conv,
             image_no_noise=image_no_noise,
             image_real=image_real,
             model_lens=model_lens,
             residuals_lens=residuals_lens,
             source_recon_2d=source_recon_2d,
             param_lens=param_lens)

    np.savez(os.path.join(output_dir, 'deconv_outputs.npz'),
             image_hr_dc=image_hr_dc,
             image_hr_conv_dc=image_hr_conv_dc,
             image_no_noise_dc=image_no_noise_dc,
             image_real_dc=image_real_dc,
             model_dc=model_dc,
             residuals_dc=residuals_dc,
             source_deconv_2d=source_deconv_2d,
             param_dc=param_dc)

    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(output_dir, 'meta_data.json'), 'w') as f:
        json.dump(p, f, indent=2)

    print(f"Metrics: {metrics}")
    return metrics
