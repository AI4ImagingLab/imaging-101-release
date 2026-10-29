"""
Linear inversion solvers for shapelet source reconstruction and deconvolution.
"""

import numpy as np
from .physics_model import (
    make_grid, image2array, array2image, re_size,
    shapelet_basis_list, ray_shoot, gaussian_convolve,
)


def build_response_matrix(numPix, deltaPix, supersampling_factor, fwhm,
                          n_max_recon, beta_recon, center_x, center_y,
                          kwargs_spemd=None, kwargs_shear=None, apply_lens=True):
    """Build design matrix A where each row is one shapelet basis function
    evaluated on source plane, convolved with PSF, and downsampled.

    :param numPix: detector pixel count per axis
    :param deltaPix: detector pixel size (arcsec)
    :param supersampling_factor: sub-pixel oversampling factor
    :param fwhm: PSF FWHM (arcsec)
    :param n_max_recon: maximum shapelet order for reconstruction
    :param beta_recon: shapelet scale for reconstruction
    :param center_x: shapelet center x
    :param center_y: shapelet center y
    :param kwargs_spemd: SPEP lens parameters (None to skip)
    :param kwargs_shear: shear parameters (None to skip)
    :param apply_lens: whether to apply ray-tracing
    :return: A array of shape (num_basis, numPix^2)
    """
    n_super = numPix * supersampling_factor
    dpix_super = deltaPix / supersampling_factor

    x_grid, y_grid = make_grid(n_super, dpix_super)

    if apply_lens and kwargs_spemd is not None:
        x_eval, y_eval = ray_shoot(x_grid, y_grid, kwargs_spemd, kwargs_shear)
    else:
        x_eval, y_eval = x_grid, y_grid

    basis = shapelet_basis_list(x_eval, y_eval, n_max_recon, beta_recon, center_x, center_y)
    num_basis = len(basis)
    num_pix = numPix * numPix
    A = np.zeros((num_basis, num_pix))

    for i, b1d in enumerate(basis):
        b2d = array2image(b1d)
        if fwhm > 0:
            b2d = gaussian_convolve(b2d, fwhm, dpix_super, truncation=5)
        if supersampling_factor > 1:
            b2d = re_size(b2d, supersampling_factor)
        b2d *= deltaPix**2
        A[i, :] = image2array(b2d)

    return A


def linear_solve(A, data_2d, background_rms, exp_time):
    """Solve for shapelet coefficients via Weighted Least Squares.

    Minimizes: sum_i w_i * (d_i - [A^T c]_i)^2
    where w_i = 1 / (sigma_bkg^2 + |d_i| / t_exp).

    :param A: response matrix (num_basis, num_pix)
    :param data_2d: observed image (numPix, numPix)
    :param background_rms: background noise sigma
    :param exp_time: exposure time
    :return: (params, model_2d) where params are coefficients, model_2d is reconstructed image
    """
    d = image2array(data_2d)
    C_D = background_rms**2 + np.abs(d) / exp_time
    w = 1.0 / C_D

    Aw = A * w[None, :]
    M = Aw.dot(A.T)
    R = Aw.dot(d)

    cond = np.linalg.cond(M)
    if cond < 1e14:
        params = np.linalg.solve(M, R)
    else:
        print(f"  Warning: ill-conditioned matrix (cond={cond:.2e}), using lstsq")
        params, _, _, _ = np.linalg.lstsq(M, R, rcond=None)

    model_1d = A.T.dot(params)
    return params, array2image(model_1d)


def reduced_residuals(model, data, background_rms, exp_time):
    """Compute reduced residuals: (model - data) / sqrt(C_D).

    C_D includes background noise and Poisson noise from the model flux.

    :param model: model image (2D)
    :param data: observed image (2D)
    :param background_rms: background noise sigma
    :param exp_time: exposure time
    :return: 2D array of reduced residuals
    """
    C_D = background_rms**2 + np.abs(model) / exp_time
    return (model - data) / np.sqrt(C_D)
