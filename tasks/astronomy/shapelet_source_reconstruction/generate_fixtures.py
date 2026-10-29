"""Generate test fixtures for shapelet_source_reconstruction task."""
import os
import sys
import numpy as np

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TASK_DIR)

from src.physics_model import (
    make_grid, image2array, array2image, re_size,
    shapelet_function, spep_deflection, shear_deflection,
    gaussian_convolve,
)
from src.solvers import build_response_matrix, linear_solve, reduced_residuals


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def main():
    fixture_dir = os.path.join(TASK_DIR, "evaluation", "fixtures")
    ensure_dir(fixture_dir)

    # ── input_resize / param_resize / output_resize ─────────────────────
    np.random.seed(10)
    img_resize = np.random.rand(10, 10)
    factor = 2
    out_resize = re_size(img_resize, factor)

    np.savez(os.path.join(fixture_dir, "input_resize.npz"), image=img_resize)
    np.savez(os.path.join(fixture_dir, "param_resize.npz"), factor=np.array(factor))
    np.savez(os.path.join(fixture_dir, "output_resize.npz"), image=out_resize)
    print("[OK] resize fixtures")

    # ── output_grid_4x4 ────────────────────────────────────────────────
    x_grid, y_grid = make_grid(4, 0.5)
    np.savez(os.path.join(fixture_dir, "output_grid_4x4.npz"), x=x_grid, y=y_grid)
    print("[OK] output_grid_4x4.npz")

    # ── shapelet fixtures ──────────────────────────────────────────────
    np.random.seed(20)
    n_max = 3
    num_param = (n_max + 1) * (n_max + 2) // 2
    amp = np.random.randn(num_param)
    beta = 2.0
    x_sh, y_sh = make_grid(16, 0.3)
    flux = shapelet_function(x_sh, y_sh, amp, n_max, beta)

    np.savez(os.path.join(fixture_dir, "input_shapelet.npz"), x=x_sh, y=y_sh, amp=amp)
    np.savez(os.path.join(fixture_dir, "param_shapelet.npz"),
             n_max=np.array(n_max), beta=np.array(beta))
    np.savez(os.path.join(fixture_dir, "output_shapelet.npz"), flux=flux)
    print("[OK] shapelet fixtures")

    # ── SPEP (SIS) fixtures ────────────────────────────────────────────
    np.random.seed(30)
    x_spep = np.array([1.0, 0.0, 0.6, -0.5, 0.3])
    y_spep = np.array([0.0, 1.0, 0.8, -0.2, 0.7])
    theta_E = 0.5
    gamma_spep = 2.0
    ax_spep, ay_spep = spep_deflection(x_spep, y_spep, theta_E, gamma_spep, 0, 0)

    np.savez(os.path.join(fixture_dir, "input_spep_sis.npz"), x=x_spep, y=y_spep)
    np.savez(os.path.join(fixture_dir, "param_spep_sis.npz"),
             theta_E=np.array(theta_E), gamma=np.array(gamma_spep))
    np.savez(os.path.join(fixture_dir, "output_spep_sis.npz"),
             alpha_x=ax_spep, alpha_y=ay_spep)
    print("[OK] SPEP (SIS) fixtures")

    # ── shear fixtures ─────────────────────────────────────────────────
    x_sh_in = np.array([1.0, 2.0, -0.5, 0.3])
    y_sh_in = np.array([3.0, 4.0, -0.2, 0.7])
    gamma1 = 0.05
    gamma2 = -0.03
    ax_shear, ay_shear = shear_deflection(x_sh_in, y_sh_in, gamma1, gamma2)

    np.savez(os.path.join(fixture_dir, "input_shear.npz"), x=x_sh_in, y=y_sh_in)
    np.savez(os.path.join(fixture_dir, "param_shear.npz"),
             gamma1=np.array(gamma1), gamma2=np.array(gamma2))
    np.savez(os.path.join(fixture_dir, "output_shear.npz"),
             alpha_x=ax_shear, alpha_y=ay_shear)
    print("[OK] shear fixtures")

    # ── gaussian convolution fixtures ──────────────────────────────────
    np.random.seed(40)
    img_conv = np.zeros((32, 32))
    img_conv[12:20, 12:20] = np.random.rand(8, 8)
    fwhm_conv = 0.15
    pix_size_conv = 0.05
    out_conv = gaussian_convolve(img_conv, fwhm_conv, pix_size_conv)

    np.savez(os.path.join(fixture_dir, "input_gaussian_conv.npz"), image=img_conv)
    np.savez(os.path.join(fixture_dir, "param_gaussian_conv.npz"),
             fwhm=np.array(fwhm_conv), pixel_size=np.array(pix_size_conv))
    np.savez(os.path.join(fixture_dir, "output_gaussian_conv.npz"), image=out_conv)
    print("[OK] gaussian convolution fixtures")

    # ── linear_solve fixtures ──────────────────────────────────────────
    np.random.seed(50)
    numPix_ls = 8
    n_max_ls = 2
    num_basis_ls = (n_max_ls + 1) * (n_max_ls + 2) // 2
    A = build_response_matrix(numPix_ls, 0.1, 1, 0.0,
                              n_max_ls, 0.5, 0.0, 0.0, apply_lens=False)
    # Create known parameters and generate data
    true_params = np.random.randn(num_basis_ls) * 0.1
    true_params[0] = 1.0  # dominant zeroth order
    clean_data_1d = A.T.dot(true_params)
    # Add noise
    background_rms = 0.01
    exp_time = 100.0
    noise = np.random.randn(numPix_ls * numPix_ls) * background_rms
    noisy_data_1d = clean_data_1d + noise
    data_noisy_2d = array2image(noisy_data_1d)

    params_solved, model_solved = linear_solve(A, data_noisy_2d, background_rms, exp_time)

    np.savez(os.path.join(fixture_dir, "input_linear_solve.npz"),
             A=A, data_noisy=data_noisy_2d)
    np.savez(os.path.join(fixture_dir, "param_linear_solve.npz"),
             background_rms=np.array(background_rms), exp_time=np.array(exp_time))
    np.savez(os.path.join(fixture_dir, "output_linear_solve.npz"),
             param_solved=params_solved, model_solved=model_solved)
    print("[OK] linear_solve fixtures")

    # ── reduced_residuals fixtures ─────────────────────────────────────
    np.random.seed(60)
    model_rr = np.random.rand(8, 8) * 10 + 1
    data_rr = model_rr + np.random.randn(8, 8) * 0.5
    bg_rms_rr = 0.5
    exp_time_rr = 50.0
    residuals = reduced_residuals(model_rr, data_rr, bg_rms_rr, exp_time_rr)

    np.savez(os.path.join(fixture_dir, "input_reduced_residuals.npz"),
             model=model_rr, data=data_rr)
    np.savez(os.path.join(fixture_dir, "param_reduced_residuals.npz"),
             background_rms=np.array(bg_rms_rr), exp_time=np.array(exp_time_rr))
    np.savez(os.path.join(fixture_dir, "output_reduced_residuals.npz"),
             residuals=residuals)
    print("[OK] reduced_residuals fixtures")

    print("\nAll shapelet_source_reconstruction fixtures generated successfully!")


if __name__ == "__main__":
    main()
