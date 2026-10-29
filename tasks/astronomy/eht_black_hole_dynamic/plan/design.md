# Design: Code Architecture

## Module Layout

```
src/
├── generate_data.py    # Synthetic data generation
├── preprocessing.py    # Data loading and preparation
├── physics_model.py    # Forward model, measurement functions, warping
├── solvers.py          # StarWarps solver + static baseline
└── visualization.py    # Metrics and plotting
```

## Function Signatures

### generate_data.py

```python
make_ring_frame(N=30, ring_radius_frac=0.22, ring_width_frac=0.055,
                asymmetry=0.5, asymmetry_angle_deg=220.0) -> (N,N) ndarray

make_rotating_video(N=30, n_frames=12, rotation_deg=90.0,
                    total_flux=2.0, base_angle_deg=220.0)
    -> (list[(N,N)], (n_frames,) angles)

simulate_eht_uv_coverage_at_time(source_ra_deg, source_dec_deg,
                                  time_utc, freq_ghz=230.0)
    -> (uv_coords, station_ids, sefds)

compute_sefd_noise(station_ids, sefds, bandwidth_hz, tau_int, eta)
    -> (M,) noise_std

generate_dataset(save_dir='data', ...) -> dict
```

### preprocessing.py

```python
load_observation(data_dir) -> dict  # per-frame vis, sigma, uv, station_ids
load_metadata(data_dir) -> dict     # N, psize, n_frames, etc.
load_ground_truth(task_dir) -> (n_frames, N, N) ndarray
build_per_frame_models(obs, meta) -> list[DFTForwardModel]
prepare_data(data_dir, task_dir) -> (obs, meta, gt)
```

### physics_model.py

```python
class DFTForwardModel:
    __init__(uv_coords, N, pixel_size_rad)
    forward(image_vec) -> vis_vec
    adjoint(vis_vec) -> image_vec
    matrix -> (M, N²) ndarray

# Measurement functions
compute_visibilities(imvec, A) -> complex array
compute_bispectrum(imvec, A_matrices) -> complex array
compute_closure_phase(imvec, A_matrices) -> complex array
compute_visibility_amplitude(imvec, A) -> real array
compute_log_closure_amplitude(imvec, A_matrices) -> real array

# Gradients (Jacobians)
grad_vis(imvec, A) -> matrix
grad_bispectrum(imvec, A_matrices) -> matrix
grad_closure_phase(imvec, A_matrices) -> matrix
grad_visibility_amplitude(imvec, A) -> matrix
grad_log_closure_amplitude(imvec, A_matrices) -> matrix

# Covariance
gauss_image_covariance(N, psize, imvec, power_dropoff, frac) -> (N², N²)

# Motion basis
affine_motion_basis(N, psize) -> (init_x, init_y, fb_x, fb_y, initTheta)
apply_motion_basis(init_x, init_y, fb_x, fb_y, theta) -> (flow_x, flow_y)

# Warping
calc_warp_matrix(N, psize, theta, ..., method='phase') -> (N², N²)
gen_freq_comp(N, psize) -> (ufull, vfull)
gen_phase_shift_matrix(ulist, vlist, ..., theta, psize) -> (N², N²)

# Gaussian algebra
product_gaussians_lem1(m1, S1, m2, S2) -> (mean, cov)
product_gaussians_lem2(A, Sigma, y, mu, Q) -> (mean, cov)
evaluate_gaussian_log(y, x, Sigma) -> scalar

# Measurement assembly
get_measurement_terms(dft_model, imvec, vis, sigma, measurement, ...)
    -> (meas_diff, ideal, F, Cov, valid)

# EM M-step
exp_neg_loglikelihood(theta, ...) -> scalar
deriv_exp_neg_loglikelihood(theta, ...) -> gradient
```

### solvers.py

```python
class StaticPerFrameSolver:
    __init__(prior_mean, prior_cov)
    reconstruct(models, obs, N, measurement, num_lin_iters)
        -> list[(N,N)]

class StarWarpsSolver:
    __init__(prior_mean, prior_cov, process_noise_cov, N, psize,
             warp_method, measurement, n_em_iters, num_lin_iters,
             interior_priors)
    reconstruct(models, obs, init_images=None)
        -> dict{frames, uncertainties, theta, log_likelihood}

# Internal functions
forward_updates(...) -> (loglike, z_pred, P_pred, z_upd, P_upd, z_lin)
backward_updates(...) -> (z_t_t, P_t_t)
smoothing_updates(...) -> (z, P, backwards_A)
compute_sufficient_statistics(...)
    -> (expVal_t, expVal_t_t, expVal_tm1_t, loglike, apx_imgs)
```

### visualization.py

```python
compute_metrics(est, ref) -> dict{nrmse, ncc}
compute_video_metrics(est_frames, ref_frames)
    -> dict{per_frame, average}
print_metrics_table(metrics_dict)
plot_video_comparison(gt, recon_dict, frame_times, save_path)
plot_metrics_over_time(metrics_dict, frame_times, save_path)
```
