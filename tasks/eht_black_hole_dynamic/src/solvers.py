"""
StarWarps solver and static baseline for dynamic black hole imaging.

Extracted from ehtim's starwarps.py (Bouman et al. 2017, arXiv:1711.01357).
No ehtim dependency — operates on plain numpy arrays.

Classes:
    StaticPerFrameSolver: reconstruct each frame independently (Gaussian MAP)
    StarWarpsSolver: full StarWarps EM with forward-backward message passing
"""

import copy
import sys

import numpy as np
import scipy.optimize

from src.physics_model import (
    DFTForwardModel,
    get_measurement_terms,
    product_gaussians_lem1,
    product_gaussians_lem2,
    evaluate_gaussian_log,
    realimag_stack,
    calc_warp_matrix,
    exp_neg_loglikelihood,
    deriv_exp_neg_loglikelihood,
    affine_motion_basis,
    affine_motion_basis_no_translation,
    gauss_image_covariance,
)


# ============================================================================
# Single-image solver (starwarps.py lines 31-56)
# ============================================================================

def solve_single_image(mu_vec, Lambda, dft_model, vis_data, sigma_data,
                       measurement='vis', num_lin_iters=5):
    """Solve for a single image using Gaussian MAP with linearized iterations.

    Args:
        mu_vec: (N²,) prior mean image vector
        Lambda: (N², N²) prior covariance
        dft_model: DFTForwardModel instance
        vis_data: observed visibilities (complex)
        sigma_data: noise standard deviations
        measurement: data term type ('vis', 'amp', etc.)
        num_lin_iters: number of linearized Gauss-Newton iterations

    Returns:
        (z_vec, P, z_lin): estimated image, covariance, linearization point
    """
    if measurement == 'vis':
        num_lin_iters = 1

    z_vec = mu_vec.copy()
    z_lin = mu_vec.copy()

    for k in range(num_lin_iters):
        meas, idealmeas, F, measCov, valid = get_measurement_terms(
            dft_model, z_lin, vis_data, sigma_data,
            measurement=measurement)
        if valid:
            z_vec, P = product_gaussians_lem2(F, measCov, meas,
                                              mu_vec, Lambda)
            if k < num_lin_iters - 1:
                z_lin = z_vec.copy()
        else:
            z_vec = mu_vec.copy()
            P = Lambda.copy()

    return (z_vec, P, z_lin)


# ============================================================================
# Forward pass (starwarps.py lines 61-226)
# ============================================================================

def forward_updates(mu_list, Lambda_list, models, obs,
                    A_warp, Q, init_images=None,
                    measurement='vis', num_lin_iters=5,
                    interior_priors=False):
    """Forward Kalman-like pass through time series.

    Args:
        mu_list: list of (N²,) prior mean vectors (one per frame or single)
        Lambda_list: list of (N², N²) prior covariances
        models: list of DFTForwardModel (one per frame)
        obs: observation dict with 'vis', 'sigma' lists
        A_warp: (N², N²) state transition (warp) matrix
        Q: (N², N²) process noise covariance
        init_images: optional list of (N²,) initial image vectors
        measurement: data term type
        num_lin_iters: linearization iterations
        interior_priors: whether to use interior priors (Lem1 fusion)

    Returns:
        (loglikelihood, z_t_tm1, P_t_tm1, z_t_t, P_t_t, z_lin)
    """
    if measurement == 'vis':
        num_lin_iters = 1

    n_frames = len(models)
    npixels = len(mu_list[0])

    # Initialize lists
    z_t_t = [np.zeros(npixels) for _ in range(n_frames)]
    P_t_t = [np.zeros((npixels, npixels)) for _ in range(n_frames)]

    z_t_tm1 = [z.copy() for z in z_t_t]
    P_t_tm1 = [P.copy() for P in P_t_t]

    z_star_t_tm1 = [z.copy() for z in z_t_t]
    P_star_t_tm1 = [P.copy() for P in P_t_t]

    z_lin = [z.copy() for z in z_t_t]

    loglikelihood_prior = 0.0
    loglikelihood_data = 0.0

    for t in range(n_frames):
        # Select prior for this frame
        mu_t = mu_list[t] if len(mu_list) > 1 else mu_list[0]
        Lambda_t = Lambda_list[t] if len(Lambda_list) > 1 else Lambda_list[0]

        # Predict step
        if t == 0:
            z_star_t_tm1[t] = mu_t.copy()
            P_star_t_tm1[t] = Lambda_t.copy()
        else:
            z_t_tm1[t] = np.dot(A_warp, z_t_t[t - 1])
            P_t_tm1[t] = Q + np.dot(A_warp, np.dot(P_t_t[t - 1], A_warp.T))

            if interior_priors:
                z_star_t_tm1[t], P_star_t_tm1[t] = product_gaussians_lem1(
                    mu_t, Lambda_t, z_t_tm1[t], P_t_tm1[t])
            else:
                z_star_t_tm1[t] = z_t_tm1[t].copy()
                P_star_t_tm1[t] = P_t_tm1[t].copy()

        # Initialize linearization point
        if init_images is None:
            z_lin[t] = z_star_t_tm1[t].copy()
        elif len(init_images) == 1:
            z_lin[t] = init_images[0].copy()
        else:
            z_lin[t] = init_images[t].copy()

        # Update step with linearized iterations
        for k in range(num_lin_iters):
            meas, idealmeas, F, measCov, valid = get_measurement_terms(
                models[t], z_lin[t], obs['vis'][t], obs['sigma'][t],
                measurement=measurement)

            if valid:
                z_t_t[t], P_t_t[t] = product_gaussians_lem2(
                    F, measCov, meas,
                    z_star_t_tm1[t], P_star_t_tm1[t])
                if k < num_lin_iters - 1:
                    z_lin[t] = z_t_t[t].copy()
            else:
                z_t_t[t] = z_star_t_tm1[t].copy()
                P_t_t[t] = P_star_t_tm1[t].copy()

        # Update log likelihoods
        if t > 0 and interior_priors:
            loglikelihood_prior += evaluate_gaussian_log(
                z_t_tm1[t], mu_t, Lambda_t + P_t_tm1[t])

        if valid:
            loglikelihood_data += evaluate_gaussian_log(
                np.dot(F, z_star_t_tm1[t]), meas,
                measCov + np.dot(F, np.dot(P_star_t_tm1[t], F.T)))

    loglikelihood = (loglikelihood_data, loglikelihood_prior,
                     loglikelihood_data + loglikelihood_prior)

    return (loglikelihood, z_t_tm1, P_t_tm1, z_t_t, P_t_t, z_lin)


# ============================================================================
# Backward pass (starwarps.py lines 230-323)
# ============================================================================

def backward_updates(mu_list, Lambda_list, models, obs,
                     A_warp, Q, apx_imgs,
                     measurement='vis'):
    """Backward message-passing through time series.

    Args:
        mu_list: list of (N²,) prior mean vectors
        Lambda_list: list of (N², N²) prior covariances
        models: list of DFTForwardModel (one per frame)
        obs: observation dict with 'vis', 'sigma' lists
        A_warp: (N², N²) state transition matrix
        Q: (N², N²) process noise covariance
        apx_imgs: list of (N²,) linearization images from forward pass
        measurement: data term type

    Returns:
        (z_t_t, P_t_t): backward-updated state estimates
    """
    n_frames = len(models)
    npixels = len(mu_list[0])

    z_t_t = [np.zeros(npixels) for _ in range(n_frames)]
    P_t_t = [np.zeros((npixels, npixels)) for _ in range(n_frames)]

    z_star_t_tp1 = [z.copy() for z in z_t_t]
    P_star_t_tp1 = [P.copy() for P in P_t_t]

    last_idx = n_frames - 1
    for t in range(last_idx, -1, -1):
        mu_t = mu_list[t] if len(mu_list) > 1 else mu_list[0]
        Lambda_t = Lambda_list[t] if len(Lambda_list) > 1 else Lambda_list[0]

        # Predict
        if t == last_idx:
            z_star_t_tp1[t] = mu_t.copy()
            P_star_t_tp1[t] = Lambda_t.copy()
        else:
            z_star_t_tp1[t], P_star_t_tp1[t] = product_gaussians_lem2(
                A_warp, Q + P_t_t[t + 1],
                z_t_t[t + 1], mu_t, Lambda_t)

        # Update
        meas, idealmeas, F, measCov, valid = get_measurement_terms(
            models[t], apx_imgs[t], obs['vis'][t], obs['sigma'][t],
            measurement=measurement)

        if valid:
            z_t_t[t], P_t_t[t] = product_gaussians_lem2(
                F, measCov, meas,
                z_star_t_tp1[t], P_star_t_tp1[t])
        else:
            z_t_t[t] = z_star_t_tp1[t].copy()
            P_t_t[t] = P_star_t_tp1[t].copy()

    return (z_t_t, P_t_t)


# ============================================================================
# RTS Smoothing (starwarps.py lines 326-345)
# ============================================================================

def smoothing_updates(z_t_t, P_t_t, z_t_tm1, P_t_tm1, A_warp):
    """Rauch-Tung-Striebel smoother.

    Args:
        z_t_t: forward filtered means
        P_t_t: forward filtered covariances
        z_t_tm1: forward predicted means
        P_t_tm1: forward predicted covariances
        A_warp: state transition matrix

    Returns:
        (z, P, backwards_A): smoothed means, covariances, smoother gains
    """
    z = [x.copy() for x in z_t_t]
    P = [x.copy() for x in P_t_t]
    backwards_A = [x.copy() for x in P_t_t]

    last_idx = len(z) - 1
    for t in range(last_idx, -1, -1):
        if t < last_idx:
            backwards_A[t] = np.dot(
                np.dot(P_t_t[t], A_warp.T),
                np.linalg.inv(P_t_tm1[t + 1]))
            z[t] = z_t_t[t] + np.dot(
                backwards_A[t],
                z[t + 1] - z_t_tm1[t + 1])
            P[t] = np.dot(
                np.dot(backwards_A[t], P[t + 1] - P_t_tm1[t + 1]),
                backwards_A[t].T) + P_t_t[t]

    return (z, P, backwards_A)


# ============================================================================
# Joint distribution (starwarps.py lines 456-493)
# ============================================================================

def joint_distribution(z, z_fwd, P_fwd, z_bwd, P_bwd, A_warp, Q):
    """Compute E[x_{t-1} x_t^T] using interior priors method.

    See StarWarps supplementary section 2.2.
    """
    expVal_tm1_t = [0.0]

    for t in range(1, len(z_fwd)):
        Sigma = Q + P_bwd[t]
        Sigma_inv = np.linalg.inv(Sigma)

        M = np.dot(P_bwd[t], np.dot(Sigma_inv, A_warp))
        (m, C) = product_gaussians_lem2(
            A_warp, Sigma, z_bwd[t], z_fwd[t - 1], P_fwd[t - 1])

        D_tmp1 = np.dot(M, np.dot(C, M.T))
        D_tmp2 = np.dot(Q, np.dot(Sigma_inv, P_bwd[t]))
        D = np.dot(C, np.dot(M.T, np.linalg.inv(D_tmp1 + D_tmp2)))

        F = C - np.dot(D, np.dot(M, C))

        z_t_hvec = np.array([z[t]])
        z_tm1_hvec = np.array([z[t - 1]])

        expVal_tm1_t.append(
            np.dot(F, np.linalg.inv(D.T)) +
            np.dot(z_tm1_hvec.T, z_t_hvec))

    return expVal_tm1_t


# ============================================================================
# E-step: compute sufficient statistics (starwarps.py lines 349-451)
# ============================================================================

def compute_sufficient_statistics(mu_list, Lambda_list, models, obs,
                                  Upsilon, theta,
                                  init_x, init_y,
                                  flowbasis_x, flowbasis_y, init_theta,
                                  N, psize,
                                  init_images=None,
                                  method='phase',
                                  measurement='vis',
                                  num_lin_iters=1,
                                  interior_priors=False):
    """Main E-step: forward-backward message passing + sufficient statistics.

    Args:
        mu_list: list of (N²,) prior mean vectors
        Lambda_list: list of (N², N²) prior covariances
        models: list of DFTForwardModel
        obs: observation dict with 'vis', 'sigma' lists
        Upsilon: (N², N²) process noise covariance Q
        theta: (nbasis,) current warp parameters
        init_x, init_y: motion basis position arrays
        flowbasis_x, flowbasis_y: motion basis flow arrays
        init_theta: identity warp parameters
        N: image dimension
        psize: pixel size in radians
        init_images: optional initialization images
        method: 'phase' for Fourier warping
        measurement: data term type
        num_lin_iters: linearization iterations
        interior_priors: whether to use interior priors

    Returns:
        (expVal_t, expVal_t_t, expVal_tm1_t, loglikelihood, apx_imgs)
    """
    if measurement == 'vis':
        num_lin_iters = 1

    # Build warp matrix from theta
    warpMtx = calc_warp_matrix(N, psize, theta, init_x, init_y,
                               flowbasis_x, flowbasis_y, init_theta,
                               method=method)
    A_warp = warpMtx
    Q = Upsilon

    # Forward pass
    loglikelihood, z_t_tm1, P_t_tm1, z_t_t, P_t_t, apx_imgs = \
        forward_updates(mu_list, Lambda_list, models, obs,
                        A_warp, Q, init_images=init_images,
                        measurement=measurement,
                        num_lin_iters=num_lin_iters,
                        interior_priors=interior_priors)

    # Backward pass or smoothing
    if interior_priors:
        z_bwd, P_bwd = backward_updates(
            mu_list, Lambda_list, models, obs,
            A_warp, Q, apx_imgs,
            measurement=measurement)

        z = [None] * len(models)
        P = [None] * len(models)
        for t in range(len(models)):
            if t == 0:
                z[t] = z_bwd[t].copy()
                P[t] = P_bwd[t].copy()
            else:
                z[t], P[t] = product_gaussians_lem1(
                    z_t_tm1[t], P_t_tm1[t],
                    z_bwd[t], P_bwd[t])
    else:
        z, P, backwards_A = smoothing_updates(
            z_t_t, P_t_t, z_t_tm1, P_t_tm1, A_warp)

    # Compute sufficient statistics
    expVal_t = [x.copy() for x in z]
    expVal_t_t = [None] * len(models)
    expVal_tm1_t = [None] * len(models)

    for t in range(len(models)):
        z_hvec = np.array([z[t]])
        expVal_t_t[t] = np.dot(z_hvec.T, z_hvec) + P[t]

        if t > 0 and not interior_priors:
            z_tm1_hvec = np.array([z[t - 1]])
            expVal_tm1_t[t] = (np.dot(z_tm1_hvec.T, z_hvec) +
                               np.dot(backwards_A[t - 1], P[t]))

    if interior_priors:
        expVal_tm1_t = joint_distribution(
            z, z_t_t, P_t_t, z_bwd, P_bwd, A_warp, Q)

    return (expVal_t, expVal_t_t, expVal_tm1_t, loglikelihood, apx_imgs)


# ============================================================================
# Static per-frame solver
# ============================================================================

class StaticPerFrameSolver:
    """Baseline: reconstruct each frame independently using Gaussian MAP.

    No temporal coupling — each frame is solved as a separate static problem.
    """

    def __init__(self, prior_mean, prior_cov):
        """
        Args:
            prior_mean: (N²,) prior mean image vector
            prior_cov: (N², N²) prior covariance matrix
        """
        self.prior_mean = prior_mean
        self.prior_cov = prior_cov

    def reconstruct(self, models, obs, N, measurement='vis',
                    num_lin_iters=5):
        """Reconstruct each frame independently.

        Args:
            models: list of DFTForwardModel (one per frame)
            obs: observation dict with 'vis', 'sigma' lists
            N: image dimension
            measurement: data term type
            num_lin_iters: linearization iterations

        Returns:
            list of (N, N) reconstructed frames
        """
        n_frames = len(models)
        frames = []

        for t in range(n_frames):
            z_vec, P, z_lin = solve_single_image(
                self.prior_mean, self.prior_cov,
                models[t], obs['vis'][t], obs['sigma'][t],
                measurement=measurement,
                num_lin_iters=num_lin_iters)
            # Clip negative pixels
            z_vec = np.maximum(z_vec, 0.0)
            frames.append(z_vec.reshape(N, N))

        return frames


# ============================================================================
# StarWarps solver
# ============================================================================

class StarWarpsSolver:
    """Full StarWarps EM algorithm with forward-backward message passing.

    Implements the algorithm from Bouman et al. 2017 (arXiv:1711.01357).
    Uses Gaussian Markov model with EM to jointly estimate video frames
    and inter-frame motion parameters.
    """

    def __init__(self, prior_mean, prior_cov, process_noise_cov,
                 N, psize,
                 warp_method='phase',
                 measurement='vis',
                 n_em_iters=30,
                 num_lin_iters=5,
                 interior_priors=True,
                 motion_basis='affine_no_translation',
                 m_step_maxiter=4000):
        """
        Args:
            prior_mean: (N²,) prior mean image vector
            prior_cov: (N², N²) prior covariance matrix
            process_noise_cov: (N², N²) process noise Q (Upsilon)
            N: image dimension
            psize: pixel size in radians
            warp_method: 'phase' for Fourier-domain warping
            measurement: data term ('vis', 'amp', 'cphase', etc.)
            n_em_iters: number of EM iterations
            num_lin_iters: linearization iterations per forward step
            interior_priors: whether to use interior priors
            motion_basis: 'affine', 'affine_no_translation', or 'translation'
            m_step_maxiter: max iterations for L-BFGS-B in M-step
        """
        self.prior_mean = prior_mean
        self.prior_cov = prior_cov
        self.process_noise_cov = process_noise_cov
        self.N = N
        self.psize = psize
        self.warp_method = warp_method
        self.measurement = measurement
        self.n_em_iters = n_em_iters
        self.num_lin_iters = num_lin_iters
        self.interior_priors = interior_priors
        self.m_step_maxiter = m_step_maxiter

        # Initialize motion basis
        if motion_basis == 'affine_no_translation':
            self.init_x, self.init_y, self.flowbasis_x, self.flowbasis_y, \
                self.init_theta = affine_motion_basis_no_translation(N, psize)
        else:
            self.init_x, self.init_y, self.flowbasis_x, self.flowbasis_y, \
                self.init_theta = affine_motion_basis(N, psize)

    def reconstruct(self, models, obs, init_images=None):
        """Run full StarWarps EM reconstruction.

        Args:
            models: list of DFTForwardModel (one per frame)
            obs: observation dict with 'vis', 'sigma' lists
            init_images: optional list of (N²,) initial images

        Returns:
            dict with:
                'frames': list of (N, N) reconstructed frames
                'uncertainties': list of (N,) per-pixel std
                'theta': final motion parameters
                'log_likelihood': list of log-likelihood values
        """
        n_frames = len(models)

        # Use single prior for all frames
        mu_list = [self.prior_mean]
        Lambda_list = [self.prior_cov]

        # Initial theta = identity
        theta = self.init_theta.copy().astype(np.float64)

        log_likelihoods = []

        for em_iter in range(self.n_em_iters):
            sys.stdout.write(
                f'\rStarWarps EM iteration {em_iter + 1}/{self.n_em_iters}')
            sys.stdout.flush()

            # E-step: compute sufficient statistics
            expVal_t, expVal_t_t, expVal_tm1_t, loglike, apx_imgs = \
                compute_sufficient_statistics(
                    mu_list, Lambda_list, models, obs,
                    self.process_noise_cov, theta,
                    self.init_x, self.init_y,
                    self.flowbasis_x, self.flowbasis_y,
                    self.init_theta,
                    self.N, self.psize,
                    init_images=init_images,
                    method=self.warp_method,
                    measurement=self.measurement,
                    num_lin_iters=self.num_lin_iters,
                    interior_priors=self.interior_priors)

            log_likelihoods.append(loglike[2])

            # After first iteration, use E-step results as initialization
            init_images = [x.copy() for x in expVal_t]

            # M-step: optimize theta (warp parameters)
            nbasis = len(theta)
            bnds = [(-1.5, 1.5)] * nbasis

            def neg_ll(th):
                return exp_neg_loglikelihood(
                    th, expVal_t, expVal_t_t, expVal_tm1_t,
                    self.N, self.psize, self.process_noise_cov,
                    self.init_x, self.init_y,
                    self.flowbasis_x, self.flowbasis_y,
                    self.init_theta, method=self.warp_method)

            def grad_neg_ll(th):
                return deriv_exp_neg_loglikelihood(
                    th, expVal_t, expVal_t_t, expVal_tm1_t,
                    self.N, self.psize, self.process_noise_cov,
                    self.init_x, self.init_y,
                    self.flowbasis_x, self.flowbasis_y,
                    self.init_theta, method=self.warp_method)

            result = scipy.optimize.minimize(
                neg_ll, theta, jac=grad_neg_ll,
                method='L-BFGS-B',
                bounds=bnds,
                options={'maxiter': self.m_step_maxiter,
                         'ftol': 1e-10,
                         'maxcor': 5000,
                         'disp': False})
            theta = result.x

        print()  # newline after progress

        # Final E-step to get best estimates
        expVal_t, expVal_t_t, expVal_tm1_t, loglike, _ = \
            compute_sufficient_statistics(
                mu_list, Lambda_list, models, obs,
                self.process_noise_cov, theta,
                self.init_x, self.init_y,
                self.flowbasis_x, self.flowbasis_y,
                self.init_theta,
                self.N, self.psize,
                init_images=init_images,
                method=self.warp_method,
                measurement=self.measurement,
                num_lin_iters=self.num_lin_iters,
                interior_priors=self.interior_priors)

        # Extract frames
        frames = []
        uncertainties = []
        for t in range(n_frames):
            img_vec = np.maximum(expVal_t[t], 0.0)
            frames.append(img_vec.reshape(self.N, self.N))
            # Uncertainty: sqrt of diagonal of covariance
            var_diag = np.diag(expVal_t_t[t] -
                               np.outer(expVal_t[t], expVal_t[t]))
            uncertainties.append(np.sqrt(np.maximum(var_diag, 0.0)))

        return {
            'frames': frames,
            'uncertainties': uncertainties,
            'theta': theta,
            'log_likelihood': log_likelihoods,
        }
