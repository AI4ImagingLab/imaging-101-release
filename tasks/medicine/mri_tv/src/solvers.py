"""
Total Variation MRI Reconstruction Solver
==========================================

Solves the multi-coil compressed sensing MRI problem via PDHG
(Primal-Dual Hybrid Gradient, aka Chambolle-Pock):

    argmin_x  (1/2) ||P F S x - y||_2^2  +  lambda * TV(x)

where:
    x     : complex image (H, W)
    S     : coil sensitivity maps
    F     : 2D centered FFT (ortho-normalized)
    P     : binary undersampling mask (as weights)
    TV(x) : isotropic total variation = sum |nabla x|

Algorithm ported from SigPy's TotalVariationRecon (sigpy/mri/app.py),
which uses the PDHG algorithm (sigpy/alg.py) with:
    - Stacked operator A = [Sense; FiniteDifference]
    - Dual proximal = Stack([L2Reg (data fidelity), Conj(L1Reg) (TV)])
    - Step sizes via power iteration

Reference
---------
Chambolle, A. & Pock, T. (2011). A first-order primal-dual algorithm for
convex problems with applications to imaging. JMIV, 40(1), 120-145.
SigPy: https://github.com/mikgroup/sigpy
"""

import numpy as np
from src.physics_model import fft2c, ifft2c


# ---------------------------------------------------------------------------
# SENSE forward / adjoint operators
# ---------------------------------------------------------------------------

def sense_forward(image, sensitivity_maps, mask_weights):
    """SENSE forward model: image -> weighted multi-coil k-space.

    Computes: y = P^{1/2} F S x

    Parameters
    ----------
    image : ndarray, (H, W) complex
    sensitivity_maps : ndarray, (C, H, W) complex
    mask_weights : ndarray, (C, H, W) or (1, H, W) float
        Square root of sampling mask, broadcast over coils.

    Returns
    -------
    y : ndarray, (C, H, W) complex
    """
    coil_images = sensitivity_maps * image[None, :, :]
    kspace = fft2c(coil_images)
    return kspace * mask_weights


def sense_adjoint(y, sensitivity_maps, mask_weights):
    """SENSE adjoint: weighted multi-coil k-space -> image.

    Computes: x = S^H F^H P^{1/2} y = sum_c conj(S_c) * IFFT(P^{1/2} * y_c)

    Parameters
    ----------
    y : ndarray, (C, H, W) complex
    sensitivity_maps : ndarray, (C, H, W) complex
    mask_weights : ndarray, (C, H, W) or (1, H, W) float

    Returns
    -------
    image : ndarray, (H, W) complex
    """
    coil_images = ifft2c(y * mask_weights)
    return np.sum(coil_images * np.conj(sensitivity_maps), axis=0)


# ---------------------------------------------------------------------------
# Finite difference operator (TV gradient)
# ---------------------------------------------------------------------------

def finite_difference(x):
    """Finite difference operator for total variation.

    Computes circular forward differences along each spatial axis.
    Ported from SigPy's linop.FiniteDifference.

    Parameters
    ----------
    x : ndarray, (H, W) complex

    Returns
    -------
    grad : ndarray, (2, H, W) complex
        grad[0] = x - roll(x, 1, axis=0)   (vertical differences)
        grad[1] = x - roll(x, 1, axis=1)   (horizontal differences)
    """
    dy = x - np.roll(x, 1, axis=0)
    dx = x - np.roll(x, 1, axis=1)
    return np.stack([dy, dx], axis=0)


def finite_difference_adjoint(grad):
    """Adjoint of the finite difference operator.

    Negative divergence: G^H = -div.

    Parameters
    ----------
    grad : ndarray, (2, H, W) complex

    Returns
    -------
    x : ndarray, (H, W) complex
    """
    dy = grad[0] - np.roll(grad[0], -1, axis=0)
    dx = grad[1] - np.roll(grad[1], -1, axis=1)
    return dy + dx


# ---------------------------------------------------------------------------
# Stacked forward / adjoint (SENSE + FiniteDifference)
# ---------------------------------------------------------------------------

def stacked_forward(x, sensitivity_maps, mask_weights):
    """Stacked operator A = [Sense; G] applied to x.

    Returns
    -------
    (y_sense, y_grad) : tuple of ndarrays
        y_sense: (C, H, W) complex — SENSE k-space
        y_grad: (2, H, W) complex — finite differences
    """
    y_sense = sense_forward(x, sensitivity_maps, mask_weights)
    y_grad = finite_difference(x)
    return y_sense, y_grad


def stacked_adjoint(u_sense, u_grad, sensitivity_maps, mask_weights):
    """Adjoint of stacked operator: A^H [u1; u2] = Sense^H u1 + G^H u2.

    Returns
    -------
    x : ndarray, (H, W) complex
    """
    x1 = sense_adjoint(u_sense, sensitivity_maps, mask_weights)
    x2 = finite_difference_adjoint(u_grad)
    return x1 + x2


# ---------------------------------------------------------------------------
# Proximal operators
# ---------------------------------------------------------------------------

def soft_thresh(lamda, x):
    """Soft thresholding (proximal of L1 norm).

    prox_{lamda * ||.||_1}(x) = sign(x) * max(|x| - lamda, 0)

    Ported from SigPy's thresh.soft_thresh.

    Parameters
    ----------
    lamda : float
        Threshold value.
    x : ndarray, complex

    Returns
    -------
    ndarray, same shape as x
    """
    abs_x = np.abs(x)
    sign = np.where(abs_x > 0, x / np.maximum(abs_x, 1e-30), 0)
    mag = np.maximum(abs_x - lamda, 0)
    return mag * sign


def prox_l2_reg(sigma, u, y):
    """Proximal operator for data fidelity: prox_{sigma * (1/2)||. - y||^2}(u).

    Solution: (u + sigma * y) / (1 + sigma)

    Ported from SigPy's prox.L2Reg.

    Parameters
    ----------
    sigma : float
        Step size.
    u : ndarray, complex
    y : ndarray, complex
        Measured data.

    Returns
    -------
    ndarray
    """
    return (u + sigma * y) / (1 + sigma)


def prox_l1_conj(sigma, u, lamda):
    """Proximal of conjugate of L1 norm (Moreau decomposition).

    prox_{sigma * (lamda * ||.||_1)^*}(u) = u - sigma * soft_thresh(lamda / sigma, u / sigma)

    This is the dual proximal for the TV regularization term.
    Ported from SigPy's prox.Conj wrapping prox.L1Reg.

    Parameters
    ----------
    sigma : float
        Dual step size.
    u : ndarray, complex
        Dual variable (gradient domain).
    lamda : float
        TV regularization weight.

    Returns
    -------
    ndarray
    """
    return u - sigma * soft_thresh(lamda / sigma, u / sigma)


# ---------------------------------------------------------------------------
# Step size estimation via power iteration
# ---------------------------------------------------------------------------

def estimate_max_eigenvalue(sensitivity_maps, mask_weights, img_shape,
                            sigma=1.0, max_iter=30):
    """Estimate max eigenvalue of A^H diag(sigma) A via power iteration.

    A is the stacked operator [Sense; G]. This determines the primal
    step size tau = 1 / max_eig.

    Ported from SigPy's app.MaxEig.

    Parameters
    ----------
    sensitivity_maps : ndarray, (C, H, W) complex
    mask_weights : ndarray, broadcastable to (C, H, W)
    img_shape : tuple, (H, W)
    sigma : float
        Dual step size.
    max_iter : int
        Number of power iterations.

    Returns
    -------
    max_eig : float
    """
    rng = np.random.default_rng(0)
    x = rng.standard_normal(img_shape) + 1j * rng.standard_normal(img_shape)
    x = x.astype(np.complex128)

    for _ in range(max_iter):
        # Apply A^H @ diag(sigma) @ A @ x
        y_sense, y_grad = stacked_forward(x, sensitivity_maps, mask_weights)
        y_sense *= sigma
        y_grad *= sigma
        x_new = stacked_adjoint(y_sense, y_grad, sensitivity_maps, mask_weights)

        # Power iteration: normalize
        norm = np.linalg.norm(x_new)
        if norm < 1e-30:
            break
        x = x_new / norm

    # Rayleigh quotient
    y_sense, y_grad = stacked_forward(x, sensitivity_maps, mask_weights)
    y_sense *= sigma
    y_grad *= sigma
    Ax = stacked_adjoint(y_sense, y_grad, sensitivity_maps, mask_weights)
    max_eig = np.real(np.vdot(x, Ax))
    return float(max_eig)


# ---------------------------------------------------------------------------
# PDHG (Chambolle-Pock) solver
# ---------------------------------------------------------------------------

def pdhg_tv_recon(masked_kspace, sensitivity_maps, lamda,
                  max_iter=100, tol=0.0):
    """TV-regularized MRI reconstruction via PDHG (Chambolle-Pock).

    Solves:  min_x  (1/2) ||P^{1/2} F S x - y||_2^2  +  lamda * TV(x)

    using the primal-dual hybrid gradient algorithm with stacked operator
    A = [Sense; FiniteDifference].

    Ported from SigPy's TotalVariationRecon (app.py) + PrimalDualHybridGradient
    (alg.py).

    Parameters
    ----------
    masked_kspace : ndarray, (C, H, W) complex
        Undersampled multi-coil k-space.
    sensitivity_maps : ndarray, (C, H, W) complex
        Coil sensitivity maps.
    lamda : float
        TV regularization weight.
    max_iter : int
        Maximum PDHG iterations.
    tol : float
        Convergence tolerance on primal residual.

    Returns
    -------
    x : ndarray, (H, W) complex
        Reconstructed image.
    """
    C, H, W = masked_kspace.shape
    img_shape = (H, W)

    # Estimate mask weights: sqrt of binary sampling mask
    # SigPy's _estimate_weights uses rss(y) > 0
    rss_y = np.sqrt(np.sum(np.abs(masked_kspace) ** 2, axis=0))
    mask = (rss_y > 0).astype(np.float64)
    mask_weights = np.sqrt(mask)[None, :, :]  # (1, H, W) broadcast

    # Pre-weight the data: y = mask^{1/2} * masked_kspace
    y_weighted = masked_kspace * mask_weights

    # Step sizes via power iteration
    sigma = 1.0
    max_eig = estimate_max_eigenvalue(sensitivity_maps, mask_weights,
                                       img_shape, sigma=sigma, max_iter=30)
    tau = 1.0 / max(max_eig, 1e-10)

    # Initialize primal and dual variables
    x = np.zeros(img_shape, dtype=np.complex128)
    u_sense = np.zeros_like(y_weighted)
    u_grad = np.zeros((2, H, W), dtype=np.complex128)
    x_ext = x.copy()

    theta = 1.0

    for it in range(max_iter):
        # --- Dual update ---
        # u_sense = proxfc_data(sigma, u_sense + sigma * Sense(x_ext))
        Ax_sense = sense_forward(x_ext, sensitivity_maps, mask_weights)
        u_sense_new = u_sense + sigma * Ax_sense
        u_sense = prox_l2_reg(sigma, u_sense_new, -y_weighted)

        # u_grad = proxfc_tv(sigma, u_grad + sigma * G(x_ext))
        Gx = finite_difference(x_ext)
        u_grad_new = u_grad + sigma * Gx
        u_grad = prox_l1_conj(sigma, u_grad_new, lamda)

        # --- Primal update ---
        x_old = x.copy()
        AHu = stacked_adjoint(u_sense, u_grad, sensitivity_maps, mask_weights)
        x = x - tau * AHu
        # proxg = NoOp (identity), so no further projection

        # --- Extrapolation ---
        x_diff = x - x_old
        resid = np.linalg.norm(x_diff) / max(np.sqrt(tau), 1e-30)
        x_ext = x + theta * x_diff

        if tol > 0 and resid < tol:
            break

    return x


# ---------------------------------------------------------------------------
# Public API (preserving original interface)
# ---------------------------------------------------------------------------

def tv_reconstruct_single(
    masked_kspace: np.ndarray,
    sensitivity_maps: np.ndarray,
    lamda: float = 1e-4,
) -> np.ndarray:
    """
    Reconstruct a single MRI image using Total Variation regularization.

    Parameters
    ----------
    masked_kspace : ndarray, (C, H, W) complex
        Undersampled multi-coil k-space for one sample.
    sensitivity_maps : ndarray, (C, H, W) complex
        Coil sensitivity maps for one sample.
    lamda : float
        TV regularization strength.

    Returns
    -------
    recon : ndarray, (H, W) complex
        Reconstructed complex image.
    """
    return pdhg_tv_recon(masked_kspace, sensitivity_maps, lamda)


def tv_reconstruct_batch(
    masked_kspace: np.ndarray,
    sensitivity_maps: np.ndarray,
    lamda: float = 1e-4,
) -> np.ndarray:
    """
    Reconstruct a batch of MRI images using Total Variation regularization.

    Parameters
    ----------
    masked_kspace : ndarray, (N, C, H, W) complex
        Undersampled multi-coil k-space.
    sensitivity_maps : ndarray, (N, C, H, W) complex
        Coil sensitivity maps.
    lamda : float
        TV regularization strength.

    Returns
    -------
    recons : ndarray, (N, H, W) complex
        Batch of reconstructed complex images.
    """
    n_samples = masked_kspace.shape[0]
    recons = []
    for i in range(n_samples):
        recon = tv_reconstruct_single(masked_kspace[i], sensitivity_maps[i], lamda)
        recons.append(recon)
    return np.stack(recons, axis=0)
