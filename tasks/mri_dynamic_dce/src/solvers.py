"""
Dynamic MRI Solvers
===================

Implements reconstruction algorithms for dynamic (DCE) MRI:

1. **Zero-filled reconstruction** — frame-by-frame IFFT baseline.
2. **Temporal TV reconstruction** — jointly reconstructs all frames
   exploiting temporal sparsity of frame-to-frame changes.

   min_x  sum_t ||M_t F x_t - y_t||^2  +  lambda * sum_t ||x_{t+1} - x_t||_1

The solver uses Proximal Gradient Descent (PGD / ISTA) in the complex
image domain.  The gradient step enforces data consistency and the
proximal step applies temporal TV denoising via Chambolle's dual
algorithm on the temporal differences of the image magnitudes, while
preserving complex phase.
"""

import numpy as np
from .physics_model import fft2c, ifft2c


def zero_filled_recon(kspace, masks=None):
    """
    Frame-by-frame zero-filled reconstruction.

    Parameters
    ----------
    kspace : ndarray, (T, N, N) complex
        Undersampled k-space per frame.
    masks : ndarray or None
        Not used.

    Returns
    -------
    recon : ndarray, (T, N, N) float64
        Magnitude images (real-valued).
    """
    return np.abs(ifft2c(kspace))


def _temporal_diff(x):
    """
    Temporal finite difference: D_t x = x[t+1] - x[t].

    Parameters
    ----------
    x : ndarray, (T, ...)

    Returns
    -------
    dx : ndarray, (T-1, ...)
    """
    return x[1:] - x[:-1]


def _temporal_diff_adjoint(dx, T):
    """
    Adjoint of temporal finite difference operator.

    Parameters
    ----------
    dx : ndarray, (T-1, ...)
    T : int

    Returns
    -------
    result : ndarray, (T, ...)
    """
    shape = (T,) + dx.shape[1:]
    result = np.zeros(shape, dtype=dx.dtype)
    result[:-1] -= dx
    result[1:] += dx
    return result


def _soft_threshold_complex(x, threshold):
    """
    Complex soft thresholding: shrinks magnitude, preserves phase.

    Parameters
    ----------
    x : ndarray (complex or real)
    threshold : float

    Returns
    -------
    ndarray
    """
    mag = np.abs(x)
    shrunk = np.maximum(mag - threshold, 0.0)
    return np.where(mag > 1e-30, x * (shrunk / mag), 0.0)


def _prox_temporal_tv_chambolle(x, lamda, n_inner=30):
    """
    Proximal operator for temporal TV via Chambolle's dual algorithm.

    Solves: prox_{lambda * TV_t}(x) = x - lambda * D_t^H p*

    where p* solves the dual problem. Works for complex arrays
    by operating on magnitude and restoring phase.

    Parameters
    ----------
    x : ndarray, (T, N, N) complex or real
    lamda : float
    n_inner : int

    Returns
    -------
    z : ndarray, (T, N, N)
    """
    T = x.shape[0]
    p = np.zeros((T - 1,) + x.shape[1:], dtype=x.dtype)
    tau = 1.0 / 4.0  # 1/(2*||D||^2), ||D||^2 <= 4 for 1D differences

    for _ in range(n_inner):
        div_p = _temporal_diff_adjoint(p, T)
        Dx = _temporal_diff(x - lamda * div_p)
        p_new = p + tau * Dx
        # Project onto L-inf ball of radius 1
        mag = np.abs(p_new)
        p = p_new / np.maximum(1.0, mag)

    return x - lamda * _temporal_diff_adjoint(p, T)


def temporal_tv_pgd(kspace, masks, lamda=0.01, max_iter=150,
                    tol=1e-5, verbose=False):
    """
    Temporal TV-regularized dynamic MRI reconstruction via
    Proximal Gradient Descent (PGD / ISTA).

    Solves:
        min_x  sum_t ||M_t F x_t - y_t||^2  +  lambda ||D_t x||_1

    Each iteration:
        1. Gradient step: x <- x - step * F^H M_t (M_t F x - y)
        2. Proximal step: temporal TV denoising via Chambolle

    Parameters
    ----------
    kspace : ndarray, (T, N, N) complex
        Undersampled k-space per frame.
    masks : ndarray, (T, N, N)
        Per-frame undersampling masks.
    lamda : float
        Temporal TV regularization weight.
    max_iter : int
        Maximum PGD iterations.
    tol : float
        Convergence tolerance.
    verbose : bool
        Print iteration info.

    Returns
    -------
    recon : ndarray, (T, N, N) float64
        Reconstructed magnitude images.
    info : dict
        'loss_history', 'num_iter'
    """
    T, N, _ = kspace.shape

    # Step size: Lipschitz constant of gradient is 1 (ortho FFT, binary mask)
    step = 1.0

    # Initialize with zero-filled
    x = ifft2c(kspace).copy()

    loss_history = []

    def _compute_loss(x_cur):
        residual = fft2c(x_cur) * masks - kspace
        data_term = 0.5 * np.sum(np.abs(residual) ** 2)
        tv_term = lamda * np.sum(np.abs(_temporal_diff(x_cur)))
        return float(data_term + tv_term)

    for it in range(max_iter):
        x_old = x.copy()

        # Gradient step
        residual = fft2c(x) * masks - kspace
        grad = ifft2c(residual * masks)
        x = x - step * grad

        # Proximal step: temporal TV
        x = _prox_temporal_tv_chambolle(x, lamda * step, n_inner=20)

        # Convergence
        rel_change = np.linalg.norm(x - x_old) / (np.linalg.norm(x) + 1e-12)

        if it % 10 == 0 or it == max_iter - 1:
            loss = _compute_loss(x)
            loss_history.append(loss)
            if verbose and (it % 20 == 0 or it == max_iter - 1):
                print(f'  PGD iter {it:3d}: loss={loss:.4e}, '
                      f'rel_change={rel_change:.4e}')

        if rel_change < tol and it > 10:
            loss = _compute_loss(x)
            loss_history.append(loss)
            if verbose:
                print(f'  Converged at iteration {it}')
            break

    recon = np.abs(x)

    info = {
        'loss_history': loss_history,
        'num_iter': it + 1,
    }
    return recon, info


def temporal_tv_admm(kspace, masks, lamda=0.01, rho=1.0,
                     max_iter=100, tol=1e-5, verbose=False):
    """
    Temporal TV-regularized dynamic MRI reconstruction via ADMM.

    Solves:
        min_x  sum_t ||M_t F x_t - y_t||^2  +  lambda ||D_t x||_1

    ADMM with splitting z = D_t x:
        x-update: (A^H A + rho D_t^H D_t) x = A^H y + rho D_t^H (z - u)
                  solved via conjugate gradient
        z-update: soft-threshold(D_t x + u, lambda / rho)
        u-update: u += D_t x - z

    Parameters
    ----------
    kspace : ndarray, (T, N, N) complex
    masks : ndarray, (T, N, N)
    lamda : float
    rho : float
    max_iter : int
    tol : float
    verbose : bool

    Returns
    -------
    recon : ndarray, (T, N, N) float64
    info : dict
    """
    T, N, _ = kspace.shape

    x = ifft2c(kspace).copy()
    z = _temporal_diff(x)
    u = np.zeros_like(z)

    AHy = ifft2c(kspace * masks)
    loss_history = []

    def _compute_loss(x_cur):
        residual = fft2c(x_cur) * masks - kspace
        data_term = 0.5 * np.sum(np.abs(residual) ** 2)
        tv_term = lamda * np.sum(np.abs(_temporal_diff(x_cur)))
        return float(data_term + tv_term)

    def _apply_lhs(v):
        AHAv = ifft2c(fft2c(v) * masks)
        DtHDtv = _temporal_diff_adjoint(_temporal_diff(v), T)
        return AHAv + rho * DtHDtv

    for it in range(max_iter):
        x_old = x.copy()

        # x-update via CG
        rhs = AHy + rho * _temporal_diff_adjoint(z - u, T)
        x = _cg_solve(_apply_lhs, rhs, x0=x, max_iter=20, tol=1e-6)

        # z-update
        Dx = _temporal_diff(x)
        z = _soft_threshold_complex(Dx + u, lamda / rho)

        # u-update
        u = u + Dx - z

        rel_change = np.linalg.norm(x - x_old) / (np.linalg.norm(x) + 1e-12)

        if it % 10 == 0 or it == max_iter - 1:
            loss = _compute_loss(x)
            loss_history.append(loss)
            if verbose and (it % 20 == 0 or it == max_iter - 1):
                print(f'  ADMM iter {it:3d}: loss={loss:.4e}, '
                      f'rel_change={rel_change:.4e}')

        if rel_change < tol and it > 5:
            loss = _compute_loss(x)
            loss_history.append(loss)
            if verbose:
                print(f'  Converged at iteration {it}')
            break

    recon = np.abs(x)
    info = {
        'loss_history': loss_history,
        'num_iter': it + 1,
    }
    return recon, info


def _cg_solve(A_func, b, x0, max_iter=20, tol=1e-6):
    """
    Conjugate gradient solver for complex-valued A x = b.

    Parameters
    ----------
    A_func : callable
    b : ndarray
    x0 : ndarray
    max_iter : int
    tol : float

    Returns
    -------
    x : ndarray
    """
    x = x0.copy()
    r = b - A_func(x)
    p = r.copy()
    rs_old = np.real(np.sum(np.conj(r) * r))
    b_norm = np.sqrt(np.real(np.sum(np.conj(b) * b))) + 1e-30

    for _ in range(max_iter):
        Ap = A_func(p)
        pAp = np.real(np.sum(np.conj(p) * Ap))
        if abs(pAp) < 1e-30:
            break
        alpha = rs_old / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = np.real(np.sum(np.conj(r) * r))
        if np.sqrt(rs_new) / b_norm < tol:
            break
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new

    return x
