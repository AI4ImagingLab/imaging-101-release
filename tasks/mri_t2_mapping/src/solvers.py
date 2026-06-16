"""
T2 mapping solvers: estimate T2 and M0 from multi-echo data.

Two methods:
1. Log-linear fit: linearize via log(S) = log(M0) - TE/T2
   (fast, closed-form, biased at low SNR)
2. Nonlinear least squares: fit S = M0*exp(-TE/T2) via Levenberg-Marquardt
   (iterative, unbiased, uses analytic Jacobian)

The Levenberg-Marquardt algorithm is extracted from the classical formulation
(Moré 1978, Marquardt 1963) which scipy.optimize.curve_fit wraps internally.

Reference
---------
Marquardt, D. W. (1963). An algorithm for least-squares estimation of
nonlinear parameters. SIAM J. Appl. Math.

Moré, J. J. (1978). The Levenberg-Marquardt algorithm: implementation and
theory. Numerical Analysis, Lecture Notes in Mathematics 630.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Mono-exponential model and its Jacobian
# ---------------------------------------------------------------------------

def mono_exp_model(TE, M0, T2):
    """
    Mono-exponential T2 decay model.

    S(TE) = M0 * exp(-TE / T2)

    Parameters
    ----------
    TE : ndarray, (N_echoes,)
        Echo times in ms.
    M0 : float
        Proton density (signal at TE=0).
    T2 : float
        Transverse relaxation time in ms.

    Returns
    -------
    signal : ndarray, (N_echoes,)
    """
    return M0 * np.exp(-TE / T2)


def mono_exp_jacobian(TE, M0, T2):
    """
    Jacobian of the mono-exponential model w.r.t. [M0, T2].

    J[:, 0] = dS/dM0 = exp(-TE / T2)
    J[:, 1] = dS/dT2 = M0 * TE / T2^2 * exp(-TE / T2)

    Parameters
    ----------
    TE : ndarray, (N_echoes,)
    M0 : float
    T2 : float

    Returns
    -------
    J : ndarray, (N_echoes, 2)
    """
    exp_term = np.exp(-TE / T2)
    J = np.zeros((len(TE), 2))
    J[:, 0] = exp_term                         # dS/dM0
    J[:, 1] = M0 * TE / (T2 ** 2) * exp_term   # dS/dT2
    return J


# ---------------------------------------------------------------------------
# Levenberg-Marquardt solver (extracted from classic LM algorithm)
# ---------------------------------------------------------------------------

def levenberg_marquardt_mono_exp(
    TE, signal, M0_init, T2_init,
    max_iter=50, tau=1e-3, eps1=1e-8, eps2=1e-8,
):
    """
    Levenberg-Marquardt algorithm for mono-exponential T2 fitting.

    Minimizes: sum_i (S_i - M0 * exp(-TE_i / T2))^2

    This is the Marquardt (1963) / Moré (1978) algorithm:
      1. Compute residual r = S - f(p) and Jacobian J
      2. Solve (J^T J + mu * diag(J^T J)) dp = J^T r
      3. If cost decreases, accept step and decrease mu; else increase mu

    Parameters
    ----------
    TE : ndarray, (N,)
        Echo times.
    signal : ndarray, (N,)
        Measured signal.
    M0_init, T2_init : float
        Initial parameter guesses.
    max_iter : int
        Maximum iterations.
    tau : float
        Initial damping factor scale.
    eps1, eps2 : float
        Convergence tolerances for gradient and step size.

    Returns
    -------
    M0, T2 : float
        Fitted parameters.
    converged : bool
    """
    p = np.array([M0_init, T2_init], dtype=np.float64)

    # Initial residual and Jacobian
    r = signal - mono_exp_model(TE, p[0], p[1])
    J = mono_exp_jacobian(TE, p[0], p[1])
    JtJ = J.T @ J
    Jtr = J.T @ r
    cost = 0.5 * np.dot(r, r)

    # Initial damping parameter
    mu = tau * np.max(np.diag(JtJ))
    nu = 2.0

    for _ in range(max_iter):
        # Check gradient convergence
        if np.max(np.abs(Jtr)) < eps1:
            return p[0], p[1], True

        # Solve (J^T J + mu * diag(J^T J)) dp = J^T r
        damping = mu * np.diag(np.diag(JtJ) + 1e-12)
        try:
            dp = np.linalg.solve(JtJ + damping, Jtr)
        except np.linalg.LinAlgError:
            return p[0], p[1], False

        # Check step size convergence
        if np.linalg.norm(dp) < eps2 * (np.linalg.norm(p) + eps2):
            return p[0], p[1], True

        # Trial step
        p_new = p + dp

        # Enforce positivity (T2 and M0 must be positive)
        p_new[0] = max(p_new[0], 1e-10)  # M0 > 0
        p_new[1] = max(p_new[1], 0.1)    # T2 > 0.1 ms

        # Evaluate new cost
        r_new = signal - mono_exp_model(TE, p_new[0], p_new[1])
        cost_new = 0.5 * np.dot(r_new, r_new)

        # Gain ratio: actual reduction / predicted reduction
        predicted_reduction = 0.5 * dp @ (mu * np.diag(np.diag(JtJ)) @ dp + Jtr)
        if predicted_reduction > 0:
            rho = (cost - cost_new) / predicted_reduction
        else:
            rho = -1  # force rejection

        if rho > 0:
            # Accept step
            p = p_new
            r = r_new
            J = mono_exp_jacobian(TE, p[0], p[1])
            JtJ = J.T @ J
            Jtr = J.T @ r
            cost = cost_new

            # Decrease damping
            mu *= max(1.0 / 3.0, 1.0 - (2.0 * rho - 1.0) ** 3)
            nu = 2.0
        else:
            # Reject step, increase damping
            mu *= nu
            nu *= 2.0

    return p[0], p[1], False


# ---------------------------------------------------------------------------
# Log-linear fit (vectorized, closed-form)
# ---------------------------------------------------------------------------

def fit_t2_loglinear(signal, TE, mask=None):
    """Estimate T2 and M0 via log-linear regression.

    Linearizes the mono-exponential model:
        log(S) = log(M0) - TE / T2

    This is a weighted linear regression in log-domain.
    Fast but biased, especially at low SNR (Rician noise bias).

    Parameters
    ----------
    signal : np.ndarray
        Multi-echo signal, shape (Ny, Nx, N_echoes).
    TE : np.ndarray
        Echo times in ms, shape (N_echoes,).
    mask : np.ndarray or None
        Boolean tissue mask, shape (Ny, Nx). If None, all pixels are fit.

    Returns
    -------
    T2_map : np.ndarray
        Estimated T2 map in ms, shape (Ny, Nx).
    M0_map : np.ndarray
        Estimated M0 map, shape (Ny, Nx).
    """
    TE = np.asarray(TE, dtype=np.float64)
    signal = np.asarray(signal, dtype=np.float64)
    Ny, Nx, N_echoes = signal.shape

    T2_map = np.zeros((Ny, Nx), dtype=np.float64)
    M0_map = np.zeros((Ny, Nx), dtype=np.float64)

    if mask is None:
        mask = np.ones((Ny, Nx), dtype=bool)

    # Vectorized log-linear fit for all masked pixels at once
    pixels = signal[mask]  # (N_pixels, N_echoes)
    pixels = np.maximum(pixels, 1e-10)
    log_signal = np.log(pixels)

    # Design matrix: [1, TE]
    A = np.column_stack([np.ones(N_echoes), TE])
    params, _, _, _ = np.linalg.lstsq(A, log_signal.T, rcond=None)

    log_M0 = params[0]
    neg_inv_T2 = params[1]

    with np.errstate(divide='ignore', invalid='ignore'):
        T2_vals = -1.0 / neg_inv_T2
    M0_vals = np.exp(log_M0)

    T2_vals = np.clip(T2_vals, 0, 5000)
    M0_vals = np.clip(M0_vals, 0, None)
    T2_vals = np.where(np.isfinite(T2_vals), T2_vals, 0.0)
    M0_vals = np.where(np.isfinite(M0_vals), M0_vals, 0.0)

    T2_map[mask] = T2_vals
    M0_map[mask] = M0_vals

    return T2_map, M0_map


# ---------------------------------------------------------------------------
# Nonlinear least squares fit (per-pixel Levenberg-Marquardt)
# ---------------------------------------------------------------------------

def fit_t2_nonlinear(signal, TE, mask=None, T2_init=None, M0_init=None):
    """Estimate T2 and M0 via nonlinear least-squares fitting.

    Fits S = M0 * exp(-TE / T2) per pixel using Levenberg-Marquardt.

    Parameters
    ----------
    signal : np.ndarray
        Multi-echo signal, shape (Ny, Nx, N_echoes).
    TE : np.ndarray
        Echo times in ms, shape (N_echoes,).
    mask : np.ndarray or None
        Boolean tissue mask, shape (Ny, Nx). If None, all pixels are fit.
    T2_init : np.ndarray or None
        Initial T2 guess, shape (Ny, Nx). If None, uses log-linear estimate.
    M0_init : np.ndarray or None
        Initial M0 guess, shape (Ny, Nx). If None, uses log-linear estimate.

    Returns
    -------
    T2_map : np.ndarray
        Estimated T2 map in ms, shape (Ny, Nx).
    M0_map : np.ndarray
        Estimated M0 map, shape (Ny, Nx).
    """
    TE = np.asarray(TE, dtype=np.float64)
    signal = np.asarray(signal, dtype=np.float64)
    Ny, Nx, N_echoes = signal.shape

    T2_map = np.zeros((Ny, Nx), dtype=np.float64)
    M0_map = np.zeros((Ny, Nx), dtype=np.float64)

    if mask is None:
        mask = np.ones((Ny, Nx), dtype=bool)

    # Get initial estimates from log-linear fit if not provided
    if T2_init is None or M0_init is None:
        T2_ll, M0_ll = fit_t2_loglinear(signal, TE, mask=mask)
        if T2_init is None:
            T2_init = T2_ll
        if M0_init is None:
            M0_init = M0_ll

    ys, xs = np.where(mask)
    for i in range(len(ys)):
        y, x = ys[i], xs[i]
        s = signal[y, x, :]
        t2_0 = max(T2_init[y, x], 1.0)
        m0_0 = max(M0_init[y, x], 1e-6)

        M0_fit, T2_fit, converged = levenberg_marquardt_mono_exp(
            TE, s, m0_0, t2_0,
        )

        # Clamp to physical range
        T2_fit = np.clip(T2_fit, 0, 5000)
        M0_fit = max(M0_fit, 0)

        if not converged:
            # Fall back to log-linear estimate
            M0_fit = M0_init[y, x]
            T2_fit = T2_init[y, x]

        M0_map[y, x] = M0_fit
        T2_map[y, x] = T2_fit

    return T2_map, M0_map
