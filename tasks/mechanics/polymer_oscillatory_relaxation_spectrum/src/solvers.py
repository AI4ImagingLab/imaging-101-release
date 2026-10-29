"""Extracted explicit Shanbhag continuous and discrete spectrum mathematics.
Copyright Sachin Shanbhag; MIT task LICENSE. Acquisition kernels are local.
The extraction omits package I/O, plotting, aliases, configuration parsers,
and deprecated interfaces; retains Bayesian scan, analytic Jacobian, NNLS,
AIC mode placement, nonlinear refinement, pruning and mode merging.
"""
from __future__ import annotations
import warnings
import numpy as np
from dataclasses import dataclass
from typing import Optional
from scipy.optimize import least_squares,nnls,minimize
from scipy.integrate import cumulative_trapezoid,quad
from scipy.interpolate import interp1d
from .physics_model import get_kern_mat,kernel_prestore,kernel_D
from .preprocessing import prepare
class ReSpectWarning(UserWarning):pass
@dataclass
class ReSpectConfig:
    ns:int=100
    plateau:bool=False
    freq_end_int:int=1
    lam_min:float=1e-10
    lam_max:float=1e3
    lam_C:Optional[float]=None
    lam_density:int=2
    SmFacLam:float=0.
    max_num_modes:Optional[int]=None
    delta_base_weight_dist:float=.2
    min_tau_spacing:float=1.25
_N_MIN=2
_PRUNE_TOL=1e-7
_TAU_WINDOW=(.02,50.)
_MAX_MERGE_TRIES=3


@dataclass
class ContinuousResult:
    """Results from the continuous spectrum fit.

    The field names are the same in pyrespect_time and pyrespect_freq.

    Attributes
    ----------
    s : np.ndarray, shape (ns,)
        Relaxation mode axis.
    H : np.ndarray, shape (ns,)
        Log relaxation spectrum H(s); the spectrum itself is exp(H).
    G_fit : np.ndarray, shape (2n,)
        Reconstructed [G' | G''] from the continuous spectrum. The two
        halves are also available as ``Gp_fit`` and ``Gpp_fit``.
    G0 : float
        Plateau modulus. 0.0 if config.plateau is False.
    lam_C : float
        Regularization parameter λ used for the final fit.
    lam : np.ndarray or None
        Array of λ values scanned on the L-curve.
        None if lam_C was pre-specified in config.
    rho : np.ndarray or None
        Residual norm at each λ. None if lam_C was pre-specified.
    eta : np.ndarray or None
        Roughness norm at each λ. None if lam_C was pre-specified.
    log_P : np.ndarray or None
        Normalized log-probability log p(λ) at each λ.
        None if lam_C was pre-specified.
    dH : np.ndarray or None, shape (ns,)
        Error band on H: spread of H over the plausible range of λ.
        None if lam_C was pre-specified.
    H_lam : np.ndarray or None, shape (ns, n_lam)
        H snapshots at each scanned λ. None if lam_C was pre-specified.
    """
    s: np.ndarray
    H: np.ndarray
    G_fit: np.ndarray
    G0: float = 0.0
    lam_C: float = 0.0
    lam: Optional[np.ndarray] = None
    rho: Optional[np.ndarray] = None
    eta: Optional[np.ndarray] = None
    log_P: Optional[np.ndarray] = None
    dH: Optional[np.ndarray] = None
    H_lam: Optional[np.ndarray] = None

def _getAmatrix(ns: int) -> np.ndarray:
    """
    Symmetric matrix A = L^T L for error analysis.
    L is the (ns-2 x ns) second-difference operator.
    """
    nl = ns - 2
    L = np.diag(np.ones(ns - 1), 1) + np.diag(np.ones(ns - 1), -1) + np.diag(-2.0 * np.ones(ns))
    L = L[1:nl + 1, :]
    return np.dot(L.T, L)

def _getBmatrix(H: np.ndarray, kernMat: np.ndarray, Gexp: np.ndarray, wexp: np.ndarray, G0: float=0.0) -> np.ndarray:
    """
    B matrix required for the Bayesian log-evidence calculation.
    B = Jr^T Jr + diag(r^T Jr), where Jr is the weighted Jacobian.
    """
    n = kernMat.shape[0] // 2
    ns = kernMat.shape[1]
    Kmatrix = np.dot((wexp / Gexp).reshape(2 * n, 1), np.ones((1, ns)))
    Jr = -kernel_D(H, kernMat) * Kmatrix
    r = wexp * (1.0 - kernel_prestore(H, kernMat, G0) / Gexp)
    return np.dot(Jr.T, Jr) + np.diag(np.dot(r.T, Jr))

def _residualLM(H: np.ndarray, lam: float, Gexp: np.ndarray, wexp: np.ndarray, kernMat: np.ndarray) -> np.ndarray:
    """
    Residual vector for the Levenberg-Marquardt solve in getH.

    Layout: [ weighted data misfit (2n) | curvature penalty (nl) ]

    If len(H) > ns, the last element of H is G0 (plateau mode).
    """
    n = kernMat.shape[0] // 2
    ns = kernMat.shape[1]
    nl = ns - 2
    r = np.zeros(2 * n + nl)
    if len(H) > ns:
        G0 = H[-1]
        H = H[:-1]
        r[:2 * n] = wexp * (1.0 - kernel_prestore(H, kernMat, G0) / Gexp)
    else:
        r[:2 * n] = wexp * (1.0 - kernel_prestore(H, kernMat) / Gexp)
    r[2 * n:] = np.sqrt(lam) * np.diff(H, n=2)
    return r

def _jacobianLM(H: np.ndarray, lam: float, Gexp: np.ndarray, wexp: np.ndarray, kernMat: np.ndarray) -> np.ndarray:
    """
    Jacobian of _residualLM with respect to H (and optionally G0).
    Returns a (2n+nl x ns) matrix, or (2n+nl x ns+1) in plateau mode.
    """
    n = kernMat.shape[0] // 2
    ns = kernMat.shape[1]
    nl = ns - 2
    L = np.diag(np.ones(ns - 1), 1) + np.diag(np.ones(ns - 1), -1) + np.diag(-2.0 * np.ones(ns))
    L = L[1:nl + 1, :]
    Kmatrix = np.dot((wexp / Gexp).reshape(2 * n, 1), np.ones((1, ns)))
    if len(H) > ns:
        G0 = H[-1]
        H = H[:-1]
        Jr = np.zeros((2 * n + nl, ns + 1))
        Jr[:2 * n, :ns] = -kernel_D(H, kernMat) * Kmatrix
        Jr[:n, ns] = -wexp[:n] / Gexp[:n]
        Jr[2 * n:, :ns] = np.sqrt(lam) * L
        Jr[2 * n:, ns] = 0.0
    else:
        Jr = np.zeros((2 * n + nl, ns))
        Jr[:2 * n, :ns] = -kernel_D(H, kernMat) * Kmatrix
        Jr[2 * n:, :ns] = np.sqrt(lam) * L
    return Jr

def _getH(lam: float, Gexp: np.ndarray, wexp: np.ndarray, H: np.ndarray, kernMat: np.ndarray, G0: Optional[float]=None):
    """
    Solve the regularized least-squares problem for a given lambda:

        min_H  || w * (1 - K[H]/Gexp) ||^2  +  lambda * ||L H||^2

    Uses scipy least_squares (Trust-Region) with analytic Jacobian.

    Parameters
    ----------
    G0 : float or None
        If provided, G0 is appended to H and jointly optimized (plateau mode).

    Returns
    -------
    H_opt : (ns,) optimal H
    G0_opt : float  (only returned when G0 is not None)
    """
    if G0 is not None:
        Hplus = np.append(H, G0)
        res = least_squares(_residualLM, Hplus, jac=_jacobianLM, args=(lam, Gexp, wexp, kernMat))
        return (res.x[:-1], res.x[-1])
    else:
        res = least_squares(_residualLM, H, jac=_jacobianLM, args=(lam, Gexp, wexp, kernMat))
        return res.x

def _InitializeH(Gexp: np.ndarray, wexp: np.ndarray, s: np.ndarray, kernMat: np.ndarray, G0: Optional[float]=None):
    """
    Generate an initial guess for H by solving with a large lambda.
    A single guess suffices because lcurve sweeps from high to low lambda.

    Returns H [, G0] matching the calling convention of _getH.
    """
    H = -5.0 * np.ones(len(s)) + np.sin(np.pi * s)
    lam = 1.0
    if G0 is not None:
        return _getH(lam, Gexp, wexp, H, kernMat, G0)
    else:
        return _getH(lam, Gexp, wexp, H, kernMat)

def _lcurve(Gexp: np.ndarray, wexp: np.ndarray, Hgs: np.ndarray, kernMat: np.ndarray, config: ReSpectConfig, G0: Optional[float]=None):
    """
    Sweep lambda from lam_max down to lam_min, computing at each step:
      - rho : || weighted residual ||
      - eta : || second difference of H ||
      - logP: log Bayesian evidence  log p(lambda)

    Early termination when logP drops more than 18 units below its peak
    (the tail contributes negligibly to the posterior mean).

    Also computes dH, the error band on H (see _error_band).

    Returns
    -------
    lamM    : float       - posterior mean lambda
    lam     : (nkept,)   - lambda values explored
    rho     : (nkept,)
    eta     : (nkept,)
    logP    : (nkept,)   - normalised (max = 0)
    dH      : (ns,)      - error band on H
    Hlambda : (ns, nkept) - H snapshots at each lambda
    """
    ns = len(Hgs)
    npoints = int(config.lam_density * (np.log10(config.lam_max) - np.log10(config.lam_min)))
    hlam = (config.lam_max / config.lam_min) ** (1.0 / (npoints - 1))
    lam_arr = config.lam_min * hlam ** np.arange(npoints)
    eta = np.zeros(npoints)
    rho = np.zeros(npoints)
    logP = np.zeros(npoints)
    H = Hgs.copy()
    logPmax = -np.inf
    Hlambda = np.zeros((ns, npoints))
    Amat = _getAmatrix(ns)
    _, LogDetN = np.linalg.slogdet(Amat)
    i_stop = 0
    for i in reversed(range(npoints)):
        lamb = lam_arr[i]
        if G0 is not None:
            H, G0 = _getH(lamb, Gexp, wexp, H, kernMat, G0)
            rho[i] = np.linalg.norm(wexp * (1.0 - kernel_prestore(H, kernMat, G0) / Gexp))
            Bmat = _getBmatrix(H, kernMat, Gexp, wexp, G0)
        else:
            H = _getH(lamb, Gexp, wexp, H, kernMat)
            rho[i] = np.linalg.norm(wexp * (1.0 - kernel_prestore(H, kernMat) / Gexp))
            Bmat = _getBmatrix(H, kernMat, Gexp, wexp)
        eta[i] = np.linalg.norm(np.diff(H, n=2))
        Hlambda[:, i] = H
        _, LogDetC = np.linalg.slogdet(lamb * Amat + Bmat)
        V = rho[i] ** 2 + lamb * eta[i] ** 2
        logP[i] = -V + 0.5 * (LogDetN + ns * np.log(lamb) - LogDetC) - lamb
        if logP[i] > logPmax:
            logPmax = logP[i]
        elif logP[i] < logPmax - 18:
            i_stop = i
            break
    lam_arr = lam_arr[i_stop:]
    logP = logP[i_stop:]
    eta = eta[i_stop:]
    rho = rho[i_stop:]
    Hlambda = Hlambda[:, i_stop:]
    logP = logP - np.max(logP)
    plam = np.exp(logP)
    plam = plam / np.sum(plam)
    lamM = np.exp(np.sum(plam * np.log(lam_arr)))
    if config.SmFacLam > 0:
        lamM = np.exp(np.log(lamM) + config.SmFacLam * (np.max(np.log(lam_arr)) - np.log(lamM)))
    elif config.SmFacLam < 0:
        lamM = np.exp(np.log(lamM) + config.SmFacLam * (np.log(lamM) - np.min(np.log(lam_arr))))
    dH = _error_band(Hlambda, plam)
    return (lamM, lam_arr, rho, eta, logP, dH, Hlambda)

def fit_continuous(w: np.ndarray, Gexp: np.ndarray, wexp: np.ndarray, config: ReSpectConfig) -> ContinuousResult:
    """
    Extract the continuous relaxation spectrum H(s) from G*(w) data.

    Parameters
    ----------
    w     : (n,)  angular frequencies
    Gexp  : (2n,) [G' | G''] concatenated
    wexp  : (2n,) per-datapoint weights
    config: ReSpectConfig

    Returns
    -------
    ContinuousResult
    """
    n = len(w)
    ns = config.ns
    wmin = w[0]
    wmax = w[-1]
    fc = config.freq_end_int
    if fc == 1:
        smin = np.exp(-np.pi / 2) / wmax
        smax = np.exp(np.pi / 2) / wmin
    elif fc == 2:
        smin = 1.0 / wmax
        smax = 1.0 / wmin
    else:
        smin = np.exp(np.pi / 2) / wmax
        smax = np.exp(-np.pi / 2) / wmin
    hs = (smax / smin) ** (1.0 / (ns - 1))
    s = smin * hs ** np.arange(ns)
    kernMat = get_kern_mat(s, w)
    G0_guess = float(np.min(Gexp)) if config.plateau else None
    if config.plateau:
        Hgs, G0 = _InitializeH(Gexp, wexp, s, kernMat, G0_guess)
    else:
        Hgs = _InitializeH(Gexp, wexp, s, kernMat)
        G0 = 0.0
    lam_arr = rho = eta = logP = dH = H_lam = None
    if config.lam_C is None:
        if config.plateau:
            lamC, lam_arr, rho, eta, logP, dH, H_lam = _lcurve(Gexp, wexp, Hgs, kernMat, config, G0)
        else:
            lamC, lam_arr, rho, eta, logP, dH, H_lam = _lcurve(Gexp, wexp, Hgs, kernMat, config)
    else:
        lamC = config.lam_C
    if config.plateau:
        H, G0 = _getH(lamC, Gexp, wexp, Hgs, kernMat, G0)
    else:
        H = _getH(lamC, Gexp, wexp, Hgs, kernMat)
        G0 = 0.0
    G_fit = kernel_prestore(H, kernMat, G0)
    return ContinuousResult(s=s, H=H, G_fit=G_fit, G0=float(G0), lam_C=float(lamC), lam=lam_arr, rho=rho, eta=eta, log_P=logP, dH=dH, H_lam=H_lam)

def _error_band(H_lam: np.ndarray, p_lam: np.ndarray) -> np.ndarray:
    """Error band on H: its spread over the plausible range of λ.

    Standard deviation of the H_λ snapshots over all grid points with
    p(λ) > 0.1. If the λ grid is so fine that no single point carries
    that much probability (large lam_density), the p(λ)-weighted standard
    deviation over all scanned points is used instead.

    Parameters
    ----------
    H_lam : np.ndarray, shape (ns, n_lam)
        H snapshots at each scanned λ.
    p_lam : np.ndarray, shape (n_lam,)
        Normalized probability of each λ (sums to 1).

    Returns
    -------
    dH : np.ndarray, shape (ns,)
    """
    ns = H_lam.shape[0]
    Hm = np.zeros(ns)
    Hm2 = np.zeros(ns)
    cnt = 0
    for i in range(len(p_lam)):
        if p_lam[i] > 0.1:
            Hm += H_lam[:, i]
            Hm2 += H_lam[:, i] ** 2
            cnt += 1
    if cnt > 0:
        Hm = Hm / cnt
        Hm2 = Hm2 / cnt
    else:
        Hm = H_lam @ p_lam
        Hm2 = H_lam ** 2 @ p_lam
    return np.sqrt(np.maximum(Hm2 - Hm ** 2, 0.0))

@dataclass
class DiscreteResult:
    """Results from the discrete spectrum fit.

    The field names are the same in pyrespect_time and pyrespect_freq.

    Attributes
    ----------
    g : np.ndarray, shape (N,)
        Maxwell mode weights g_i.
    tau : np.ndarray, shape (N,)
        Maxwell relaxation times τ_i, in increasing order.
    dtau : np.ndarray, shape (N,)
        Uncertainty estimates for τ_i. Entries are np.nan when the NLLS
        fine-tuning step was not used.
    N : int
        Number of Maxwell modes returned (len(g)).
    G_fit : np.ndarray, shape (2n,)
        Reconstructed [G' | G''] from the discrete spectrum. The two
        halves are also available as ``Gp_fit`` and ``Gpp_fit``.
    G0 : float
        Plateau modulus. 0.0 if config.plateau is False.
    error : float
        Weighted sum of squared relative residuals of the discrete fit.
    wt_base : np.ndarray
        Scanned base weight values w_b.
    AIC_bst : np.ndarray
        Best AIC value at each w_b.
    N_bst : np.ndarray
        Number of modes requested (the N that enters the AIC penalty)
        at the best AIC for each w_b.
    nz_N_bst : np.ndarray
        Number of modes that survive pruning at the best AIC for each w_b.
    """
    g: np.ndarray
    tau: np.ndarray
    dtau: np.ndarray
    N: int
    G_fit: np.ndarray
    G0: float = 0.0
    error: float = 0.0
    wt_base: Optional[np.ndarray] = None
    AIC_bst: Optional[np.ndarray] = None
    N_bst: Optional[np.ndarray] = None
    nz_N_bst: Optional[np.ndarray] = None

def fit_discrete(w: np.ndarray, Gexp: np.ndarray, wexp: np.ndarray, cont_result: ContinuousResult, config: ReSpectConfig) -> DiscreteResult:
    """Fit a discrete Maxwell spectrum to dynamic moduli data.

    Uses the continuous spectrum from fit_continuous() to guide the
    placement of discrete modes, selects their number by minimizing

        AIC = 2 N + 2 C_error * error(N)

    over a grid of base weights w_b, fine-tunes the τ positions by NLLS,
    and merges modes that are too close (τ_{i+1}/τ_i < min_tau_spacing).

    Parameters
    ----------
    w : np.ndarray, shape (n,)
        Angular frequencies.
    Gexp : np.ndarray, shape (2n,)
        Experimental data, concatenated [G' | G''].
    wexp : np.ndarray, shape (2n,)
        Per-datapoint weights, concatenated like Gexp.
    cont_result : ContinuousResult
        Output of fit_continuous(). Provides s, H, and G_fit.
    config : ReSpectConfig
        Configuration object.

    Returns
    -------
    DiscreteResult
    """
    s = cont_result.s
    H = cont_result.H
    plateau = config.plateau
    Nv = _mode_counts(np.max(w) / np.min(w), len(w), config.max_num_modes)
    npts = len(Nv)
    Gc = cont_result.G_fit
    C_error = 1.0 / np.std(wexp * (Gc / Gexp - 1.0))
    delta = config.delta_base_weight_dist
    wt_base = delta * np.arange(1, int(1.0 / delta))
    n_wb = len(wt_base)
    AIC_bst = np.zeros(n_wb)
    N_bst = np.zeros(n_wb, dtype=int)
    nz_N_bst = np.zeros(n_wb, dtype=int)
    for ib, wb in enumerate(wt_base):
        wt = _get_weights(H, w, s, wb)
        ev = np.zeros(npts)
        nz_Nv = np.zeros(npts, dtype=int)
        for i, N in enumerate(Nv):
            z, _ = _grid_density(np.log(s), wt, N)
            _, tau, ev[i], _ = _maxwell_modes(z, w, Gexp, wexp, plateau)
            nz_Nv[i] = len(tau)
        AIC = 2.0 * Nv + 2.0 * C_error * ev
        AIC_bst[ib] = np.min(AIC)
        N_bst[ib] = Nv[np.argmin(AIC)]
        nz_N_bst[ib] = nz_Nv[np.argmin(AIC)]
    i_best = np.argmin(AIC_bst)
    wt = _get_weights(H, w, s, wt_base[i_best])
    z, _ = _grid_density(np.log(s), wt, int(N_bst[i_best]))
    g, tau, _, _ = _maxwell_modes(z, w, Gexp, wexp, plateau)
    dtau = np.full(len(tau), np.nan)
    ok, g_f, tau_f, dtau_f = _fine_tune(tau, w, Gexp, wexp, plateau)
    if ok:
        g, tau, dtau = (g_f, tau_f, dtau_f)
    itry = 0
    while len(tau) > 1 and itry < _MAX_MERGE_TRIES:
        tau_spacing = tau[1:] / tau[:-1]
        if np.min(tau_spacing) >= config.min_tau_spacing:
            break
        tau_m = _merge_modes(g, tau, int(np.argmin(tau_spacing)))
        ok, g_f, tau_f, dtau_f = _fine_tune(tau_m, w, Gexp, wexp, plateau)
        if ok:
            g, tau, dtau = (g_f, tau_f, dtau_f)
        else:
            g, tau, _, _ = _maxwell_modes(np.log(tau_m), w, Gexp, wexp, plateau)
            dtau = np.full(len(tau), np.nan)
        itry += 1
    G0 = 0.0
    if plateau:
        G0 = float(g[-1])
        g = g[:-1]
    G_fit = _maxwell_kernel(tau, w) @ g
    G_fit[:len(w)] += G0
    error = float(np.sum((wexp * (G_fit / Gexp - 1.0)) ** 2))
    return DiscreteResult(g=g, tau=tau, dtau=dtau, N=len(g), G_fit=G_fit, G0=G0, error=error, wt_base=wt_base, AIC_bst=AIC_bst, N_bst=N_bst, nz_N_bst=nz_N_bst)

def _maxwell_kernel(tau: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Maxwell kernel for [G' | G''], shape (2n, N)."""
    S, W = np.meshgrid(tau, w)
    ws = S * W
    ws2 = ws ** 2
    return np.vstack((ws2 / (1 + ws2), ws / (1 + ws2)))

def _design_matrix(tau: np.ndarray, w: np.ndarray, plateau: bool) -> np.ndarray:
    """Kernel with an extra column for G0 when plateau=True.

    G0 contributes to the G' rows only.
    """
    K = _maxwell_kernel(tau, w)
    if plateau:
        n = len(w)
        col = np.zeros((2 * n, 1))
        col[:n] = 1.0
        K = np.hstack((K, col))
    return K

def _model(w: np.ndarray, tau: np.ndarray, g: np.ndarray, plateau: bool) -> np.ndarray:
    """Model prediction for modes (g, tau); G0 is g[-1] when plateau=True."""
    if not plateau:
        return _maxwell_kernel(tau, w) @ g
    G = _maxwell_kernel(tau, w) @ g[:-1]
    G[:len(w)] += g[-1]
    return G

def _tau_bounds(w: np.ndarray) -> tuple[float, float]:
    """Allowed range of τ for the given data window."""
    return (_TAU_WINDOW[0] / np.max(w), _TAU_WINDOW[1] / np.min(w))

def _mode_counts(span: float, n: int, max_num_modes: Optional[int]) -> np.ndarray:
    """Values of N scanned by the AIC search.

    Parameters
    ----------
    span : float
        Ratio of the largest to the smallest abscissa of the data.
    n : int
        Number of data abscissae.
    max_num_modes : int or None
        User cap on N.

    Returns
    -------
    Nv : np.ndarray of int
    """
    decades = np.log10(span)
    N_min = int(max(np.floor(0.5 * decades), _N_MIN))
    N_max = int(min(np.floor(3.0 * decades), n / 4))
    if max_num_modes is not None:
        N_max = min(N_max, max_num_modes)
    if N_max > N_min:
        return np.arange(N_min, N_max + 1, dtype=int)
    warnings.warn(f'Only N = {N_max} is scanned for the discrete spectrum; make sure max_num_modes is set prudently.', ReSpectWarning, stacklevel=4)
    return np.arange(N_max, N_max + 1, dtype=int)

def _nn_lls(w: np.ndarray, tau: np.ndarray, Gexp: np.ndarray, wexp: np.ndarray, plateau: bool) -> tuple[np.ndarray, float]:
    """Solve the non-negative least squares problem for Maxwell weights.

    Minimizes || w * (K g / G_exp - 1) ||^2 subject to g >= 0.

    Parameters
    ----------
    w : np.ndarray, shape (n,)
    tau : np.ndarray, shape (N,)
    Gexp : np.ndarray, shape (2n,)
    wexp : np.ndarray, shape (2n,)
    plateau : bool
        If True, G0 is fitted as an extra, last element of g.

    Returns
    -------
    g : np.ndarray
        Non-negative weights (and G0 as last element if plateau=True).
    error : float
        Weighted residual sum of squares.
    """
    K = _design_matrix(tau, w, plateau)
    Kp = (wexp / Gexp).reshape(-1, 1) * K
    g = nnls(Kp, wexp, maxiter=100000)[0]
    error = float(np.sum((wexp * (K @ g / Gexp - 1.0)) ** 2))
    return (g, error)

def _maxwell_modes(z: np.ndarray, w: np.ndarray, Gexp: np.ndarray, wexp: np.ndarray, plateau: bool) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Solve for Maxwell modes at log-spaced positions z = log(τ).

    Solves NNLS, then drops modes outside the allowed τ window and modes
    with negligible weight (g_i / max(g) < _PRUNE_TOL), and sorts the
    rest by τ.

    Parameters
    ----------
    z : np.ndarray, shape (N,)
        Log relaxation times log(τ).
    w : np.ndarray, shape (n,)
    Gexp : np.ndarray, shape (2n,)
    wexp : np.ndarray, shape (2n,)
    plateau : bool

    Returns
    -------
    g : np.ndarray
        Weights of the surviving modes (G0 appended if plateau=True).
    tau : np.ndarray
        Relaxation times of the surviving modes, increasing.
    error : float
        Weighted residual sum of squares of the NNLS solution.
    keep : np.ndarray of int
        Indices into z of the surviving modes, in the order returned.
    """
    tau = np.exp(z)
    g, error = _nn_lls(w, tau, Gexp, wexp, plateau)
    g_modes = g[:-1] if plateau else g
    lo, hi = _tau_bounds(w)
    in_window = (tau >= lo) & (tau <= hi)
    keep = np.where(in_window)[0]
    g_ref = np.max(g_modes[keep]) if len(keep) else 1.0
    keep = keep[g_modes[keep] / g_ref >= _PRUNE_TOL]
    keep = keep[np.argsort(tau[keep])]
    g_out = g_modes[keep]
    if plateau:
        g_out = np.append(g_out, g[-1])
    return (g_out, tau[keep], error, keep)

def _get_weights(H: np.ndarray, w: np.ndarray, s: np.ndarray, wb: float) -> np.ndarray:
    """Compute the weight of each relaxation mode on the continuous axis.

    Weights reflect each mode's contribution to G*(w), blended with a
    uniform baseline controlled by wb.

    Parameters
    ----------
    H : np.ndarray, shape (ns,)
        Log continuous spectrum.
    w : np.ndarray, shape (n,)
    s : np.ndarray, shape (ns,)
    wb : float
        Base weight blending factor in [0, 1).

    Returns
    -------
    wt : np.ndarray, shape (ns,)
        Normalized weights for mode placement.
    """
    ns = len(s)
    n = len(w)
    hs = np.zeros(ns)
    hs[0] = 0.5 * np.log(s[1] / s[0])
    hs[-1] = 0.5 * np.log(s[-1] / s[-2])
    hs[1:-1] = 0.5 * (np.log(s[2:]) - np.log(s[:-2]))
    kern = _maxwell_kernel(s, w)
    wij = np.dot(kern, np.diag(hs * np.exp(H)))
    K = np.dot(kern, hs * np.exp(H))
    for i in range(n):
        wij[i, :] = wij[i, :] / K[i]
    wt = np.sum(wij, axis=0)
    wt = wt / np.trapezoid(wt, np.log(s))
    wt = (1.0 - wb) * wt + wb * np.mean(wt) * np.ones(ns)
    return wt

def _grid_density(x: np.ndarray, px: np.ndarray, N: int) -> tuple[np.ndarray, np.ndarray]:
    """Distribute N points according to a density function px(x).

    Places quadrature points such that each interval carries equal
    probability mass under px, with endpoints of x always included.

    Parameters
    ----------
    x : np.ndarray
        Domain points (need not be equispaced).
    px : np.ndarray
        Density or probability distribution (positive, need not be normalized).
    N : int
        Number of output points.

    Returns
    -------
    z : np.ndarray, shape (N,)
        Points distributed according to px.
    h : np.ndarray, shape (N,)
        Interval widths (useful for quadrature).
    """
    npts = 100
    xi = np.linspace(x.min(), x.max(), npts)
    fint = interp1d(x, px, kind='cubic')
    pint = fint(xi)
    ci = cumulative_trapezoid(pint, xi, initial=0)
    pint = pint / ci[-1]
    ci = ci / ci[-1]
    alfa = 1.0 / max(N - 1, 1)
    zij = np.zeros(N + 1)
    z = np.zeros(N)
    z[0] = x.min()
    z[-1] = x.max()
    beta = np.arange(0.5, N - 0.5) * alfa
    zij[0] = z[0]
    zij[-1] = z[-1]
    fint_inv = interp1d(ci, xi, kind='cubic')
    zij[1:N] = fint_inv(beta)
    h = np.diff(zij)
    beta = np.arange(1, N - 1) * alfa
    z[1:-1] = fint_inv(beta)
    return (z, h)

def _merge_modes(g: np.ndarray, tau: np.ndarray, imode: int) -> np.ndarray:
    """Merge modes imode and imode+1 into a single mode.

    Finds the best single-mode approximation to the two-mode pair
    by minimizing the integrated squared relative difference.

    Parameters
    ----------
    g : np.ndarray
        Current mode weights.
    tau : np.ndarray
        Current relaxation times.
    imode : int
        Index of the first mode to merge.

    Returns
    -------
    tau_new : np.ndarray
        Updated relaxation times with one fewer mode.
    """
    g1, tau1 = (g[imode], tau[imode])
    g2, tau2 = (g[imode + 1], tau[imode + 1])

    def _integrand(w: float, gn: float, taun: float) -> float:
        wt = w * taun
        Gnp = gn * wt ** 2 / (1 + wt ** 2)
        Gnpp = gn * wt / (1 + wt ** 2)
        wt = w * tau1
        Gop = g1 * wt ** 2 / (1 + wt ** 2)
        Gopp = g1 * wt / (1 + wt ** 2)
        wt = w * tau2
        Gop += g2 * wt ** 2 / (1 + wt ** 2)
        Gopp += g2 * wt / (1 + wt ** 2)
        return (Gnp / Gop - 1.0) ** 2 + (Gnpp / Gopp - 1.0) ** 2

    def _cost(par: np.ndarray) -> float:
        """Integrated squared relative error between merged and original."""
        wmin = min(1.0 / tau1, 1.0 / tau2) / 10.0
        wmax = max(1.0 / tau1, 1.0 / tau2) * 10.0
        return quad(_integrand, wmin, wmax, args=(par[0], par[1]))[0]
    res = minimize(_cost, np.array([g1 + g2, 0.5 * (tau1 + tau2)]))
    tau_new = np.delete(tau, imode + 1)
    tau_new[imode] = res.x[1]
    return tau_new

def _fine_tune(tau: np.ndarray, w: np.ndarray, Gexp: np.ndarray, wexp: np.ndarray, plateau: bool) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray]:
    """Fine-tune mode positions via non-linear least squares.

    Refines the τ positions by NLLS (weights g are re-solved by NNLS at
    every step), keeping τ inside the allowed window. The refinement is
    reported as successful only if it converges and does not make the
    fit worse than the starting point.

    Parameters
    ----------
    tau : np.ndarray, shape (N,)
        Initial relaxation times.
    w : np.ndarray, shape (n,)
    Gexp : np.ndarray, shape (2n,)
    wexp : np.ndarray, shape (2n,)
    plateau : bool

    Returns
    -------
    success : bool
        If False, the caller should keep its starting solution.
    g : np.ndarray
        Weights at the refined positions (G0 appended if plateau=True).
    tau : np.ndarray
        Refined relaxation times, increasing.
    dtau : np.ndarray
        Uncertainty estimates for tau from the NLLS Jacobian.
    """

    def _residuals(tau_: np.ndarray) -> np.ndarray:
        g_, _ = _nn_lls(w, tau_, Gexp, wexp, plateau)
        return wexp * (_model(w, tau_, g_, plateau) / Gexp - 1.0)
    init_error = np.linalg.norm(_residuals(tau))
    try:
        res = least_squares(_residuals, tau, bounds=_tau_bounds(w))
        cov = np.linalg.pinv(res.jac.T @ res.jac) * (res.fun ** 2).mean()
        dtau = np.sqrt(np.diag(cov))
    except Exception:
        warnings.warn('NLLS fine-tuning of the discrete modes did not converge; keeping the NNLS solution. Uncertainty estimates (dtau) will be NaN.', ReSpectWarning, stacklevel=2)
        g, tau, _, _ = _maxwell_modes(np.log(tau), w, Gexp, wexp, plateau)
        return (False, g, tau, np.full(len(tau), np.nan))
    g, tau, _, keep = _maxwell_modes(np.log(res.x), w, Gexp, wexp, plateau)
    final_error = np.linalg.norm(_residuals(tau))
    return (bool(final_error <= init_error), g, tau, dtau[keep])

def solve(data,config=None):
    config=config or ReSpectConfig();w,g,weights=prepare(data)
    continuous=fit_continuous(w,g,weights,config)
    discrete=fit_discrete(w,g,weights,continuous,config)
    return w,continuous,discrete
def reconstruct(data):
    w,c,d=solve(data)
    return {'reconstruction':np.exp(c.H)[None,:],'maxwell_modes':np.array([d.g,d.tau])[None,:],'fitted_moduli':np.stack([c.G_fit,d.G_fit])[None,:]}
