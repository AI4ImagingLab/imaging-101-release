"""Explicit local regularized inversion and vertical/horizontal uncertainty propagation.
Adapted from Alasdair Wilson demregpy, MIT; source revision and changes in test/validation.md.
"""
import warnings
import numpy as np
from numpy.linalg import svd
from scipy.optimize import brentq
from .physics_model import response_matrix
def _warn_nmu_deprecated(nmu):
    """Warn if the deprecated ``nmu`` argument was passed."""
    if nmu is not None:
        warnings.warn('nmu is deprecated and has no effect, because the regularisation parameter is now solved for exactly. It will be removed in demregpy 2.0.', DeprecationWarning, stacklevel=3)

def dem_pix(dnin, ednin, rmatrix, logt, dlogt, glc, reg_tweak=1.0, max_iter=10, rgt_fact=1.5, dem_norm0=None, nmu=None, warn=True, l_emd=False):
    """
    Recover a DEM for one observation vector.

    Parameters
    ----------
    dnin : array_like
        Input counts for one observation.
    ednin : array_like
        Uncertainties on ``dnin``.
    rmatrix : ndarray
        Temperature response of each channel.
    logt : array_like
        Log temperature bins.
    dlogt : array_like
        Size of temperature bins.
    glc : array_like
        Length-``nf`` 0/1 mask selecting the filters used for EM loci
        weighting.
    reg_tweak : float, optional
        Initial Chisq target, by default 1.0
    max_iter : int, optional
        Max number of iterations to reach target chisq before giving up, by default 10
    rgt_fact : float, optional
        Factor to increase chisq by each iteration, by default 1.5
    dem_norm0 : array_like, optional
        Initial guess at the dem shape, by default 0
    nmu : int, optional
        Deprecated and has no effect. Will be removed in demregpy 2.0.
    warn : bool, optional
        Print warnings, by default False
    l_emd : bool, optional
        Remove sqrt from constraint matrix, by default False

    Returns
    -------
    dem : ndarray
        Recovered DEM values.
    edem : ndarray
        Vertical uncertainties on ``dem``.
    elogt : ndarray
        Horizontal temperature resolution estimates in log10(T).
    chisq : array_like
        Reduced chi-squared value.
    dn_reg : ndarray
        Reconstructed counts for each filter.
    """
    _warn_nmu_deprecated(nmu)
    nf = rmatrix.shape[1]
    nt = logt.shape[0]
    if not np.all(np.isfinite(dnin)):
        raise ValueError('dnin must contain only finite values')
    if np.any(dnin < 0):
        raise ValueError('dnin must be non-negative')
    user_supplied_dem_norm0 = dem_norm0 is not None
    if dem_norm0 is None:
        dem_norm0 = np.ones(nt)
    elif np.isscalar(dem_norm0):
        raise ValueError('dem_norm0 must be an array matching (nt,), not a scalar')
    if nt < 3:
        raise ValueError('logt/dlogt must define at least 3 DEM bins')
    ltt = np.min(logt) + 1e-08 + (np.max(logt) - np.min(logt)) * np.arange(51) / (52 - 1.0)
    dem = np.zeros(nt)
    edem = np.zeros(nt)
    elogt = np.zeros(nt)
    chisq = 0
    dn_reg = np.zeros(nf)
    rmatrixin = rmatrix / ednin[np.newaxis, :]
    dn = dnin / ednin
    edn = ednin / ednin
    ndem = 1
    piter = 0
    rgt = reg_tweak
    if not user_supplied_dem_norm0 or np.all(dem_norm0 == 1.0) or dem_norm0[0] == 0:
        if np.sum(glc) > 0.0:
            gdglc = (glc > 0).nonzero()[0]
            emloci = np.zeros((nt, gdglc.shape[0]))
            for ee in np.arange(gdglc.shape[0]):
                emloci[:, ee] = dnin[gdglc[ee]] / rmatrix[:, gdglc[ee]]
            dem_model = np.zeros(nt)
            for ttt in np.arange(nt):
                nz = np.nonzero(emloci[ttt, :])[0]
                dem_model[ttt] = np.min(emloci[ttt, nz]) if nz.size > 0 else 0.0
            dem_reg_lwght = dem_model
        else:
            ldiag = 1.0 / np.sqrt(dlogt[:])
            basis = _standard_form_svd(rmatrixin.T, ldiag)
            lamb = _discrepancy_lambda(basis, dn, rgt * np.sum(edn ** 2))
            kdag = _regularised_inverse(basis, lamb)
            dr0 = (kdag @ dn).squeeze()
            fcofmax = 0.0001
            mask = (dr0 > 0) & (dr0 > fcofmax * np.max(dr0))
            fill = fcofmax * np.max(dr0) if np.any(mask) else 1.0
            dem_reg_lwght = np.full(nt, fill)
            dem_reg_lwght[mask] = dr0[mask]
        dem_reg_lwght = np.convolve(dem_reg_lwght[1:-1], np.ones(5) / 5)[1:-1] / np.max(dem_reg_lwght[:])
        dem_reg_lwght[dem_reg_lwght <= 1e-08] = 1e-08
    else:
        dem_reg_lwght = dem_norm0
    if l_emd:
        ldiag = 1 / abs(dem_reg_lwght)
    else:
        ldiag = np.sqrt(dlogt) / np.sqrt(abs(dem_reg_lwght))
    basis = _standard_form_svd(rmatrixin.T, ldiag)
    err_term = np.sum(edn ** 2)
    while ndem > 0 and piter < max_iter:
        lamb = _discrepancy_lambda(basis, dn, rgt * err_term)
        kdag = _regularised_inverse(basis, lamb)
        dem_reg_out = (kdag @ dn).squeeze()
        ndem = len(dem_reg_out[dem_reg_out < 0])
        rgt = rgt_fact * rgt
        piter += 1
    if warn and piter == max_iter:
        print('Warning, positivity loop hit max iterations, so increase max_iter? Or rgt_fact too small?')
    dem = dem_reg_out
    dn_reg = (rmatrix.T @ dem_reg_out).squeeze()
    residuals = (dnin - dn_reg) / ednin
    chisq = np.sum(residuals ** 2) / nf
    edem = np.sqrt(np.sum(kdag ** 2, axis=1))
    kdagk = kdag @ rmatrixin.T
    kdagk_max = np.max(kdagk, axis=0)
    elogt = np.zeros(nt)
    for kk in np.arange(nt):
        rr = np.interp(ltt, logt, kdagk[:, kk])
        hm_mask = rr >= kdagk_max[kk] / 2.0
        elogt[kk] = dlogt[kk]
        if np.sum(hm_mask) > 0:
            elogt[kk] = (ltt[hm_mask][-1] - ltt[hm_mask][0]) / 2
    return (dem, edem, elogt, chisq, dn_reg)

def _standard_form_svd(A, ldiag):
    """
    SVD of the response with the diagonal constraint divided out, ``A L^-1 = U diag(s) V^T``.

    The solution of min ||A x - d||^2 + lam ||L x||^2 is then
    ``x = L^-1 V diag(s / (s^2 + lam)) U^T d`` for any regularisation strength ``lam``, so
    trying many ``lam`` needs only this one decomposition.

    Returns a dict with ``U`` (nf, k), ``s`` (k,), ``V`` (nt, k) and ``linv`` (nt,).
    """
    ldiag = np.asarray(ldiag, dtype=float)
    linv = np.divide(1.0, ldiag, out=np.zeros_like(ldiag), where=ldiag != 0)
    U, s, Vt = svd(A * linv[np.newaxis, :], full_matrices=False)
    return {'U': U, 's': s, 'V': Vt.T, 'linv': linv}

def _misfit(basis, coef, outside, lam):
    """Misfit ||A x - d||^2 of the regularised solution at strength ``lam``."""
    s2 = basis['s'] ** 2
    return np.sum((lam / (s2 + lam) * coef) ** 2) + outside

def _discrepancy_lambda(basis, data, target):
    """
    Find the regularisation strength at which the misfit just reaches ``target`` (the
    discrepancy principle).

    Stronger regularisation always fits worse, so there is a single answer and it is found exactly.
    If ``target`` can't be reached, the nearer end of the search range is returned.
    """
    coef = basis['U'].T @ data
    outside = max(float(np.sum(data ** 2) - np.sum(coef ** 2)), 0.0)
    s2 = basis['s'] ** 2
    s2pos = s2[s2 > 0]
    if s2pos.size == 0:
        return 1.0
    lo, hi = (np.log(s2pos.min() * 1e-08), np.log(s2pos.max() * 100000000.0))
    f = lambda loglam: _misfit(basis, coef, outside, np.exp(loglam)) - target
    flo, fhi = (f(lo), f(hi))
    if flo >= 0:
        return float(np.exp(lo))
    if fhi <= 0:
        return float(np.exp(hi))
    return float(np.exp(brentq(f, lo, hi, xtol=1e-10)))

def _regularised_inverse(basis, lam):
    """The (nt, nf) matrix ``L^-1 V diag(s / (s^2 + lam)) U^T`` taking weighted data to the DEM."""
    s = basis['s']
    filt = s / (s ** 2 + lam)
    return basis['linv'][:, None] * (basis['V'] @ (filt[:, None] * basis['U'].T))

def solve(data,reg_tweak=1.0,gloci=False,error_scale=1.0):
    """Return DEM, vertical sigma, temperature sigma, chi2/channel, fitted count rates.
    Each spatial pixel uses independent whitened SVD/Tikhonov inversion.
    Default self-weighting, discrepancy target 1, positivity relaxation factor1.5, at most10 solves.
    """
    matrix,logt,dlogt=response_matrix(data['response'],data['response_log_temperature'],data['temperature_edges'])
    rate=data['count_rates'];err=data['uncertainty']*error_scale
    shape=rate.shape[:-1];nt=matrix.shape[0]
    results=[np.zeros(shape+(nt,)) for i in range(3)]+[np.zeros(shape),np.zeros_like(rate)]
    for index in np.ndindex(shape):
        value=dem_pix(rate[index],err[index],matrix*1e15,logt,dlogt,np.full(6,int(gloci)),reg_tweak=reg_tweak,warn=False)
        for target,source in zip(results,value):target[index]=source
    results[0]*=1e15;results[1]*=1e15;results[2]/=2*np.sqrt(2*np.log(2))
    return tuple(results)
def reconstruct(data):
    """Return float64 DEM (1,48,48,20), in cm^-5 K^-1."""
    return {'reconstruction':solve(data)[0][None]}
