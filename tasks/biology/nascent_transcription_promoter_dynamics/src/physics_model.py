"""Explicit Gaussian convolution and signed-log emission mathematics.
Adapted from Bowles et al. Apache-2.0 source; see LICENSE and validation.
Prehistory OFF correction is explicit in the time-independent emission basis.
"""
import numpy as np
import scipy
import scipy.special
import scipy.sparse
import itertools
from functools import lru_cache
def ms2_loading_coeff(kappa, W):
    if not 0<kappa<2 or W<2:raise ValueError('Selected source response supports 0<kappa<2 and W>=2; benchmark kappa=1.5')
    alpha = kappa
    coeff = np.ones((1, W), dtype=float)
    alpha_ceil = np.ceil(alpha)
    alpha_floor = np.floor(alpha)
    coeff[0:int(alpha_floor):1, 0] = (np.linspace(1, int(alpha_floor), endpoint=True, num=int(alpha_floor)) - 0.5) / alpha
    coeff[0, int(alpha_ceil) - 1] = alpha_ceil - alpha + (alpha ** 2 - (alpha_ceil - 1) ** 2) / (2 * alpha)
    return coeff

def _get_adjusted_uncached(state, K, W, ms2_coeff):
    ms2_coeff_flipped = ms2_coeff
    one_accumulator = 0
    zero_accumulator = 0
    for count in np.arange(0, W):
        if state & 1 == 1:
            one_accumulator = one_accumulator + ms2_coeff_flipped[0, count]
        else:
            zero_accumulator = zero_accumulator + ms2_coeff_flipped[0, count]
        state = state >> 1
    return_list = []
    return_list.append(one_accumulator)
    return_list.append(zero_accumulator)
    return return_list

@lru_cache(maxsize=65536)
def _cached_adjusted(state,K,W,coefficient_values):
    return tuple(_get_adjusted_uncached(state,K,W,np.array(coefficient_values).reshape(1,W)))

def get_adjusted(state,K,W,ms2_coeff):
    return _cached_adjusted(int(state),int(K),int(W),tuple(np.asarray(ms2_coeff).ravel()))

def calcObservationLikelihood(lambda_logF, noise_tempF, dataF, veef, INPUT_STATE, K, W, ms2_coeff_flipped):
    ms2_coeff = ms2_coeff_flipped
    adjusted_list = get_adjusted(INPUT_STATE, K, W, ms2_coeff)
    eta = 0.5 * (lambda_logF - np.log(2 * np.pi)) - 0.5 * (1 / noise_tempF ** 2) * (dataF - (adjusted_list[1] * veef[0, 0] + adjusted_list[0] * veef[1, 0])) ** 2
    return eta

def _compute_dynamic_F_uncached(state, length, W, K, ms2_coeff_flipped, count_reduction_manual):
    ms2_coeff = ms2_coeff_flipped
    trace_length = length
    state_flipped = K ** W - state - 1
    adjust = get_adjusted(state_flipped, K, W, ms2_coeff)
    adjust_ones = adjust[0]
    adjust_zeros = adjust[1]
    F1_log = np.log(adjust_ones)
    F0_log = np.log(adjust_zeros)
    log_f0_terms = np.zeros((1, trace_length))
    for i in np.arange(0, trace_length):
        log_f0_terms[0, i] = F0_log
    log_f1_terms_saved = np.zeros((1, trace_length))
    for i in np.arange(0, trace_length):
        log_f1_terms_saved[0, i] = F1_log
    log_F_terms = []
    log_F_terms.append(log_f1_terms_saved)
    log_F_terms.append(log_f0_terms)
    return log_F_terms

@lru_cache(maxsize=4096)
def _cached_dynamic_F(state,length,W,K,coefficient_values):
    # A deterministic state/response operator, never a reconstruction cache.
    # The corrected prehistory-consistent basis is independent of time/data.
    a=np.array(coefficient_values).reshape(1,W)
    out=_compute_dynamic_F_uncached(state,length,W,K,a,None)
    for value in out:value.setflags(write=False)
    return tuple(out)

def compute_dynamic_F(state,length,W,K,ms2_coeff_flipped,count_reduction_manual):
    return _cached_dynamic_F(int(state),int(length),int(W),int(K),tuple(np.asarray(ms2_coeff_flipped).ravel()))

def logsumexp_numba(X):
    r = 0.0
    for x in X:
        r += np.exp(x)
    return np.log(r)

def log_sum_exp(arr, signs):
    arr_max = np.max(arr[:, :])
    term2_array = np.multiply(signs, np.exp(arr - arr_max))
    term2 = np.sum(term2_array)
    logsum = np.array([arr_max + np.log(np.abs(term2)), np.sign(term2)])
    return logsum

def v_log_solve(m_lg, m_sgn, b_lg, b_sgn):

    def sub2ind(array_shape, rows, cols):
        return rows * array_shape[1] + cols

    def permutation_parity(lst):
        I = scipy.sparse.eye(len(lst)).toarray()
        parity = np.linalg.det(I[:, lst - 1])
        return parity

    def log_sum_exp(arr, signs):
        arr_max = np.max(arr[:, :])
        term2_array = np.multiply(signs, np.exp(arr - arr_max))
        term2 = np.sum(np.ravel(term2_array))
        logsum = np.array([arr_max + np.log(np.abs(term2)), np.sign(term2)])
        return logsum

    def find_perms(m_lg):
        n = np.size(m_lg, 0)
        perm_input = np.arange(1, n + 1)
        perm_list = list(itertools.permutations(perm_input))
        perm_list2 = np.array(perm_list)
        perm_num = np.size(perm_list2, 0)
        return (perm_num, perm_list2)

    def log_determinant(m_lg, m_sgn):
        perm_answer = find_perms(m_lg)
        perm_num = perm_answer[0]
        perm_list2 = perm_answer[1]
        logs = np.zeros((perm_num, 1))
        signs = np.zeros((perm_num, 1))
        for i in np.arange(0, perm_num):
            rows = np.arange(0, perm_num)
            cols = perm_list2[i, :]
            ind1 = sub2ind(np.array([2, 2]), rows[0], cols[0])
            ind2 = sub2ind(np.array([2, 2]), rows[1], cols[1])
            raveled = np.ravel(m_sgn, order='C')
            raveled = np.expand_dims(raveled, axis=1)
            signs[i, 0] = permutation_parity(cols) * np.prod(np.concatenate((raveled[ind1 - 1,], raveled[ind2 - 1,]), axis=0))
            raveled2 = np.ravel(m_lg, order='C')
            raveled2 = np.expand_dims(raveled2, axis=1)
            logs[i, 0] = np.sum(np.concatenate((raveled2[ind1 - 1,], raveled2[ind2 - 1,]), axis=0), axis=0)
        log_det = log_sum_exp(logs, signs)[0]
        det_m = log_det
        return (det_m, np.sign(det_m))
    n = np.size(m_lg, 0)
    v_lgs = np.zeros((n, 1))
    v_sgns = np.zeros((n, 1))
    det_m = log_determinant(m_lg, m_sgn)
    for j in np.arange(0, n):
        m_log_j = m_lg.copy()
        m_log_j[:, j] = b_lg
        m_sgn_j = m_sgn
        m_sgn_j[:, j] = b_sgn
        det_j = log_determinant(m_log_j, m_sgn_j)
        v_lgs[j,] = det_j[0] - det_m[0]
        v_sgns[j,] = det_j[1] * det_m[1]
    v_sln = np.concatenate((np.transpose(v_lgs), np.transpose(v_sgns)), axis=0)
    return v_sln
