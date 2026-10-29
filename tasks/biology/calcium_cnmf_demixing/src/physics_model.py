# Derived numerical mathematics: attribution and complete license texts in test/validation.md.


import numpy as np

def fluorescence_forward(A,C,b,f):
    """Return pixels-by-time intensity A@C+b@f in native fluorescence counts."""
    return A@C+b@f

def fluorescence_adjoint(residual,C,f):
    """Spatial adjoint gradients of half squared residual loss."""
    return residual@C.T,residual@f.T
import scipy, scipy.signal, scipy.linalg

def get_noise_fft(Y, noise_range=[0.25, 0.5], noise_method='logmexp', max_num_samples_fft=3072, opencv=True):
    T = Y.shape[-1]
    if T > max_num_samples_fft:
        Y = np.concatenate((Y[..., 1:max_num_samples_fft // 3 + 1], Y[..., int(T // 2 - max_num_samples_fft / 3 / 2):int(T // 2 + max_num_samples_fft / 3 / 2)], Y[..., -max_num_samples_fft // 3:]), axis=-1)
        T = Y.shape[-1]
    ff = np.arange(0, 0.5 + 1.0 / T, 1.0 / T)
    ind1 = ff > noise_range[0]
    ind2 = ff <= noise_range[1]
    ind = np.logical_and(ind1, ind2)
    if Y.ndim > 1:
        if opencv:
            import cv2
            try:
                cv2.setNumThreads(0)
            except:
                pass
            psdx_list = []
            for y in Y.reshape(-1, T):
                dft = cv2.dft(y, flags=cv2.DFT_COMPLEX_OUTPUT).squeeze()[:len(ind)][ind]
                psdx_list.append(np.sum(1.0 / T * dft * dft, 1))
            psdx = np.reshape(psdx_list, Y.shape[:-1] + (-1,))
        else:
            xdft = np.fft.rfft(Y.astype(np.float64, copy=False), axis=-1)
            xdft = xdft[..., ind[:xdft.shape[-1]]]
            psdx = 1.0 / T * abs(xdft) ** 2
        psdx *= 2
        sn = mean_psd(psdx, method=noise_method)
    else:
        xdft = np.fliplr(np.fft.rfft(Y.astype(np.float64, copy=False)))
        psdx = 1.0 / T * xdft ** 2
        psdx[1:] *= 2
        sn = mean_psd(psdx[ind[:psdx.shape[0]]], method=noise_method)
    return (sn, psdx)

def mean_psd(y, method='logmexp'):
    if method == 'mean':
        mp = np.sqrt(np.mean(y / 2, axis=-1))
    elif method == 'median':
        mp = np.sqrt(np.median(y / 2, axis=-1))
    else:
        mp = np.log((y + 1e-10) / 2)
        mp = np.mean(mp, axis=-1)
        mp = np.exp(mp)
        mp = np.sqrt(mp)
    return mp

def estimate_parameters(fluor, p=2, sn=None, g=None, range_ff=[0.25, 0.5], method='logmexp', lags=5, fudge_factor=1.0):
    if sn is None:
        sn = GetSn(fluor, range_ff, method)
    if g is None:
        if p == 0:
            g = np.array(0)
        else:
            g = estimate_time_constant(fluor, p, sn, lags, fudge_factor)
    return (g, sn)

def estimate_time_constant(fluor, p=2, sn=None, lags=5, fudge_factor=1.0):
    if sn is None:
        sn = GetSn(fluor)
    lags += p
    xc = axcov(fluor, lags)
    xc = xc[:, np.newaxis]
    A = scipy.linalg.toeplitz(c=np.ravel(xc[lags + np.arange(lags)]), r=np.ravel(xc[lags + np.arange(p)])) - float(sn) ** 2 * np.eye(lags, p)
    g = np.linalg.lstsq(A, xc[lags + 1:], rcond=None)[0]
    gr = np.roots(np.concatenate([np.array([1]), -g.flatten()]))
    gr = (gr + gr.conjugate()) / 2.0
    np.random.seed(45)
    gr[gr > 1] = 0.95 + np.random.normal(0, 0.01, np.sum(gr > 1))
    gr[gr < 0] = 0.15 + np.random.normal(0, 0.01, np.sum(gr < 0))
    g = np.poly(fudge_factor * gr)
    g = -g[1:]
    return g.flatten()

def GetSn(fluor, range_ff=[0.25, 0.5], method='logmexp'):
    ff, Pxx = scipy.signal.welch(fluor)
    ind1 = ff > range_ff[0]
    ind2 = ff < range_ff[1]
    ind = np.logical_and(ind1, ind2)
    Pxx_ind = Pxx[ind]
    if method == 'mean':
        return np.sqrt(np.mean(Pxx_ind / 2))
    elif method == 'median':
        return np.sqrt(np.median(Pxx_ind / 2))
    elif method == 'logmexp':
        return np.sqrt(np.exp(np.mean(np.log(Pxx_ind / 2))))
    else:
        raise Exception('Invalid method requested for GetSn')

def update_order_greedy(A, flag_AA=True):
    K = A.shape[-1]
    parllcomp: list = []
    for i in range(K):
        new_list = True
        for ls in parllcomp:
            if flag_AA:
                if A[i, ls].nnz == 0:
                    ls.append(i)
                    new_list = False
                    break
            elif A[:, i].T.dot(A[:, ls]).nnz == 0:
                ls.append(i)
                new_list = False
                break
        if new_list:
            parllcomp.append([i])
    len_parrllcomp = [len(ls) for ls in parllcomp]
    return (parllcomp, len_parrllcomp)

def axcov(data, maxlag=5):
    data = data - np.mean(data)
    T = len(data)
    bins = np.size(data)
    xcov = np.fft.fft(data.astype(np.float64, copy=False), np.power(2, nextpow2(2 * bins - 1)))
    xcov = np.fft.ifft(np.square(np.abs(xcov)))
    xcov = np.concatenate([xcov[np.arange(xcov.size - maxlag, xcov.size)], xcov[np.arange(0, maxlag + 1)]])
    return np.real(xcov / T)

def nextpow2(value):
    exponent = 0
    avalue = np.abs(value)
    while avalue > np.power(2, exponent):
        exponent += 1
    return exponent

def blocked_projection(Y, B, transpose=False):
    """Match oracle streaming casts and its sequential float32 block accumulation."""
    B = B.astype(np.float32)
    d, T = Y.shape
    block = max(1, d // 19)
    out = np.zeros((T if transpose else d, B.shape[-1]), dtype=np.float32)
    for start in range(0, d, block):
        ids = np.arange(start, min(start + block, d))
        if transpose:
            value = (B.T.tocsc()[:, ids] @ Y[ids]).T if scipy.sparse.issparse(B) else Y[ids].T @ B[ids]
            out = out + value.astype(np.float32)
        else:
            value = (B.T @ Y[ids].T).T if scipy.sparse.issparse(B) else Y[ids] @ B
            out[ids] = value.astype(np.float32)
    return out
