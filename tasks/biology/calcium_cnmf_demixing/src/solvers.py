# Derived numerical mathematics: attribution and complete license texts in test/validation.md.
from .physics_model import get_noise_fft,estimate_parameters,update_order_greedy,blocked_projection

def reconstruct(movie,progress=None):
    """Reconstruct the complete movie and return submission and review arrays."""
    from .preprocessing import motion_correct_movie
    import time
    start=time.perf_counter();np.random.seed(0)
    motion=motion_correct_movie(movie,progress=None)
    r=fit(motion['corrected'],progress=(lambda *args:None) if progress is None else progress)
    accepted,snr,rv=quality(motion['corrected'],r)
    A=r['A'][:,accepted]
    output={
        'reconstruction':A.toarray().T.reshape(len(accepted),170,170,order='F')[None].astype(np.float64),
        'fluorescence':r['C'][accepted][None].astype(np.float64),
        'innovations':r['S'][accepted][None].astype(np.float64),
        'dff':dff(A,r['b'],r['C'][accepted],r['f'],r['YrA'][accepted])[None].astype(np.float64),
        'background_spatial':r['b'].T.reshape(2,170,170,order='F')[None].astype(np.float64),
        'background_temporal':r['f'][None].astype(np.float64),
        'rigid_shifts':motion['shifts_rigid'][None].astype(np.float64),
        'nonrigid_shifts':motion['shifts'][None].astype(np.float64),
    }
    review={'all_A':r['A'].toarray(),'all_C':r['C'],'all_YrA':r['YrA'],'b':r['b'],'f':r['f'],'corrected_movie':motion['corrected'],'accepted':accepted,'SNR':snr,'spatial_correlation':rv,'mean_corrected':motion['corrected'].mean(0),'seconds':time.perf_counter()-start}
    return output,review

"""Explicit NNDSVDar and cyclic projected-coordinate background NMF.
Derived from scikit-learn's BSD-3 implementation, selected unregularized branch.
"""
import numpy as np
from scipy import linalg

def randomized_svd(X, k):
    trans = X.shape[0] < X.shape[1]
    M = X.T if trans else X
    Q = np.random.normal(size=(M.shape[1], k + 10)).astype(X.dtype, copy=False)
    for _ in range(7 if k < 0.1 * min(X.shape) else 4):
        Q, _ = linalg.lu(M @ Q, permute_l=True, check_finite=False)
        Q, _ = linalg.lu(M.T @ Q, permute_l=True, check_finite=False)
    Q, _ = linalg.qr(M @ Q, mode='economic', check_finite=False)
    Uh, s, V = linalg.svd(Q.T @ M, full_matrices=False, lapack_driver='gesdd')
    U = Q @ Uh
    if trans:
        idx = np.argmax(np.abs(V), axis=1)
        sgn = np.sign(V[np.arange(V.shape[0]), idx])
    else:
        idx = np.argmax(np.abs(U), axis=0)
        sgn = np.sign(U[idx, np.arange(U.shape[1])])
    U *= sgn
    V *= sgn[:, None]
    return (V[:k].T, s[:k], U[:, :k].T) if trans else (U[:, :k], s[:k], V[:k])

def initialize_background(X, k, fill_seed=None):
    U, S, V = randomized_svd(X, k)
    W = np.zeros_like(U)
    H = np.zeros_like(V)
    W[:, 0] = np.sqrt(S[0]) * np.abs(U[:, 0])
    H[0] = np.sqrt(S[0]) * np.abs(V[0])
    for j in range(1, k):
        x, y = (U[:, j], V[j])
        xp, yp = (np.maximum(x, 0), np.maximum(y, 0))
        xn, yn = (np.abs(np.minimum(x, 0)), np.abs(np.minimum(y, 0)))
        nxp, nyp = (linalg.norm(xp), linalg.norm(yp))
        nxn, nyn = (linalg.norm(xn), linalg.norm(yn))
        if nxp * nyp > nxn * nyn:
            u, v, sigma = (xp / nxp, yp / nyp, nxp * nyp)
        else:
            u, v, sigma = (xn / nxn, yn / nyn, nxn * nyn)
        lam = np.sqrt(S[j] * sigma)
        W[:, j] = lam * u
        H[j] = lam * v
    W[W < 1e-06] = 0
    H[H < 1e-06] = 0
    avg = X.mean()
    if fill_seed is not None:
        np.random.seed(fill_seed)
    W[W == 0] = abs(avg * np.random.standard_normal(np.count_nonzero(W == 0)) / 100)
    H[H == 0] = abs(avg * np.random.standard_normal(np.count_nonzero(H == 0)) / 100)
    return (W, H)

def coordinate_update(X, W, H):
    gram = H.T @ H
    cross = X @ H
    violation = 0.0
    for k in range(W.shape[1]):
        grad = W @ gram[:, k] - cross[:, k]
        violation += np.abs(np.where(W[:, k] == 0, np.minimum(0, grad), grad)).sum()
        if gram[k, k] != 0:
            W[:, k] = np.maximum(W[:, k] - grad / gram[k, k], 0)
    return float(violation)

class NMF:

    def __init__(self, n_components=2, max_iter=200, init='nndsvdar', tol=0.0001, random_state=None):
        if init != 'nndsvdar':
            raise ValueError('The selected method uses NNDSVDar')
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit_transform(self, X):
        state = np.random.get_state()
        if self.random_state is not None:
            np.random.seed(self.random_state)
        W, H = initialize_background(X, self.n_components, fill_seed=self.random_state)
        if self.random_state is not None:
            np.random.set_state(state)
        W = np.ascontiguousarray(W)
        Ht = np.ascontiguousarray(H.T)
        for it in range(1, self.max_iter + 1):
            violation = coordinate_update(X, W, Ht) + coordinate_update(X.T, Ht, W)
            if it == 1:
                initial = violation
            if initial == 0 or violation / initial <= self.tol:
                break
        self.n_iter_ = it
        self.components_ = Ht.T
        return W

    def fit(self, X):
        self.fit_transform(X)
        return self

import logging, sys, warnings
import numpy as np
import scipy, scipy.ndimage, scipy.signal, scipy.sparse as spr
import cv2

def downscale(Y, ds, opencv=True):
    movie = Y.transpose(2, 0, 1)
    t, h, w = movie.shape
    movie = np.asarray([cv2.resize(frame, (int(w / ds[1]), int(h / ds[0])), fx=1 / ds[0], fy=1 / ds[1], interpolation=cv2.INTER_AREA) for frame in movie])
    t, h, w = movie.shape
    movie = cv2.resize(movie.reshape(t, h * w), (h * w, int(t / ds[2])), fx=1, fy=1 / ds[2], interpolation=cv2.INTER_AREA).reshape(max(1, int(t / ds[2])), h, w)
    return movie.transpose(1, 2, 0)

def resize(Y, size, interpolation=cv2.INTER_LINEAR):
    if Y.ndim == 2:
        return cv2.resize(Y, tuple(size[::-1]), interpolation=interpolation)
    elif Y.ndim == 3:
        if np.isfortran(Y):
            return cv2.resize(np.array([cv2.resize(y, size[:2], interpolation=interpolation) for y in Y.T]).T.reshape((-1, Y.shape[-1]), order='F'), (size[-1], np.prod(size[:2])), interpolation=interpolation).reshape(size, order='F')
        else:
            return np.array([cv2.resize(y, size[:0:-1], interpolation=interpolation) for y in cv2.resize(Y.reshape((len(Y), -1), order='F'), (np.prod(Y.shape[1:]), size[0]), interpolation=interpolation).reshape((size[0],) + Y.shape[1:], order='F')])
    else:
        raise NotImplementedError

def initialize_components(Y, K=4, gSig=(4, 4), gSiz=(9, 9), ssub=2, tsub=2, nIter=5, maxIter=5, nb=2, kernel=None, normalize_init=True, img=None, rolling_sum=True, rolling_length=100, greedyroi_nmf_init_method='nndsvdar', greedyroi_nmf_max_iter=200):
    logger = logging.getLogger('caiman')
    method = 'greedy_roi'
    if gSiz is None:
        gSiz = 2 * (np.asarray(gSig) + 0.5).astype(int) + 1
    d, T = (Y.shape[:-1], Y.shape[-1])
    gSig = np.asarray(gSig, dtype=float) / ssub
    gSiz = np.round(np.asarray(gSiz) / ssub).astype(int)
    if normalize_init:
        logger.info('Variance Normalization')
        if img is None:
            img = np.mean(Y, axis=-1)
            img += np.median(img)
            img += np.finfo(np.float32).eps
        Y = Y / np.reshape(img, d + (-1,), order='F')
    else:
        Y = np.array(Y)
    if ssub != 1 or tsub != 1:
        logger.info('Spatial/Temporal downsampling 2-photon')
        Y_ds = downscale(Y, tuple([ssub] * len(d) + [tsub]), opencv=True)
    else:
        Y_ds = Y
    ds = Y_ds.shape[:-1]
    if nb > min(np.prod(ds), Y_ds.shape[-1]):
        nb = -1
    logger.info(f"Roi Initialization with method {'greedy_roi'}...")
    Ain, Cin, _, b_in, f_in = greedyROI(Y_ds, nr=K, gSig=gSig, gSiz=gSiz, nIter=nIter, kernel=kernel, nb=nb, rolling_sum=rolling_sum, rolling_length=rolling_length, seed_method='auto', nmf_overrides={'init_method': greedyroi_nmf_init_method, 'max_iter': greedyroi_nmf_max_iter})
    logger.info('Refining Components using HALS NMF iterations')
    Ain, Cin, b_in, f_in = hals(Y_ds, Ain, Cin, b_in, f_in, maxIter=maxIter)
    K = Ain.shape[-1]
    if Ain.size > 0 and (not False) and (ssub != 1):
        Ain = np.reshape(Ain, ds + (K,), order='F')
        if len(ds) == 2:
            Ain = resize(Ain, d + (K,))
        else:
            Ain = np.reshape([resize(a, d[1:] + (K,)) for a in Ain], (ds[0], d[1] * d[2], K), order='F')
            Ain = resize(Ain, (d[0], d[1] * d[2], K))
        Ain = np.reshape(Ain, (np.prod(d), K), order='F')
    sparse_b = spr.issparse(b_in)
    if (nb > 0 or nb == -1) and (ssub != 1 or tsub != 1):
        b_in = np.reshape(b_in.toarray() if sparse_b else b_in, ds + (-1,), order='F')
        if len(ds) == 2:
            b_in = resize(b_in, d + (b_in.shape[-1],))
        else:
            b_in = np.reshape([resize(b, d[1:] + (b_in.shape[-1],)) for b in b_in], (ds[0], d[1] * d[2], -1), order='F')
            b_in = resize(b_in, (d[0], d[1] * d[2], b_in.shape[-1]))
        b_in = np.reshape(b_in, (np.prod(d), -1), order='F')
        if sparse_b:
            b_in = spr.csc_matrix(b_in)
        f_in = resize(np.atleast_2d(f_in), [b_in.shape[-1], T])
    if Ain.size > 0:
        Cin = resize(Cin, [K, T])
        center = np.asarray([scipy.ndimage.center_of_mass(a.reshape(d, order='F')) for a in Ain.T])
    else:
        Cin = np.empty((K, T), dtype=np.float32)
        center = []
    if normalize_init:
        if Ain.size > 0:
            Ain = Ain * np.reshape(img, (np.prod(d), -1), order='F')
        if sparse_b:
            b_in = spr.diags(img.ravel(order='F')).dot(b_in)
        else:
            b_in = b_in * np.reshape(img, (np.prod(d), -1), order='F')
    return (scipy.sparse.csc_matrix(Ain), Cin, b_in, f_in, center)

def greedyROI(Y, nr=30, gSig=[5, 5], gSiz=[11, 11], nIter=5, kernel=None, nb=1, rolling_sum=False, rolling_length: int=100, seed_method: str='auto', nmf_overrides: dict=None):
    logger = logging.getLogger('caiman')
    logger.info(f'Greedy initialization of spatial and temporal components using spatial Gaussian filtering. Params={nmf_overrides}')
    d = Y.shape
    Y[np.isnan(Y)] = 0
    med = np.median(Y, axis=-1)
    Y = Y - med[..., np.newaxis]
    gHalf = np.array(gSiz) // 2
    gSiz = 2 * gHalf + 1
    A = np.zeros((np.prod(d[0:-1]), nr), dtype=np.float32)
    C = np.zeros((nr, d[-1]), dtype=np.float32)
    center = np.zeros((nr, Y.ndim - 1), dtype='uint16')
    rho = imblur(Y, sig=gSig, siz=gSiz, nDimBlur=Y.ndim - 1, kernel=kernel)
    if rolling_sum:
        logger.info('Using rolling sum for initialization (RollingGreedyROI)')
        rolling_filter = np.ones(rolling_length, dtype=np.float32) / rolling_length
        rho_s = scipy.signal.lfilter(rolling_filter, 1.0, rho ** 2)
        v = np.amax(rho_s, axis=-1)
    else:
        logger.info('Using total sum for initialization (GreedyROI)')
        v = np.sum(rho ** 2, axis=-1)
    for k in range(nr):
        ind = np.argmax(v)
        ij = np.unravel_index(ind, d[0:-1])
        for c, i in enumerate(ij):
            center[k, c] = i
        ijSig = [[np.maximum(ij[c] - gHalf[c], 0), np.minimum(ij[c] + gHalf[c] + 1, d[c])] for c in range(len(ij))]
        dataTemp = np.array(Y[tuple([slice(*a) for a in ijSig])].copy(), dtype=np.float32)
        traceTemp = np.array(np.squeeze(rho[ij]), dtype=np.float32)
        coef, score = finetune(dataTemp, traceTemp, nIter=nIter)
        C[k, :] = np.squeeze(score)
        dataSig = coef[..., np.newaxis] * score.reshape([1] * (Y.ndim - 1) + [-1])
        xySig = np.meshgrid(*[np.arange(s[0], s[1]) for s in ijSig], indexing='xy')
        arr = np.array([np.reshape(s, (1, np.size(s)), order='F').squeeze() for s in xySig], dtype=int)
        indices = np.ravel_multi_index(arr, d[0:-1], order='F')
        A[indices, k] = np.reshape(coef, (1, np.size(coef)), order='C').squeeze()
        Y[tuple([slice(*a) for a in ijSig])] -= dataSig.copy()
        if k < nr - 1 or 'auto'.lower() != 'auto':
            Mod = [[np.maximum(ij[c] - 2 * gHalf[c], 0), np.minimum(ij[c] + 2 * gHalf[c] + 1, d[c])] for c in range(len(ij))]
            ModLen = [m[1] - m[0] for m in Mod]
            Lag = [ijSig[c] - Mod[c][0] for c in range(len(ij))]
            dataTemp = np.zeros(ModLen)
            dataTemp[tuple([slice(*a) for a in Lag])] = coef
            dataTemp = imblur(dataTemp[..., np.newaxis], sig=gSig, siz=gSiz, kernel=kernel)
            temp = dataTemp * score.reshape([1] * (Y.ndim - 1) + [-1])
            rho[tuple([slice(*a) for a in Mod])] -= temp.copy()
            if rolling_sum:
                rho_filt = scipy.signal.lfilter(rolling_filter, 1.0, rho[tuple([slice(*a) for a in Mod])] ** 2)
                v[tuple([slice(*a) for a in Mod])] = np.amax(rho_filt, axis=-1)
            else:
                v[tuple([slice(*a) for a in Mod])] = np.sum(rho[tuple([slice(*a) for a in Mod])] ** 2, axis=-1)
    center = center.tolist()
    res = np.reshape(Y, (np.prod(d[0:-1]), d[-1]), order='F') + med.flatten(order='F')[:, None]
    nmf_init_method = 'nndsvdar'
    nmf_max_iter = 200
    if nmf_overrides is not None and len(nmf_overrides) > 1:
        if 'max_iter' in nmf_overrides:
            nmf_max_iter = int(nmf_overrides['max_iter'])
        if 'init_method' in nmf_overrides:
            nmf_init_method = nmf_overrides['init_method']
    model = NMF(n_components=nb, max_iter=nmf_max_iter, init=nmf_init_method)
    b_in = model.fit_transform(np.maximum(res, np.float64(0))).astype(np.float32)
    f_in = model.components_.astype(np.float32)
    return (A, C, np.array(center, dtype='uint16'), b_in, f_in)

def finetune(Y, cin, nIter=5):
    debug_ = False
    for _ in range(nIter):
        a = np.maximum(np.dot(Y, cin), 0)
        a = a / np.sqrt(np.sum(a ** 2) + np.finfo(np.float32).eps)
        cin = np.sum(Y * a[..., np.newaxis], tuple(np.arange(Y.ndim - 1)))
    return (a, cin)

def imblur(Y, sig=5, siz=11, nDimBlur=None, kernel=None, opencv=True):
    if kernel is None:
        if nDimBlur is None:
            nDimBlur = Y.ndim - 1
        else:
            nDimBlur = np.min((Y.ndim, nDimBlur))
        if np.isscalar(sig):
            sig = sig * np.ones(nDimBlur)
        if np.isscalar(siz):
            siz = siz * np.ones(nDimBlur)
        X = Y.copy()
        if opencv and nDimBlur == 2:
            if X.ndim > 2:
                for frame in range(X.shape[-1]):
                    if sys.version_info >= (3, 0):
                        X[:, :, frame] = cv2.GaussianBlur(X[:, :, frame], tuple(siz), sig[0], None, sig[1], cv2.BORDER_CONSTANT)
                    else:
                        X[:, :, frame] = cv2.GaussianBlur(X[:, :, frame], tuple(siz), sig[0], sig[1], cv2.BORDER_CONSTANT, 0)
            elif sys.version_info >= (3, 0):
                X = cv2.GaussianBlur(X, tuple(siz), sig[0], None, sig[1], cv2.BORDER_CONSTANT)
            else:
                X = cv2.GaussianBlur(X, tuple(siz), sig[0], sig[1], cv2.BORDER_CONSTANT, 0)
        else:
            for i in range(nDimBlur):
                h = np.exp(-np.arange(-np.floor(siz[i] / 2), np.floor(siz[i] / 2) + 1) ** 2 / (2 * sig[i] ** 2))
                h /= np.sqrt(h.dot(h))
                shape = [1] * len(Y.shape)
                shape[i] = -1
                X = scipy.ndimage.correlate(X, h.reshape(shape), mode='constant')
    else:
        X = scipy.ndimage.correlate(Y, kernel[..., np.newaxis], mode='constant')
    return X

def hals(Y, A, C, b, f, bSiz=3, maxIter=5):
    dims, T = (Y.shape[:-1], Y.shape[-1])
    K = A.shape[1]
    nb = b.shape[1]
    if bSiz is not None:
        if isinstance(bSiz, (int, float)):
            bSiz = [bSiz] * len(dims)
        ind_A = scipy.ndimage.uniform_filter(np.reshape(A, dims + (K,), order='F'), size=bSiz + [0])
        ind_A = np.reshape(ind_A > 1e-10, (np.prod(dims), K), order='F')
    else:
        ind_A = A > 1e-10
    ind_A = spr.csc_matrix(ind_A)

    def HALS4activity(Yr, A, C, iters=2):
        U = A.T.dot(Yr)
        V = A.T.dot(A) + np.finfo(A.dtype).eps
        for _ in range(iters):
            for m in range(len(U)):
                C[m] = np.clip(C[m] + (U[m] - V[m].dot(C)) / V[m, m], 0, np.inf)
        return C

    def HALS4shape(Yr, A, C, iters=2):
        U = C.dot(Yr.T)
        V = C.dot(C.T) + np.finfo(C.dtype).eps
        for _ in range(iters):
            for m in range(K):
                ind_pixels = np.squeeze(ind_A[:, m].toarray())
                A[ind_pixels, m] = np.clip(A[ind_pixels, m] + (U[m, ind_pixels] - V[m].dot(A[ind_pixels].T)) / V[m, m], 0, np.inf)
            for m in range(nb):
                A[:, K + m] = np.clip(A[:, K + m] + (U[K + m] - V[K + m].dot(A.T)) / V[K + m, K + m], 0, np.inf)
        return A
    Ab = np.c_[A, b]
    Cf = np.r_[C, f.reshape(nb, -1)]
    for _ in range(maxIter):
        Cf = HALS4activity(np.reshape(Y, (np.prod(dims), T), order='F'), Ab, Cf)
        Ab = HALS4shape(np.reshape(Y, (np.prod(dims), T), order='F'), Ab, Cf)
    return (Ab[:, :-nb], Cf[:-nb], Ab[:, -nb:], Cf[-nb:].reshape(nb, -1))

"""Positive LARS homotopy with the upstream finite-path interpolation convention."""
import numpy as np

def positive_lars(X, y, alpha):
    scale = np.std(X, axis=0)
    scale[scale < 10 * np.finfo(float).eps] = 1
    X = X / scale
    X -= X.mean(0)
    y = y - y.mean()
    G = X.T @ X
    q = X.T @ y
    coef = np.zeros(X.shape[1])
    active = []
    excluded = set()
    prev = coef.copy()
    prev_alpha = 0.0
    drop = False
    for it in range(501):
        inactive = [j for j in range(len(coef)) if j not in active and j not in excluded]
        corr = q - G @ coef
        j = max(inactive, key=lambda j: corr[j]) if inactive else None
        C = float(corr[j]) if j is not None else 0.0
        current_alpha = C / len(y)
        if current_alpha <= alpha + np.finfo(np.float32).eps:
            if abs(current_alpha - alpha) > np.finfo(np.float32).eps and it > 0:
                ss = (prev_alpha - alpha) / (prev_alpha - current_alpha)
                coef = prev + ss * (coef - prev)
            break
        if it >= 500 or len(active) >= len(coef):
            break
        if not drop:
            active.append(j)
            inactive.remove(j)
        Ga = G[np.ix_(active, active)]
        if np.linalg.cond(Ga) > 1000000000000000.0:
            excluded.add(active.pop())
            continue
        v = np.linalg.solve(Ga, np.ones(len(active)))
        AA = 1 / np.sqrt(v.sum())
        w = v * AA
        a = G[:, active] @ w
        a = np.around(a, decimals=15)
        candidates = [(C - corr[k]) / (AA - a[k] + np.finfo(np.float32).tiny) for k in inactive]
        gamma = min([x for x in candidates if x > 0] + [C / AA])
        dropped = None
        for k, wk in zip(active, w):
            step = -coef[k] / (wk + np.finfo(np.float32).tiny)
            if 0 < step < gamma:
                gamma = step
                dropped = k
        prev = coef.copy()
        prev_alpha = current_alpha
        coef[active] += gamma * w
        drop = dropped is not None
        if drop:
            coef[dropped] = 0
            active.remove(dropped)
    return coef

import numpy as np
from scipy import sparse
from scipy.sparse import csc_matrix, csr_matrix
from scipy.ndimage import grey_dilation, iterate_structure, generate_binary_structure, median_filter, binary_closing, label

def HALS4shape_bckgrnd(Y_resf, B, F, ind_B, iters=5):
    K = B.shape[-1]
    U = Y_resf.T
    V = F.dot(F.T)
    for _ in range(iters):
        for m in range(K):
            ind_pixels = ind_B[m]
            B[ind_pixels, m] = np.clip(B[ind_pixels, m] + (U[m, ind_pixels] - V[m].dot(B[ind_pixels].T)) / V[m, m], 0, np.inf)
    return B

def threshold_components_parallel(pars):
    A_i, i, dims, medw, d, thr_method, se, ss, maxthr, nrgthr, extract_cc = pars
    A_i = A_i.toarray()
    A_temp = np.reshape(A_i, dims[::-1])
    A_temp = median_filter(A_temp, medw)
    if thr_method == 'max':
        BW = A_temp > maxthr * np.max(A_temp)
    elif thr_method == 'nrg':
        Asor = np.sort(np.squeeze(np.reshape(A_temp, (d, 1))))[::-1]
        temp = np.cumsum(Asor ** 2)
        ff = np.squeeze(np.where(temp < nrgthr * temp[-1]))
        if ff.size > 0:
            ind = ff if ff.ndim == 0 else ff[-1]
            A_temp[A_temp < Asor[ind]] = 0
            BW = A_temp >= Asor[ind]
        else:
            BW = np.zeros_like(A_temp)
    Ath = np.squeeze(np.reshape(A_temp, (d, 1)))
    Ath2 = np.zeros(d)
    BW = binary_closing(BW.astype(int), structure=se.T)
    if BW.max() == 0:
        return (csr_matrix(Ath2), i)
    if extract_cc:
        labeled_array, num_features = label(BW, structure=ss.T)
        labeled_array = np.squeeze(np.reshape(labeled_array, (d, 1)))
        nrg = np.zeros((num_features, 1))
        for j in range(num_features):
            nrg[j] = np.sum(Ath[labeled_array == j + 1] ** 2)
        indm = np.argmax(nrg)
        Ath2[labeled_array == indm + 1] = Ath[labeled_array == indm + 1]
    else:
        BW = BW.flatten()
        Ath2[BW] = Ath[BW]
    return (csr_matrix(Ath2), i)

def update_spatial(Y, A, C, b, f, sn, dims, streamed=False):
    A = sparse.csc_matrix(A)
    nr = C.shape[0]
    d, T = Y.shape
    combined = sparse.hstack([A, sparse.csc_matrix(b)]).tocsc()
    core = iterate_structure(generate_binary_structure(2, 1), 2).astype(int)
    support = np.empty(combined.shape, dtype=bool)
    for k in range(combined.shape[1]):
        image = combined[:, k].toarray().reshape(dims[::-1])
        support[:, k] = grey_dilation(image, footprint=core).ravel() > 0
    empty = np.where((C.sum(1) == 0) | np.isnan(C.sum(1)))[0]
    if len(empty):
        keep = np.setdiff1d(np.arange(nr), empty)
        A = A[:, keep]
        C = C[keep]
        support = np.hstack([support[:, keep], support[:, nr:]])
        nr = len(keep)
    C = C.copy()
    norms = np.sqrt((C ** 2).sum(1))
    A = A @ sparse.diags(norms)
    C /= norms[:, None] + np.finfo(np.float32).eps
    Cf = np.vstack([C, f])
    cct = np.diag(C @ C.T)
    rows = []
    cols = []
    values = []
    for px in range(d):
        ix = np.flatnonzero(support[px])
        if len(ix) == 0 or ix.min() >= nr or sn[px] <= 0:
            continue
        neuronal = ix[ix < nr]
        alpha = 0.5 * float(sn[px]) * float(np.sqrt(np.max(cct[neuronal]))) / T
        coef = positive_lars(Cf[ix].T.astype(np.float64), np.asarray(Y[px], dtype=np.float64), alpha)
        good = np.where(coef > 0)[0]
        rows.extend([px] * len(good))
        cols.extend(ix[good])
        values.extend(coef[good])
    spatial = sparse.csc_matrix((values, (rows, cols)), shape=(d, nr + len(f)))
    thresholded = []
    for k in range(spatial.shape[1]):
        a, _ = threshold_components_parallel([spatial[:, k], k, dims, (3, 3), d, 'nrg', np.ones((3, 3), dtype=np.uint8), np.ones((3, 3), dtype=np.uint8), 0.1, 0.9999, True])
        thresholded.append(a.T)
    spatial = sparse.hstack(thresholded).tocsc()
    empty = np.where(np.asarray(spatial.sum(0)).ravel() == 0)[0]
    removed_neurons = empty[empty < nr]
    if len(removed_neurons):
        keep = np.setdiff1d(np.arange(spatial.shape[1]), removed_neurons)
        spatial = spatial[:, keep]
        C = np.delete(C, removed_neurons, axis=0)
        oldnr = nr
        nr -= len(removed_neurons)
        empty = empty - oldnr
    else:
        empty = empty - nr
    removed_background = empty[empty >= 0]
    f = np.delete(f, removed_background, axis=0)
    b = np.delete(b, removed_background, axis=1)
    A = spatial[:, :nr].tocsr()
    residual_projection = (blocked_projection(Y, f.T) if streamed else Y @ f.T) - A @ (C[:nr] @ f.T)
    b = HALS4shape_bckgrnd(residual_projection, b, f, [np.flatnonzero(x) for x in b.T])
    return (A.tocsc(), b, C, f)

import copy
import numpy as np
from math import sqrt, log, fabs
fmax = max

class _Pool:

    def __init__(self):
        self.v = 0
        self.w = 0
        self.t = 0
        self.l = 0

    def __setattr__(self, key, value):
        object.__setattr__(self, key, np.float32(value) if key in ('v', 'w') else int(value))

def constrained_oasisAR1(y, g, sn, optimize_b=False, b_nonneg=True, optimize_g=0, decimate=1, max_iter=5, penalty=1, s_min=-3):
    """ Infer the most likely discretized spike train underlying an AR(1) fluorescence trace

    Solves the noise constrained sparse non-negative deconvolution problem
    min |s|_1 subject to |c-y|^2 = sn^2 T and s_t = c_t-g c_{t-1} >= 0

    Parameters
    ----------
    y : array of float
        One dimensional array containing the fluorescence intensities (with baseline
        already subtracted, if known, see optimize_b) with one entry per time-bin.
    g : float
        Parameter of the AR(1) process that models the fluorescence impulse response.
    sn : float
        Standard deviation of the noise distribution.
    optimize_b : bool, optional, default False
        Optimize baseline if True else it is set to 0, see y.
    b_nonneg: bool, optional, default True
        Enforce strictly non-negative baseline if True.
    optimize_g : int, optional, default 0
        Number of large, isolated events to consider for optimizing g.
        No optimization if optimize_g=0.
    decimate : int, optional, default 1
        Decimation factor for estimating hyper-parameters faster on decimated data.
    max_iter : int, optional, default 5
        Maximal number of iterations.
    penalty : int, optional, default 1
        Sparsity penalty. 1: min |s|_1  0: min |s|_0
    s_min : float, optional, default -3
        Minimal non-zero activity within each bin (minimal 'spike size').
        For negative values the threshold is |s_min| * sn * sqrt(1-g)
        If 0 the threshold is determined automatically such that RSS <= sn^2 T

    Returns
    -------
    c : array of float
        The inferred denoised fluorescence signal at each time-bin.
    s : array of float
        Discretized deconvolved neural activity (spikes).
    b : float
        Fluorescence baseline value.
    g : float
        Parameter of the AR(1) process that models the fluorescence impulse response.
    lam : float
        Sparsity penalty parameter lambda of dual problem.

    References
    ----------
    * Friedrich J and Paninski L, NIPS 2016
    * Friedrich J, Zhou P, and Paninski L, PLOS Computational Biology 2017
    """
    if optimize_g!=0 or decimate!=1 or penalty!=1:
        raise ValueError('This extracted selected AR1 method supports optimize_g=0, decimate=1, penalty=1')
    P = []
    newpool = _Pool()
    T = len(y)
    thresh = sn * sn * T
    h = g ** np.arange(T, dtype=np.float32)
    c = np.empty(T, dtype=np.float32)
    lam = 0

    def oasis1strun(y, g, c):
        P = []
        newpool = _Pool()
        T = len(y)
        newpool.v, newpool.w, newpool.t, newpool.l = (y[0], 1, 0, 1)
        P.append(copy.copy(newpool))
        i = 0
        t = 1
        while t < T:
            newpool.v = y[t]
            newpool.w, newpool.t, newpool.l = (1, t, 1)
            P.append(copy.copy(newpool))
            t += 1
            i += 1
            while i > 0 and P[i - 1].v / P[i - 1].w * g ** P[i - 1].l > P[i].v / P[i].w:
                i -= 1
                P[i].v += P[i + 1].v * g ** P[i].l
                P[i].w += P[i + 1].w * g ** (2 * P[i].l)
                P[i].l += P[i + 1].l
                P.pop()
        c = np.empty(T, dtype=np.float32)
        for j in range(i + 1):
            tmp = fmax(P[j].v, 0) / P[j].w
            for k in range(P[j].l):
                c[k + P[j].t] = tmp
                tmp *= g
        return (c, P)

    def oasis(P, g, c):
        i = 0
        while i < len(P) - 1:
            i += 1
            while i > 0 and P[i - 1].v / P[i - 1].w * g ** P[i - 1].l > P[i].v / P[i].w:
                i -= 1
                P[i].v += P[i + 1].v * g ** P[i].l
                P[i].w += P[i + 1].w * g ** (2 * P[i].l)
                P[i].l += P[i + 1].l
                P.pop(i + 1)
        c = np.empty(P[len(P) - 1].t + P[len(P) - 1].l, dtype=np.float32)
        for j in range(i + 1):
            tmp = fmax(P[j].v, 0) / P[j].w
            for k in range(P[j].l):
                c[k + P[j].t] = tmp
                tmp *= g
        return (c, P)
    g_converged = False
    count = 0
    if not optimize_b:
        c, P = oasis1strun(y, g, c)
        tmp = np.empty(T, dtype=np.float32)
        res = y - c
        RSS = res.dot(res)
        b = 0
        while RSS < thresh * (1 - 0.0001) and c.sum() > 1e-09:
            for i in range(len(P)):
                if i == len(P) - 1:
                    aa = 1 / P[i].w
                    for j in range(P[i].l):
                        tmp[P[i].t + j] = aa
                        aa *= g
                else:
                    aa = (1 - g ** P[i].l) / P[i].w
                    for j in range(P[i].l):
                        tmp[P[i].t + j] = aa
                        aa *= g
            aa = tmp.dot(tmp)
            bb = res.dot(tmp)
            cc = RSS - thresh
            dlam = (-bb + sqrt(bb * bb - aa * cc)) / aa
            lam += dlam
            for i in range(len(P) - 1):
                P[i].v -= dlam * (1 - g ** P[i].l)
            P[len(P) - 1].v -= dlam
            c, P = oasis(P, g, c)
            res = y - c
            RSS = res.dot(res)
    else:
        b = np.percentile(y, np.float64(15))
        if b_nonneg:
            b = fmax(b, 0)
        c, P = oasis1strun(y - b, g, c)
        db = fmax(np.mean(y - c), 0 if b_nonneg else -np.inf) - b
        b += db
        lam -= db / (1 - g)
        i = len(P) - 1
        P[i].v -= lam * g ** P[i].l
        c[P[i].t:P[i].t + P[i].l] = fmax(0, P[i].v) / P[i].w * h[:P[i].l]
        res = y - b - c
        RSS = res.dot(res)
        tmp = np.empty(T, dtype=np.float32)
        while fabs(RSS - thresh) > thresh * 0.0001 and c.sum() > 1e-09 and (count < max_iter):
            count += 1
            for i in range(len(P)):
                if i == len(P) - 1:
                    aa = 1 / P[i].w
                    for j in range(P[i].l):
                        tmp[P[i].t + j] = aa
                        aa *= g
                else:
                    aa = (1 - g ** P[i].l) / P[i].w
                    for j in range(P[i].l):
                        tmp[P[i].t + j] = aa
                        aa *= g
            tmp -= 1.0 / T / (1 - g) * float(np.sum([(1 - g ** P[i].l) ** 2 / P[i].w for i in range(len(P))]))
            aa = tmp.dot(tmp)
            bb = res.dot(tmp)
            cc = RSS - thresh
            if bb * bb - aa * cc > 0:
                dphi = (-bb + sqrt(bb * bb - aa * cc)) / aa
            else:
                dphi = -bb / aa
            if b_nonneg:
                dphi = fmax(dphi, -b / (1 - g))
            b += dphi * (1 - g)
            for i in range(len(P)):
                P[i].v -= dphi * (1 - g ** P[i].l)
            c, P = oasis(P, g, c)
            db = fmax(np.mean(y - c), 0 if b_nonneg else -np.inf) - b
            b += db
            dlam = -db / (1 - g)
            lam += dlam
            i = len(P) - 1
            P[i].v -= dlam * g ** P[i].l
            c[P[i].t:P[i].t + P[i].l] = fmax(0, P[i].v) / P[i].w * h[:P[i].l]
            res = y - c - b
            RSS = res.dot(res)
    s = c.copy()
    s[0] = 0
    s[1:] -= g * c[:-1]
    return (c, s, b, g, lam)

"""Local CNMF temporal residual updates, AR estimation, noise-constrained OASIS."""
import numpy as np
import scipy.linalg
from scipy import sparse

def deconvolve_trace(y, p=1):
    g, sn = estimate_parameters(y, p=p, method='mean', lags=5, fudge_factor=0.96)
    if p == 0:
        calcium = np.maximum(y, 0)
        return (calcium, calcium.copy(), 0.0, 0.0, sn, np.array(0), None)
    g0 = np.float32(g[0])
    sn0 = np.float32(sn)
    c, s, b, g0, lam = constrained_oasisAR1(y.astype(np.float32), g0, sn0, optimize_b=True, b_nonneg=False)
    c1 = c[0]
    c -= c1 * g0 ** np.arange(len(y))
    trace = c + b + np.dot(c1, g0 ** np.arange(len(y)))
    return (trace, s, b, c1, sn, np.ravel(g0), lam)

def update_temporal(Y, A, b, C, f, p=1, streamed=False):
    nr = A.shape[1]
    nb = b.shape[1]
    Ab = sparse.hstack([A, b]).tocsc()
    S = np.zeros(C.shape)
    Cin = np.vstack([C, f])
    C = Cin.copy()
    norm = np.asarray(Ab.power(2).sum(0)).ravel() + np.finfo(np.float32).eps
    YA = blocked_projection(Y, Ab.tocsr(), transpose=True) if streamed else (Ab.T @ Y).T
    YA = YA @ sparse.diags(1 / norm)
    AA = (Ab.T @ Ab @ sparse.diags(1 / norm)).tocsr()
    residual = YA - (AA.T @ Cin).T
    groups, _ = update_order_greedy(AA[:nr, :nr])
    baseline = np.repeat(None, nr)
    c1 = np.repeat(None, nr)
    noise = np.repeat(None, nr)
    g = np.repeat(None, nr)
    lam = np.repeat(None, nr)
    for _ in range(2):
        for group in groups:
            ix = np.asarray(group)
            y = residual[:, ix] + Cin[ix].T
            Ct = np.zeros((len(ix), Y.shape[1]))
            St = np.zeros_like(Ct)
            for j, k in enumerate(ix):
                Ct[j], St[j], baseline[k], c1[k], noise[k], g[k], lam[k] = deconvolve_trace(np.asarray(y[:, j]).ravel(), p=p)
            residual -= (AA[ix].T @ (Ct - C[ix])).T
            C[ix] = Ct
            S[ix] = St
        for k in range(nr, nr + nb):
            trace = np.maximum(residual[:, k] + Cin[k], -np.inf)
            residual -= (AA[k].T @ (trace - Cin[k])[None]).T
            C[k] = trace
        if scipy.linalg.norm(Cin - C, 'fro') <= 0.001 * float(scipy.linalg.norm(C, 'fro')):
            break
        Cin = C.copy()
    empty = np.flatnonzero(C.sum(1) == 0)
    if len(empty):
        keep = np.setdiff1d(np.arange(Ab.shape[1]), empty)
        Ab = Ab[:, keep]
        C = np.delete(C, empty, 0)
        residual = np.delete(residual, empty, 1)
        S = np.delete(S, empty, 0)
        noise = np.delete(noise, empty)
        g = np.delete(g, empty)
        baseline = np.delete(baseline, empty)
        c1 = np.delete(c1, empty)
        lam = np.delete(lam, empty)
        bg_removed = [k for k in empty - nr if k > 0]
        nr -= len(empty) - len(bg_removed)
    return {'C': C[:nr], 'A': Ab[:, :nr].tocsc(), 'b': Ab[:, nr:].toarray(), 'f': C[nr:], 'S': S, 'bl': baseline, 'c1': c1, 'neurons_sn': noise, 'g': g, 'YrA': np.asarray(residual[:, :nr]).T, 'lam': lam}

"""Explicit overlapping-patch CNMF, rank-one merging and global refinement."""
import numpy as np
from scipy import sparse, linalg, stats
from scipy.sparse.csgraph import connected_components

def normalize(r):
    n = np.sqrt(np.asarray(r['A'].power(2).sum(0)).ravel())
    r['A'] = r['A'] @ sparse.diags(1 / (n + np.finfo(np.float32).eps))
    for key in ['C', 'S', 'YrA']:
        if r.get(key) is not None:
            r[key] = n[:, None] * r[key]
    for key in ['bl', 'c1', 'neurons_sn']:
        if r.get(key) is not None:
            r[key] = n * r[key]
    n = np.sqrt((r['b'] ** 2).sum(0))
    r['b'] = r['b'] @ sparse.diags(1 / (n + np.finfo(np.float32).eps))
    r['f'] = n[:, None] * r['f']
    return r

def patches(dims):
    centers = [list(range(15, n - 15, 24)) + [n - 15] for n in dims]
    for x in centers[0]:
        xs = np.arange(max(0, x - 15), min(dims[0], x + 16))
        for y in centers[1]:
            ys = np.arange(max(0, y - 15), min(dims[1], y + 16))
            xx, yy = np.meshgrid(xs, ys)
            yield (np.sort(np.ravel_multi_index((xx, yy), dims, order='F').ravel()), (len(xs), len(ys)))

def initialize_patches(movie, progress=print):
    T, h, w = movie.shape
    Y = movie.transpose(1, 2, 0).reshape(-1, T, order='F')
    d = h * w
    As = []
    Cs = []
    Rs = []
    Bs = []
    Fs = []
    mask = np.zeros(d, np.uint8)
    sn = np.zeros(d)
    for number, (ix, shape) in enumerate(patches((h, w))):
        patch = Y[ix].reshape(shape + (T,), order='F')
        noise, _ = get_noise_fft(Y[ix], noise_method='mean')
        A, C, b, f, _ = initialize_components(patch, nb=1)
        A = sparse.csc_matrix(A)
        Ab = sparse.hstack([A, b]).tocsc()
        norm = np.asarray(Ab.power(2).sum(0)).ravel() + np.finfo(np.float32).eps
        inv = sparse.diags(1 / norm)
        Cf = np.vstack([C, f])
        R = ((Ab.T @ Y[ix]).T @ inv - ((Ab.T @ Ab @ inv).T @ Cf).T)[:, :len(C)].T
        r = normalize(dict(A=A, C=C, b=b, f=f, YrA=R))
        full = sparse.csc_matrix((r['A'].data, ix[r['A'].indices], r['A'].indptr), shape=(d, A.shape[1]), dtype=np.float32)
        As.append(full)
        Cs.append(r['C'].astype(np.float32))
        Rs.append(r['YrA'].astype(np.float32))
        B = sparse.csc_matrix((r['b'].ravel(), ix, [0, len(ix)]), shape=(d, 1))
        Bs.append(B)
        Fs.append(r['f'].astype(np.float32))
        mask[ix] += 1
        sn[ix] = noise
        progress('patch', number + 1, 49)
    im = sparse.diags((1 / (mask + np.finfo(np.float32).eps)).astype(np.float32))
    A = (im @ sparse.hstack(As)).tocsc()
    B = im @ sparse.hstack(Bs)
    F = np.vstack(Fs)
    model = NMF(n_components=2, tol=1e-10, max_iter=100, random_state=1).fit(np.maximum(F, np.float64(0)))
    f = np.atleast_2d(model.components_)
    for _ in range(100):
        f /= np.sqrt((f ** 2).sum(1)[:, None]) + np.finfo(np.float32).eps
        try:
            b = np.fmax(B @ (F @ f.T) @ np.linalg.inv(f @ f.T), 0)
        except np.linalg.LinAlgError:
            b = np.fmax(B @ linalg.lstsq(f.T, F.T)[0].T, 0)
        try:
            f = np.linalg.solve(b.T @ b, (B.T @ b).T @ F)
        except np.linalg.LinAlgError:
            f = linalg.lstsq(b, B.toarray())[0] @ F
    norms = np.sqrt((b * b).sum(0))
    b = (b / (norms + np.finfo(np.float32).eps)).astype(np.float32)
    f *= norms[:, None]
    return dict(A=A, C=np.vstack(Cs), YrA=np.vstack(Rs), b=b, f=f, sn=sn)

def merge(r, p=1, mx=1000):
    A = r['A'].tocsc()
    C = r['C']
    R = r.get('YrA', np.zeros_like(C))
    nr = len(C)
    overlap = sparse.triu(A.T @ A, k=1).tocoo()
    cor = sparse.lil_matrix((nr, nr))
    for i, j in zip(overlap.row, overlap.col):
        cor[i, j] = stats.pearsonr(C[i].astype(float), C[j].astype(float)).statistic
    graph = (cor + cor.T > 0.85).multiply(sparse.triu(A.T @ A, k=1) > 0)
    n, labels = connected_components(graph)
    groups = [np.where(labels == i)[0] for i in range(n) if np.count_nonzero(labels == i) > 1]
    if not groups:
        return (r, 0)
    scores = [sum((float(cor[i, j]) for a, i in enumerate(group) for j in group[a + 1:])) for group in groups]
    order = np.argsort(np.asarray(scores))[::-1] if len(groups) > 1 else [0]
    groups = [groups[i] for i in order[:mx]]
    newA = []
    newC = []
    newR = []
    newS = []
    for group in groups:
        aa = A[:, group]
        cc = C[group] + R[group]
        weights = np.sqrt(np.asarray(aa.power(2).sum(0)).ravel() * (cc * cc).sum(1))
        a = aa @ weights
        for _ in range(10):
            c = np.maximum(aa.T @ a @ cc / (a @ a), 0)
            nc = c @ c
            if nc == 0:
                break
            a = np.maximum(aa @ (cc @ c) / nc, 0)
        norm = np.sqrt(a @ a)
        a /= norm
        c *= norm
        y = aa.T @ a @ cc / (a @ a)
        if r.get('g') is not None:
            raise RuntimeError('Unverified fixed-AR merge branch reached')
        trace, s, b, c1, noise, g, lam = deconvolve_trace(y, p=p)
        if p:
            trace = trace - b - c1 * np.ravel(g)[0] ** np.arange(len(y))
        newA.append(a)
        newC.append(trace)
        newR.append(y - trace)
        newS.append(s)
    keep = np.setdiff1d(np.arange(nr), np.unique(np.concatenate(groups)))
    r['A'] = sparse.hstack([A[:, keep], sparse.csc_matrix(np.asarray(newA).T)]).tocsc()
    r['C'] = np.vstack([C[keep], newC])
    r['YrA'] = np.vstack([R[keep], newR])
    if r.get('S') is not None:
        r['S'] = np.vstack([r['S'][keep], newS])
    return (r, len(groups))

def fit(movie, progress=print, checkpoint=None):
    T, h, w = movie.shape
    Y = movie.transpose(1, 2, 0).reshape(-1, T, order='F')
    r = initialize_patches(movie, progress)
    while True:
        r, n = merge(r)
        progress('merge', n, len(r['C']))
        if not n:
            break
    sn = r['sn']
    r.update(update_temporal(Y, r['A'], r['b'], r['C'], r['f'], streamed=True))
    normalize(r)
    if checkpoint:
        checkpoint('initial', r)
    for stage, p in enumerate([0, 1]):
        progress('spatial', stage, len(r['C']))
        A, b, C, f = update_spatial(Y, r['A'], r['C'], r['b'], r['f'], sn, (h, w), streamed=True)
        r.update(update_temporal(Y, A, b, C, f, p=p, streamed=True))
        if stage == 0:
            r, n = merge(r, p=0, mx=50)
            progress('refit_merge', n, len(r['C']))
    normalize(r)
    if checkpoint:
        checkpoint('refit', r)
    return r

import numpy as np
import scipy, scipy.stats, scipy.ndimage, scipy.optimize
from scipy import fftpack
from scipy.ndimage import percentile_filter
from scipy.sparse import csc_matrix
from scipy.stats import norm
import logging

def quality(movie, r, corrected_mode=False):
    T, h, w = movie.shape
    Y = movie.transpose(1, 2, 0)
    snr = []
    correlation = []
    for start in range(0, len(r['C']), 50):
        ix = slice(start, start + 50)
        fitness, _, _, _, rv, _ = evaluate_components(Y, (r['C'][ix] + r['YrA'][ix]).copy(), r['A'][:, ix], r['C'][ix], r['b'], r['f'], 30, N=12, Npeaks=10, corrected_mode=corrected_mode)
        snr.extend(-norm.ppf(np.exp(fitness / 12)))
        correlation.extend(rv)
    snr = np.asarray(snr)
    correlation = np.asarray(correlation)
    selected = np.flatnonzero(((snr > 2) | (correlation >= 0.85)) & (snr > 0.5) & (correlation > -1))
    return (selected, snr, correlation)

def dff(A, b, C, f, YrA):
    norm = np.sqrt(np.asarray(A.power(2).sum(0)).ravel())
    A = A @ scipy.sparse.diags(1 / norm)
    F = norm[:, None] * (C + YrA)
    B = A.T @ b @ f
    quantiles, _ = df_percentile(F[:, :250], axis=1)
    Fd = np.stack([percentile_filter(row, q, 250) for row, q in zip(F, quantiles)])
    Bd = np.stack([percentile_filter(row, q, 250) for row, q in zip(B, quantiles)])
    return (F - Fd) / (Bd + Fd)

def mode_robust_fast(inputData, axis=None):
    if axis is not None:

        def fnc(x):
            return mode_robust_fast(x)
        dataMode = np.apply_along_axis(fnc, axis, inputData)
    else:
        data = inputData.ravel()
        data = np.sort(data)
        dataMode = _hsm(data)
    return dataMode

def release_compatibility_mode(inputData, axis=None, dtype=None):
    if axis is not None:

        def fnc(x):
            return release_compatibility_mode(x, dtype=dtype)
        dataMode = np.apply_along_axis(fnc, axis, inputData)
    else:

        def _hsm(data):
            if data.size == 1:
                return data[0]
            elif data.size == 2:
                return data.mean()
            elif data.size == 3:
                i1 = data[1] - data[0]
                i2 = data[2] - data[1]
                if i1 < i2:
                    return data[:2].mean()
                elif i2 > i1:
                    return data[1:].mean()
                else:
                    return data[1]
            else:
                wMin = np.inf
                N = data.size // 2 + data.size % 2
                for i in range(0, N):
                    w = data[i + N - 1] - data[i]
                    if w < wMin:
                        wMin = w
                        j = i
                return _hsm(data[j:j + N])
        data = inputData.ravel()
        if type(data).__name__ == 'MaskedArray':
            data = data.compressed()
        if dtype is not None:
            data = data.astype(dtype)
        data = np.sort(data)
        dataMode = _hsm(data)
    return dataMode

def _hsm(data):
    if data.size == 1:
        return data[0]
    elif data.size == 2:
        return data.mean()
    elif data.size == 3:
        i1 = data[1] - data[0]
        i2 = data[2] - data[1]
        if i1 < i2:
            return data[:2].mean()
        elif i2 > i1:
            return data[1:].mean()
        else:
            return data[1]
    else:
        wMin = np.inf
        N = data.size // 2 + data.size % 2
        for i in range(0, N):
            w = data[i + N - 1] - data[i]
            if w < wMin:
                wMin = w
                j = i
        return _hsm(data[j:j + N])

def df_percentile(inputData, axis=None):
    logger = logging.getLogger('caiman')
    if axis is not None:

        def fnc(x):
            return df_percentile(x)
        result = np.apply_along_axis(fnc, axis, inputData)
        data_prct = result[:, 0]
        val = result[:, 1]
    else:
        err = True
        err_count = 0
        max_err_count = 10
        while err and err_count < max_err_count:
            try:
                bandwidth, mesh, density, cdf = kde(inputData)
                err = False
            except ValueError:
                err_count += 1
                logger.warning(f'kde() failure {err_count}. Concatenating traces and recomputing.')
                if not isinstance(inputData, list):
                    inputData = inputData.tolist()
                inputData += inputData
        try:
            cdf
        except NameError:
            logger.warning('Max err count reached. Reverting to median.')
            data_prct = 50
            val = np.median(np.array(inputData))
        else:
            data_prct = cdf[np.argmax(density)] * 100
            val = mesh[np.argmax(density)]
        if data_prct >= 100 or data_prct < 0:
            logger.warning('Invalid percentile (<0 or >100) computed. Reverting to median.')
            data_prct = 50
            val = np.median(np.array(inputData))
        if np.isnan(data_prct):
            logger.warning('NaN percentile computed. Reverting to median.')
            data_prct = 50
            val = np.median(np.array(inputData))
    return (data_prct, val)

def kde(data, N=None, MIN=None, MAX=None):
    N = 2 ** 12 if N is None else int(2 ** np.ceil(np.log2(N)))
    if MIN is None or MAX is None:
        minimum = min(data)
        maximum = max(data)
        Range = maximum - minimum
        MIN = minimum - Range / 10 if MIN is None else MIN
        MAX = maximum + Range / 10 if MAX is None else MAX
    R = MAX - MIN
    M = len(data)
    DataHist, bins = np.histogram(data, bins=N, range=(MIN, MAX))
    DataHist = DataHist / M
    DCTData = fftpack.dct(DataHist, norm=None)
    I = [iN * iN for iN in range(1, N)]
    SqDCTData = (DCTData[1:] / 2) ** 2
    guess = 0.1
    try:
        t_star = scipy.optimize.brentq(fixed_point, 0, guess, args=(M, I, SqDCTData))
    except ValueError as e:
        raise ValueError(f'Unable to find root: {str(e)}')
    SmDCTData = DCTData * np.exp(-np.arange(N) ** 2 * np.pi ** 2 * t_star / 2)
    density = fftpack.idct(SmDCTData, norm=None) * N / R
    mesh = [(bins[i] + bins[i + 1]) / 2 for i in range(N)]
    bandwidth = np.sqrt(t_star) * R
    density = density / np.trapezoid(density, mesh)
    cdf = np.cumsum(density) * (mesh[1] - mesh[0])
    return (bandwidth, mesh, density, cdf)

def fixed_point(t, M, I, a2):
    l = 7
    I = np.float64(I)
    M = np.float64(M)
    a2 = np.float64(a2)
    f = 2 * np.pi ** (2 * l) * np.sum(I ** l * a2 * np.exp(-I * np.pi ** 2 * t))
    for s in range(l, 1, -1):
        K0 = np.prod(range(1, 2 * s, 2)) / np.sqrt(2 * np.pi)
        const = (1 + (1 / 2) ** (s + 1 / 2)) / 3
        time = (2 * const * K0 / M / f) ** (2 / (3 + 2 * s))
        f = 2 * np.pi ** (2 * s) * np.sum(I ** s * a2 * np.exp(-I * np.pi ** 2 * time))
    return t - (2 * M * np.sqrt(np.pi) * f) ** (-2 / 5)

def peak_indexes(y, thres=0.3, min_dist=1, thres_abs=False):
    if isinstance(y, np.ndarray) and np.issubdtype(y.dtype, np.unsignedinteger):
        raise ValueError('y must be signed')
    if not thres_abs:
        thres = thres * (np.max(y) - np.min(y)) + np.min(y)
    min_dist = int(min_dist)
    dy = np.diff(y)
    zeros, = np.where(dy == 0)
    if len(zeros) == len(y) - 1:
        return np.array([], dtype=np.int64)
    if len(zeros):
        zeros_diff = np.diff(zeros)
        zeros_diff_not_one, = np.add(np.where(zeros_diff != 1), 1)
        zero_plateaus = np.split(zeros, zeros_diff_not_one)
        if zero_plateaus[0][0] == 0:
            dy[zero_plateaus[0]] = dy[zero_plateaus[0][-1] + 1]
            zero_plateaus.pop(0)
        if len(zero_plateaus) and zero_plateaus[-1][-1] == len(dy) - 1:
            dy[zero_plateaus[-1]] = dy[zero_plateaus[-1][0] - 1]
            zero_plateaus.pop(-1)
        for plateau in zero_plateaus:
            median = np.median(plateau)
            dy[plateau[plateau < median]] = dy[plateau[0] - 1]
            dy[plateau[plateau >= median]] = dy[plateau[-1] + 1]
    peaks = np.where((np.hstack([dy, 0.0]) < 0.0) & (np.hstack([0.0, dy]) > 0.0) & np.greater(y, thres))[0]
    if peaks.size > 1 and min_dist > 1:
        highest = peaks[np.argsort(y[peaks])][::-1]
        rem = np.ones(y.size, dtype=bool)
        rem[peaks] = False
        for peak in highest:
            if not rem[peak]:
                sl = slice(max(0, peak - min_dist), peak + min_dist + 1)
                rem[sl] = True
                rem[peak] = False
        peaks = np.arange(y.size)[~rem]
    return peaks

def symmetric_half_sample_mode(values):
    """Independent symmetric shortest-half-interval mode for diagnostic comparison."""
    data=np.sort(np.ravel(values))
    while len(data)>3:
        n=(len(data)+1)//2
        start=int(np.argmin(data[n-1:]-data[:len(data)-n+1]))
        data=data[start:start+n]
    if len(data)<3:return data.mean()
    left,right=data[1]-data[0],data[2]-data[1]
    return data[:2].mean() if left<right else data[1:].mean() if right<left else data[1]

def compute_event_exceptionality(traces, robust_std=False, N=5, use_mode_fast=False, sigma_factor=3.0, corrected_mode=False):
    if N == 0:
        raise Exception('FATAL: N=0 is not a valid value for compute_event_exceptionality()')
    T = traces.shape[-1]
    if use_mode_fast:
        md = mode_robust_fast(traces, axis=1)
    else:
        md = np.apply_along_axis(symmetric_half_sample_mode,1,traces) if corrected_mode else release_compatibility_mode(traces, axis=1)
    ff1 = traces - md[:, None]
    ff1 = -ff1 * (ff1 < 0)
    if robust_std:
        ff1 = np.sort(ff1, axis=1)
        ff1[ff1 == 0] = np.nan
        Ns = np.round(np.sum(ff1 > 0, 1) * 0.5)
        iqr_h = np.zeros(traces.shape[0])
        for idx, _ in enumerate(ff1):
            iqr_h[idx] = ff1[idx, -Ns[idx]]
        sd_r = 2 * iqr_h / 1.349
    else:
        Ns = np.sum(ff1 > 0, 1)
        sd_r = np.sqrt(np.sum(ff1 ** 2, 1) / Ns)
    z = (traces - md[:, None]) / (sigma_factor * sd_r[:, None])
    erf = scipy.special.log_ndtr(-z)
    erfc = np.cumsum(erf, 1)
    erfc[:, N:] -= erfc[:, :-N]
    fitness = np.min(erfc, 1)
    return (fitness, erfc, sd_r, md)

def find_activity_intervals(C, Npeaks=5, tB=-3, tA=10, thres=0.3):
    logger = logging.getLogger('caiman')
    K, T = C.shape
    L: list = []
    for i in range(K):
        if np.sum(np.abs(np.diff(C[i, :]))) == 0:
            L.append([])
            logger.debug(f'empty component at: {i}')
            continue
        indexes = peak_indexes(C[i, :], thres=thres)
        srt_ind = indexes[np.argsort(C[i, indexes])][::-1]
        srt_ind = srt_ind[:Npeaks]
        L.append(srt_ind)
    LOC = []
    for i in range(K):
        if len(L[i]) > 0:
            interval = np.kron(L[i], np.ones(int(np.round(tA - tB)), dtype=int)) + np.kron(np.ones(len(L[i]), dtype=int), np.arange(tB, tA))
            interval[interval < 0] = 0
            interval[interval > T - 1] = T - 1
            LOC.append(np.array(list(set(interval))))
        else:
            LOC.append(None)
    return LOC

def classify_components_ep(Y, A, C, b, f, Athresh=0.1, Npeaks=5, tB=-3, tA=10, thres=0.3):
    logger = logging.getLogger('caiman')
    K, _ = C.shape
    A = csc_matrix(A)
    AA = (A.T * A).toarray()
    nA = np.sqrt(np.array(A.power(2).sum(0)))
    AA /= np.outer(nA, nA.T)
    AA -= np.eye(K)
    LOC = find_activity_intervals(C, Npeaks=Npeaks, tB=tB, tA=tA, thres=thres)
    rval = np.zeros(K)
    significant_samples: list[Any] = []
    for i in range(K):
        if (i + 1) % 200 == 0:
            logger.info(f'Components evaluated: {i}')
        if LOC[i] is not None:
            atemp = A[:, i].toarray().flatten()
            atemp[np.isnan(atemp)] = np.nanmean(atemp)
            ovlp_cmp = np.where(AA[:, i] > Athresh)[0]
            indexes = set(LOC[i])
            for _, j in enumerate(ovlp_cmp):
                if LOC[j] is not None:
                    indexes = indexes - set(LOC[j])
            if len(indexes) == 0:
                indexes = set(LOC[i])
                logger.warning(f'Component {i} is only active jointly with neighboring components. Space ' + 'correlation calculation might be unreliable.')
            indexes = np.array(list(indexes)).astype(int)
            px = np.where(atemp > 0)[0]
            if px.size < 3:
                logger.warning(f'Component {i} is almost empty. Space correlation is set to 0.')
                rval[i] = 0
                significant_samples.append({0})
            else:
                ysqr = np.array(Y[px, :])
                ysqr[np.isnan(ysqr)] = np.nanmean(ysqr)
                mY = np.mean(ysqr[:, indexes], axis=-1)
                significant_samples.append(indexes)
                rval[i] = scipy.stats.pearsonr(mY, atemp[px])[0]
        else:
            rval[i] = 0
            significant_samples.append(0)
    return (rval, significant_samples)

def evaluate_components(Y, traces, A, C, b, f, final_frate, remove_baseline=True, N=5, robust_std=False, Athresh=0.1, Npeaks=5, thresh_C=0.3, sigma_factor=3.0, corrected_mode=False):
    logger = logging.getLogger('caiman')
    tB = np.minimum(-2, np.floor(-5.0 / 30 * final_frate))
    tA = np.maximum(5, np.ceil(25.0 / 30 * final_frate))
    logger.info(f'tB={tB!r},tA={tA!r}')
    dims, T = (Y.shape[:-1], Y.shape[-1])
    Yr = np.reshape(Y, (np.prod(dims), T), order='F')
    logger.debug('Computing event exceptionality delta')
    fitness_delta, erfc_delta, _, _ = compute_event_exceptionality(np.diff(traces, axis=1), robust_std=robust_std, N=N, sigma_factor=sigma_factor, corrected_mode=corrected_mode)
    logger.debug('Removing Baseline')
    if remove_baseline:
        num_samps_bl = np.minimum(traces.shape[-1] // 5, 800)
        slow_baseline = False
        if slow_baseline:
            traces = traces - scipy.ndimage.percentile_filter(traces, 8, size=[1, num_samps_bl])
        else:
            downsampfact = num_samps_bl
            elm_missing = int(np.ceil(T * 1.0 / downsampfact) * downsampfact - T)
            padbefore = int(np.floor(elm_missing / 2.0))
            padafter = int(np.ceil(elm_missing / 2.0))
            tr_tmp = np.pad(traces.T, ((padbefore, padafter), (0, 0)), mode='reflect')
            numFramesNew, num_traces = tr_tmp.shape
            logger.debug('Binning data ...')
            tr_BL = np.reshape(tr_tmp, (downsampfact, numFramesNew // downsampfact, num_traces), order='F')
            tr_BL = np.percentile(tr_BL, 8, axis=0)
            logger.debug('Interpolating data ...')
            logger.debug(tr_BL.shape)
            tr_BL = scipy.ndimage.zoom(np.array(tr_BL, dtype=np.float32), [downsampfact, 1], order=3, mode='constant', cval=0.0, prefilter=True)
            if padafter == 0:
                traces -= tr_BL.T
            else:
                traces -= tr_BL[padbefore:-padafter].T
    logger.debug('Computing event exceptionality')
    fitness_raw, erfc_raw, _, _ = compute_event_exceptionality(traces, robust_std=robust_std, N=N, sigma_factor=sigma_factor, corrected_mode=corrected_mode)
    logger.debug('Evaluating spatial footprint')
    r_values, significant_samples = classify_components_ep(Yr, A, C, b, f, Athresh=Athresh, Npeaks=Npeaks, tB=tB, tA=tA, thres=thresh_C)
    return (fitness_raw, fitness_delta, erfc_raw, erfc_delta, r_values, significant_samples)
