"""Explicit local hemodynamic basis, AR regression and convolution mathematics.

Copyright(c)2018 International Neuroinformatics Coordinating Facility, MIT.
Mathematical functions extracted from validated rsHRF author implementation.
No rsHRF, SPM, neuroimaging, regression-estimator or domain solver is imported.
"""
import numpy as np
from scipy import stats,linalg,signal
from scipy.sparse import lil_matrix
from scipy.special import gammaln

def orthogonalize_basis(X, OPT="pad"):
    """
    Recursive Gram-Schmidt orthogonalisation of basis functions
    @X - matrix
    @OPT - 'norm' - for Euclidean normalisation
           'pad'  - for zero padding of null space (default)
    """

    def gs_cofficient(v1, v2):
        return np.dot(v2, v1) / np.dot(v1, v1)

    def multiply(cofficient, v):
        return map((lambda x: x * cofficient), v)

    def proj(v1, v2):
        return multiply(gs_cofficient(v1, v2), v1)

    def gs(X, row_vecs=True, norm=True):
        if not row_vecs:
            X = X.T
        Y = X[0:1, :].copy()
        for i in range(1, X.shape[0]):
            proj = np.diag(
                (X[i, :].dot(Y.T) / np.linalg.norm(Y, axis=1) ** 2).flat
            ).dot(Y)
            Y = np.vstack((Y, X[i, :] - proj.sum(0)))
        if norm:
            Y = np.diag(1 / np.linalg.norm(Y, axis=1)).dot(Y)
        if row_vecs:
            return Y
        else:
            return Y.T

    if OPT == "norm":
        return gs(X, row_vecs=False, norm=True)
    elif OPT == "pad":
        return gs(X, row_vecs=False, norm=False)
    else:
        return X


def double_gamma_hrf(RT, P=None, fMRI_T=16):
    """
    @RT - scan repeat time
    @P  - parameters of the response function (two gamma functions)

    defaults  (seconds)
    %	P[0] - Delay of Response (relative to onset)	    6
    %	P[1] - Delay of Undershoot (relative to onset)     16
    %	P[2] - Dispersion of Response			            1
    %	P[3] - Dispersion of Undershoot			            1
    %	P[4] - Ratio of Response to Undershoot		        6
    %	P[5] - Onset (seconds)				                0
    %	P[6] - Length of Kernel (seconds)		           32

    hrf  - hemodynamic response function
    P    - parameters of the response function
    """
    p = np.array([6, 16, 1, 1, 6, 0, 32], dtype=float)
    if P is not None:
        p[0 : len(P)] = P
    _spm_Gpdf = lambda x, h, l: np.exp(
        h * np.log(l) + (h - 1) * np.log(x) - (l * x) - gammaln(h)
    )
    # modelled hemodynamic response function - {mixture of Gammas}
    dt = RT / float(fMRI_T)
    u = np.arange(0, int(p[6] / dt + 1)) - p[5] / dt
    # Known division-by-zero and a negative value for logarithm's argument is invalid.
    # Negative value occurs when onset is shifted for the time derivative.
    with np.errstate(divide="ignore", invalid="ignore"):
        hrf = (
            _spm_Gpdf(u, p[0] / p[2], dt / p[2])
            - _spm_Gpdf(u, p[1] / p[3], dt / p[3]) / p[4]
        )
    idx = np.arange(0, int((p[6] / RT) + 1)) * fMRI_T
    hrf = hrf[idx]
    hrf = np.nan_to_num(hrf)
    hrf = hrf / np.sum(hrf)
    return hrf


def canonical_basis_raw(xBF):
    dt = xBF["dt"]
    fMRI_T = xBF["T"]
    p = np.array([6, 16, 1, 1, 6, 0, 32], dtype=float)
    p[len(p) - 1] = xBF["len"]
    bf = double_gamma_hrf(dt, p, fMRI_T)
    bf = bf[:, np.newaxis]
    # time-derivative
    if xBF["TD_DD"]:
        dp = 1
        p[5] = p[5] + dp
        D = (bf[:, 0] - double_gamma_hrf(dt, p, fMRI_T)) / dp
        D = D[:, np.newaxis]
        bf = np.append(bf, D, axis=1)
        p[5] = p[5] - dp
        # dispersion-derivative
        if xBF["TD_DD"] == 2:
            dp = 0.01
            p[2] = p[2] + dp
            D = (bf[:, 0] - double_gamma_hrf(dt, p, fMRI_T)) / dp
            D = D[:, np.newaxis]
            bf = np.append(bf, D, axis=1)
    return bf


def ideal_filter(x, TR, Bands, m=5000):
    nvar = x.shape[1]
    nbin = int(np.ceil(nvar / m))
    for i in range(1, nbin + 1):
        if i != nbin:
            ind_X = [j for j in range((i - 1) * m, i * m)]
        else:
            ind_X = [j for j in range((i - 1) * m, nvar)]
        x1 = x[:, ind_X]
        x1 = conn_filter(TR, Bands, x1) + np.mean(x1, axis=0, keepdims=True)
        x[:, ind_X] = x1
    return x


def conn_filter(rt, filter, x):
    Nx = x.shape[0]
    fy = np.fft.fft(np.concatenate((x, np.flipud(x)), axis=0), axis=0)
    f = np.arange(fy.shape[0])
    f = f.reshape(1, -1)
    f = np.min((f, fy.shape[0] - f), axis=0)
    low = filter[0] * (rt * fy.shape[0])
    high = filter[1] * (rt * fy.shape[0])
    idx_low = np.argwhere(np.any(f < low, axis=0))
    idx_high = np.argwhere(np.any(f >= high, axis=0))
    idx = np.concatenate((idx_low, idx_high)).reshape(-1)
    fy[idx, :] = 0.0
    y = np.real(np.fft.ifft(fy, axis=0))
    y = y[0:Nx, :]
    return y


def regress(y, X):
    """
    This function was originally written by Guo-Rong Wu, the output of this function MUST coincide with the one in function wgr_regress in rsHRF_estimation_FIR.m which is a "copy from function [b,bint,r,rint,stats] = regress(y,X,alpha)"
    """
    n, ncolX = X.shape
    Q, R, perm = linalg.qr(X, mode="economic", pivoting=True)
    if R.ndim == 0:
        p = 0
    elif R.ndim == 1:
        p = int(
            np.abs(R[0]).item() > 0
        )  # Converting to scalar using .item() for compatibility and safe access
    else:
        if np.amin(R.shape) == 1:
            # Accessing scalar safely via R.flat[0] for robustness
            p = int(np.abs(R.flat[0]) > 0)
        else:
            p = np.sum(
                np.abs(np.diagonal(R)) > abs(max(n, ncolX) * np.spacing(R[0][0]))
            )
    if p < ncolX:
        R = R[0:p, 0:p]
        Q = Q[:, 0:p]
        perm = perm[0:p]
    b = np.zeros((ncolX))
    if R.shape[0] == R.shape[1]:
        try:
            b[perm] = linalg.solve(R, np.matmul(Q.T, y))
        except linalg.LinAlgError:
            b[perm] = linalg.lstsq(R, np.matmul(Q.T, y))[0]
    else:
        b[perm] = linalg.lstsq(R, np.matmul(Q.T, y))[0]
    return b


def glsco(X, Y, sMRI=[], AR_lag=0, max_iter=20):
    """
    Linear regression when disturbance terms follow AR(p)
    -----------------------------------
    Model:
    Yt = Xt * Beta + ut ,
    ut = Phi1 * u(t-1) + ... + Phip * u(t-p) + et
    where et ~ N(0,s^2)
    -----------------------------------
    Algorithm:
    Cochrane-Orcutt iterated regression (Feasible generalized least squares)
    -----------------------------------
    Usage:
    Y = dependent variable (n * 1 vector)
    X = regressors (n * k matrix)
    AR_lag = number of lags in AR process
    -----------------------------------
    Returns:
    Beta = estimator corresponding to the k regressors

    This function was originally written by Guo-Rong Wu, the output of this function MUST coincide with the one in function wgr_glsco_sm in rsHRF_estimation_FIR.m.
    """

    nobs, nvar = X.shape
    if isinstance(sMRI, list) and len(sMRI) == 0:
        Beta = regress(Y, X)
    else:
        sMRI = np.array(sMRI)
        try:
            Beta = linalg.solve(np.matmul(X.T, X) + sMRI, np.matmul(X.T, Y))
        except linalg.LinAlgError:
            Beta = linalg.lstsq(np.matmul(X.T, X) + sMRI, np.matmul(X.T, Y))[0]
    resid = Y - np.matmul(X, Beta)
    if AR_lag == 0:
        res_sum = np.cov(resid)
        return res_sum, Beta
    max_tol = min(1e-6, max(np.absolute(Beta)) / 1000)
    for r in range(max_iter):
        Beta_temp = Beta
        X_AR = np.zeros((nobs - (2 * AR_lag), AR_lag))
        for m in range(AR_lag):
            X_AR[:, m] = resid[AR_lag - m - 1 : nobs - AR_lag - m - 1]
        Y_AR = resid[AR_lag : nobs - AR_lag]
        AR_para = regress(Y_AR, X_AR)
        X_main = X[AR_lag:nobs, :]
        Y_main = Y[AR_lag:nobs]
        for m in range(AR_lag):
            X_main = X_main - (AR_para[m] * (X[AR_lag - m - 1 : nobs - m - 1, :]))
            Y_main = Y_main - (AR_para[m] * (Y[AR_lag - m - 1 : nobs - m - 1]))
        if isinstance(sMRI, list) and len(sMRI) == 0:
            Beta = regress(Y_main, X_main)
        else:
            try:
                Beta = linalg.solve(
                    np.matmul(X_main.T, X_main) + sMRI, np.matmul(X_main.T, Y_main)
                )
            except linalg.LinAlgError:
                Beta = linalg.lstsq(
                    np.matmul(X_main.T, X_main) + sMRI, np.matmul(X_main.T, Y_main)
                )[0]
        resid = Y[AR_lag:nobs] - X[AR_lag:nobs, :].dot(Beta)
        if max(np.absolute(Beta - Beta_temp)) < max_tol:
            break
    res_sum = np.cov(resid)
    return res_sum, Beta


def knee_pt(y):
    res_x, _id = knee_pt_helper(y)
    # knee_pt_helper returns nan sentinel on bad input, causing
    # meaningless errors. In the MATLAB version, an error() is raised here.
    if not np.isfinite(_id):
        return res_x, _id
    _idm = np.argmin(y)
    ratio = np.abs(y[_id] - y[_idm]) / np.abs(np.max(y) - np.min(y))
    if ratio > 0.5:
        idx_of_result = _idm
    else:
        idx_of_result = _id
    return res_x, idx_of_result


def knee_pt_helper(y, x=None):
    x_was_none = False
    use_absolute_dev_p = True
    res_x = np.nan
    idx_of_result = np.nan

    if type(y) is not np.ndarray:
        print("knee_pt: y must be a numpy 1D vector")
        return res_x, idx_of_result
    else:
        if y.ndim >= 2:
            print("knee_pt: y must be 1 dimensional")
            return res_x, idx_of_result
        if np.size(y) == 0:
            print("knee_pt: y can not be an empty vector")
            return res_x, idx_of_result
        else:
            if x is None:
                x_was_none = True
                x = np.arange(1, np.amax(y.shape) + 1, dtype=int)
            if x.shape != y.shape:
                print("knee_pt: y and x must have the same dimensions")
                return res_x, idx_of_result
            if y.size < 3:
                res_x, idx_of_result = np.min(y), np.argmin(y)
                return res_x, idx_of_result
            if np.all(np.diff(x) >= 0) and (not x_was_none):
                idx = np.argsort(x)
                y = np.sort(y)
                x = np.sort(x)
            else:
                idx = np.arange(0, np.amax(x.shape))
            sigma_xy = np.cumsum(np.multiply(x, y), axis=0)
            sigma_x = np.cumsum(x, axis=0)
            sigma_y = np.cumsum(y, axis=0)
            sigma_xx = np.cumsum(np.multiply(x, x), axis=0)
            n = np.arange(1, np.amax(y.shape) + 1).conj().T
            det = np.multiply(n, sigma_xx) - np.multiply(sigma_x, sigma_x)
            # det[0] is always 0 because there is only one
            # point in the first step of cumulative summation and a line cannot pass through
            # single point. It is safe to silence the warning because det[0] is never read,
            # as the error_curve loop starts from index 1.
            with np.errstate(divide="ignore", invalid="ignore"):
                mfwd = (np.multiply(n, sigma_xy) - np.multiply(sigma_x, sigma_y)) / det
                bfwd = -1 * (
                    (np.multiply(sigma_x, sigma_xy) - np.multiply(sigma_xx, sigma_y))
                    / det
                )

            sigma_xy = np.cumsum(np.multiply(x[::-1], y[::-1]), axis=0)
            sigma_x = np.cumsum(x[::-1], axis=0)
            sigma_y = np.cumsum(y[::-1], axis=0)
            sigma_xx = np.cumsum(np.multiply(x[::-1], x[::-1]), axis=0)
            n = np.arange(1, np.amax(y.shape) + 1).conj().T
            det = np.multiply(n, sigma_xx) - np.multiply(sigma_x, sigma_x)
            # same reason as above
            with np.errstate(divide="ignore", invalid="ignore"):
                mbck = (
                    (np.multiply(n, sigma_xy) - np.multiply(sigma_x, sigma_y)) / det
                )[::-1]
                bbck = (
                    -1
                    * (
                        (
                            np.multiply(sigma_x, sigma_xy)
                            - np.multiply(sigma_xx, sigma_y)
                        )
                        / det
                    )
                )[::-1]

            error_curve = np.full(y.shape, np.nan)
            for breakpt in range(1, np.amax((y - 1).shape)):
                delsfwd = (
                    np.multiply(mfwd[breakpt], x[0 : breakpt + 1]) + bfwd[breakpt]
                ) - y[0 : breakpt + 1]
                delsbck = (np.multiply(mbck[breakpt], x[breakpt:]) + bbck[breakpt]) - y[
                    breakpt:
                ]
                if use_absolute_dev_p:
                    error_curve[breakpt] = np.sum(np.abs(delsfwd)) + np.sum(
                        np.abs(delsbck)
                    )
                else:
                    error_curve[breakpt] = np.sqrt(
                        np.sum(np.multiply(delsfwd, delsfwd))
                    ) + np.sqrt(np.sum(np.multiply(delsbck, delsbck)))
            try:
                loc = np.nanargmin(error_curve)
            except ValueError as e:
                loc = 0
            res_x = x[loc]
            idx_of_result = idx[loc]
    return res_x, idx_of_result


def onset_design(u, bf, T, T0, nscans):
    """
    @u - BOLD event vector (microtime).
    @bf - basis set matrix
    @T - microtime resolution (number of time bins per scan)
    @T0 - microtime onset (reference time bin, see slice timing)

    This function was originally written by Guo-Rong Wu, the output of this function MUST coincide with the one in function wgr_onset_design in rsHRF_estimation_temporal_basis.m.
    """

    ind = np.arange(0, max(u.shape))
    X = np.empty((0, len(ind)))
    for p in range(bf.shape[1]):
        x = np.convolve(u, bf[:, p])
        x = x[ind]
        X = np.append(X, [x], axis=0)
    X = X.T
    """
    Resample regressors at acquisition times
    """
    X = X[(np.arange(0, nscans) * T) + (T0 - 1), :]
    return X


def glm_estimation(dat, u, bf, T, T0, AR_lag):
    """
    @u - BOLD event vector (microtime).

    This function was originally written by Guo-Rong Wu, the output of this function MUST coincide with the one in function wgr_glm_estimation in rsHRF_estimation_temporal_basis.m.
    """

    nscans = dat.shape[0]
    x = onset_design(u, bf, T, T0, nscans)
    X = np.append(x, np.ones((nscans, 1)), axis=1)
    res_sum, Beta = glsco(X, dat, AR_lag=AR_lag)
    return np.real(res_sum), Beta


def hrf_fit(dat, xBF, u, bf):
    """
    @u    - BOLD event vector (microtime).
    @nlag - time lag from neural event to BOLD event

    This function was originally written by Guo-Rong Wu, the output of this function MUST coincide with the one in function wgr_hrf_fit in rsHRF_estimation_temporal_basis.m.
    """

    lag = xBF["lag"]
    AR_lag = xBF["AR_lag"]
    nlag = len(lag)
    erm = np.zeros((1, nlag))
    beta = np.zeros((bf.shape[1] + 1, nlag))
    for i in range(nlag):
        u_lag = np.append(u[0, lag[i] :], np.zeros((1, lag[i]))).T
        erm[0, i], beta[:, i] = glm_estimation(
            dat, u_lag, bf, xBF["T"], xBF["T0"], AR_lag
        )
    x, idx = knee_pt(np.ravel(erm))
    if idx == nlag - 1:
        idx = idx - 1
    beta_hrf = beta[:, idx + 1]
    beta_hrf = np.append(beta_hrf, lag[idx + 1])
    return beta_hrf


def BOLD_event_vector(N, matrix, thr, k, temporal_mask):
    """
    Signal at a timepoint is counted as an event
    when the following conditions are met:
    Signal exceeds the threshold thr[0] and it is greater
    than its k-neighbours on both two sides.
    i.e. it is a local peak exceeding the threshold.

    This function was originally written by Guo-Rong Wu, the output of this function MUST coincide with the one in rsHRF_find_event_vector.m.
    """

    data = lil_matrix((1, N))
    matrix = matrix[:, np.newaxis]
    matrix = np.nan_to_num(matrix)
    if 0 in np.array(temporal_mask).shape:
        matrix = stats.zscore(matrix, ddof=1)
        for t in range(1 + k, N - k + 1):
            if (
                matrix[t - 1, 0] > thr[0]
                and np.all(matrix[t - k - 1 : t - 1, 0] < matrix[t - 1, 0])
                and np.all(matrix[t - 1, 0] > matrix[t : t + k, 0])
            ):
                data[0, t - 1] = 1
    else:
        keep = np.asarray(temporal_mask, dtype=bool)
        datm = np.mean(matrix[keep])
        datstd = np.std(matrix[keep])
        if datstd == 0:
            datstd = 1
        matrix = (matrix - datm) / datstd
        for t in range(1 + k, N - k + 1):
            if keep[t - 1]:
                if (
                    matrix[t - 1, 0] > thr[0]
                    and np.all(matrix[t - k - 1 : t - 1, 0] < matrix[t - 1, 0])
                    and np.all(matrix[t - 1, 0] > matrix[t : t + k, 0])
                ):
                    data[0, t - 1] = 1.0
    return data


def parameters():
    """Actual fixed author default canonical-path parameters."""
    return dict(TR=2.,T=3,T0=1,dt=2/3,len=24,TD_DD=2,AR_lag=1,
                lag=np.arange(6,13,dtype=int),thr=1,localK=1,temporal_mask=[])
def canonical_basis():
    """37x3 unnormalized orthogonal canonical/time/dispersion basis."""
    return orthogonalize_basis(canonical_basis_raw(parameters()))
def kernel_at_tr(hrf):
    """Zero-padded Kaiser5 polyphase antialiasing, microtime factor3."""
    return signal.resample_poly(hrf,1,3,axis=0)
def convolve_periodic(drive,hrf):
    """Time-first periodic convolution with HRF zero-padded to record length."""
    h=kernel_at_tr(hrf);n=len(drive)
    H=np.fft.fft(np.pad(h,((0,n-len(h)),(0,0))),axis=0)
    return np.fft.ifft(H*np.fft.fft(drive,axis=0),axis=0).real
def deconvolve_ridge(observation,hrf):
    """Author noniterative spectral Tikhonov inverse, lambda=.1mean|H|²."""
    h=kernel_at_tr(hrf);n=len(observation);out=np.zeros_like(observation)
    for i in range(observation.shape[1]):
        H=np.fft.fft(np.append(h[:,i],np.zeros((n-len(h),1))),axis=0)
        M=np.fft.fft(observation[:,i]);power=H*H.conj()
        out[:,i]=np.fft.ifft(H.conj()*M/(power+.1*np.mean(power))).real
    return out
