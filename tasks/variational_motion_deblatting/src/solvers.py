"""Local explicit extraction of the MIT-licensed deblatting method.
Copyright (c) 2022 Denys Rozumnyi; see test/validation.md, consolidated original licenses.
"""
import numpy as np
from numpy.fft import fft2, ifft2
from scipy import sparse
import scipy.sparse.linalg

from .physics_model import *
from .preprocessing import boundingBox
class Params:

    def __init__(self):
        self.projection_mode = 'exact'
        self.loop_maxiter = 200
        self.maxiter = 50
        self.cg_maxiter = 25
        self.cg_tol = 1e-05
        self.rel_tol_h = 0.009
        self.rel_tol_f = 0.0001
        self.lp = 1
        self.gamma = 1.0
        self.alpha_h = 1.0
        self.beta_h = 1000.0 * self.alpha_h
        self.sum1 = True
        self.do_fit = True
        self.alpha_f = 2 ** (-12)
        self.lambda_T = 0.001
        self.lambda_R = 0.01
        self.beta_fm = 0.001
        self.beta_f = 10 * self.alpha_f
        self.pyramid_eps = 1
        self.alpha_cross_f = 2 ** (-13)
        self.beta_cross_f = 10 * self.alpha_cross_f
        self.verbose = True
        self.do_cost = False
        self.visualize = True

def estimateFMH(I, B, M=None, F=None, Hmask=None, params=None):
    if params is None:
        params = Params()
    if I.shape != B.shape:
        raise Exception('Shapes must be equal!')
    if M is None:
        M = np.ones(I.shape[:2])
    if F is None:
        F = np.ones((M.shape[0], M.shape[1], I.shape[2]))
    if Hmask is None:
        Hmask = np.ones(I.shape[:2]).astype(bool)
        Hmask_small = Hmask
    else:
        pads = np.ceil((np.array(M.shape) - 1) / 2).astype(int)
        rmin, rmax, cmin, cmax = boundingBox(Hmask, pads)
        I = I[rmin:rmax, cmin:cmax, :]
        B = B[rmin:rmax, cmin:cmax, :]
        Hmask_small = Hmask[rmin:rmax, cmin:cmax]
    H = np.zeros(I.shape[:2])
    params.maxiter = 1
    rel_tol2 = params.rel_tol_h ** 2
    stateh = StateH()
    statefm = StateFM()
    for iter in range(params.loop_maxiter):
        H_old = H
        H, stateh = estimateH(I, B, M, F, Hmask=Hmask_small, state=stateh, params=params)
        F, M, statefm = estimateFM(I, B, H, M, F, state=statefm, params=params)
        reldiff2 = np.sum((H_old - H) ** 2) / np.sum(H ** 2)
        if reldiff2 < rel_tol2:
            break
    if params.do_fit:
        H = psffit(H)
        params.maxiter = 50
        F, M = estimateFM(I, B, H, M, F, params=params)
    He = np.zeros(Hmask.shape[:2])
    He[Hmask] = H[Hmask_small]
    return (He, F, M)

def estimateFM(I, B, H, M=None, F=None, F_T=None, M_T=None, state=None, params=None):
    if params is None:
        params = Params()
    if M is None:
        if F is not None:
            M = np.zeros(F.shape[:2])
        else:
            M = np.zeros(I.shape[:2])
    if F is None:
        F = np.zeros((M.shape[0], M.shape[1], 3))
    Fshape = F.shape
    f = vec3(F)
    m = M.flatten('F')
    if F_T is not None:
        F_T = vec3(F_T)
    if M_T is not None:
        M_T = M_T.flatten('F')
    Me = np.zeros(I.shape[:2])
    Fe = np.zeros(I.shape)
    idx_f, idy_f, idz_f = psfshift_idx(F.shape, I.shape)
    idx_m, idy_m = psfshift_idx(M.shape, I.shape[:2])
    Dx = None
    if state is not None:
        Dx = state.Dx
        Dy = state.Dy
        DTD = state.DTD
        Rn = state.Rn
        vx = state.vx
        vy = state.vy
        ax = state.ax
        ay = state.ay
        vx_m = state.vx_m
        vy_m = state.vy_m
        ax_m = state.ax_m
        ay_m = state.ay_m
        vf = state.vf
        af = state.af
        vm = state.vm
        am = state.am
    if Dx is None:
        Dx, Dy = createDerivatives0(Fshape)
        DTD = Dx.T @ Dx + Dy.T @ Dy
        vx = np.zeros((Dx.shape[0], Fshape[2]))
        vy = np.zeros((Dy.shape[0], Fshape[2]))
        ay = 0
        ax = 0
        vx_m = np.zeros((Dx.shape[0], 1))
        vy_m = np.zeros((Dy.shape[0], 1))
        ay_m = 0
        ax_m = 0
        af = 0
        vf = 0
        am = 0
        vm = 0
        if params.lambda_R > 0:
            RnA = createRnMatrix(Fshape[:2]).toarray()
            Rn = RnA.T @ RnA - RnA.T - RnA + np.eye(RnA.shape[0])
            Rn = sparse.csc_matrix(Rn)
    fH = fft2(H, axes=(0, 1))
    HT = np.conj(fH)
    HT3 = np.repeat(HT[:, :, np.newaxis], 3, axis=2)
    rhs_f = np.real(ifft2(HT3 * fft2(I - B, axes=(0, 1)), axes=(0, 1)))
    rhs_f = params.gamma * np.reshape(rhs_f[idx_f, idy_f, idz_f], (-1, Fshape[2]), 'F')
    if params.lambda_T > 0 and F_T is not None:
        rhs_f += params.lambda_T * F_T
    rhs_m = np.real(ifft2(HT * fft2(np.sum(B * (I - B), 2), axes=(0, 1)), axes=(0, 1)))
    rhs_m = -params.gamma * rhs_m[idx_m, idy_m]
    if params.lambda_T > 0 and M_T is not None:
        rhs_m += params.lambda_T * M_T
    beta_tv4 = np.repeat(params.beta_f, Fshape[2] + 1)
    rel_tol2 = params.rel_tol_f ** 2
    for iter in range(params.maxiter):
        fdx = Dx @ f
        fdy = Dy @ f
        mdx = Dx @ m
        mdy = Dy @ m
        f_old = f
        m_old = m
        if params.alpha_f > 0 and params.beta_f > 0:
            val_x = fdx + ax
            val_y = fdy + ay
            shrink_factor = lp_proximal_mapping(np.sqrt(val_x ** 2 + val_y ** 2), params.alpha_f / params.beta_f, params.lp)
            vx = val_x * shrink_factor
            vy = val_y * shrink_factor
            ax = ax + fdx - vx
            ay = ay + fdy - vy
            val_x = mdx + ax_m
            val_y = mdy + ay_m
            shrink_factor = lp_proximal_mapping(np.sqrt(val_x ** 2 + val_y ** 2), params.alpha_f / params.beta_f, params.lp)
            vx_m = val_x * shrink_factor
            vy_m = val_y * shrink_factor
            ax_m = ax_m + mdx - vx_m
            ay_m = ay_m + mdy - vy_m
        if params.beta_fm > 0:
            vf = f + af
            vm = m + am
            if params.pyramid_eps > 0:
                projector = project2pyramid_exact if params.projection_mode == 'exact' else project2pyramid
                vm, vf = projector(vm, vf, params.pyramid_eps)
            else:
                vf[vf < 0] = 0
                vm[vm < 0] = 0
                vm[vm > 1] = 1
            af = af + f - vf
            am = am + m - vm
        rhs1 = rhs_f + params.beta_f * (Dx.T @ (vx - ax) + Dy.T @ (vy - ay)) + params.beta_fm * (vf - af)
        rhs2 = rhs_m + params.beta_f * (Dx.T @ (vx_m - ax_m) + Dy.T @ (vy_m - ay_m)).flatten('F') + params.beta_fm * (vm - am)

        def estimateFM_cg_Ax(fmfun0):
            fmfun = np.reshape(fmfun0, (-1, 4), 'F')
            xf = fmfun[:, :Fshape[2]]
            xm = fmfun[:, -1]
            Fe[idx_f, idy_f, idz_f] = xf.flatten('F')
            Me[idx_m, idy_m] = xm
            HF = fH[:, :, np.newaxis] * fft2(Fe, axes=(0, 1))
            bHM = B * np.real(ifft2(fH * fft2(Me, axes=(0, 1)), axes=(0, 1)))[:, :, np.newaxis]
            yf = np.real(ifft2(HT3 * (HF - fft2(bHM, axes=(0, 1))), axes=(0, 1)))
            yf = params.gamma * np.reshape(yf[idx_f, idy_f, idz_f], (-1, Fshape[2]), 'F')
            ym = np.real(ifft2(HT * fft2(np.sum(B * (bHM - np.real(ifft2(HF, axes=(0, 1)))), 2), axes=(0, 1)), axes=(0, 1)))
            ym = params.gamma * ym[idx_m, idy_m]
            if params.lambda_T > 0 and F_T is not None:
                yf = yf + params.lambda_T * xf
            if params.lambda_T > 0 and M_T is not None:
                ym = ym + params.lambda_T * xm
            if params.lambda_R > 0:
                ym = ym + params.lambda_R * (Rn @ xm)
            res = np.c_[yf, ym] + beta_tv4 * (DTD @ fmfun) + params.beta_fm * fmfun
            return res.flatten('F')
        A = scipy.sparse.linalg.LinearOperator((4 * f.shape[0], 4 * f.shape[0]), matvec=estimateFM_cg_Ax)
        fm, info = scipy.sparse.linalg.cg(A, np.c_[rhs1, rhs2].flatten('F'), np.c_[f, m].flatten('F'), rtol=params.cg_tol, maxiter=params.cg_maxiter)
        fm = np.reshape(fm, (-1, 4), 'F')
        f = fm[:, :Fshape[2]]
        m = fm[:, -1]
        if state is not None:
            continue
        ff = f.flatten()
        df = ff - f_old.flatten()
        dm = m - m_old
        rel_diff2_f = df @ df / (ff @ ff)
        rel_diff2_m = dm @ dm / (m @ m)
        if rel_diff2_f < rel_tol2 and rel_diff2_m < rel_tol2:
            break
    f_img = ivec3(f, Fshape)
    m_img = ivec3(m, Fshape[:2])
    if state is None:
        return (f_img, m_img)
    else:
        state.Dx = Dx
        state.Dy = Dy
        state.DTD = DTD
        state.Rn = Rn
        state.vx = vx
        state.vy = vy
        state.ax = ax
        state.ay = ay
        state.vx_m = vx_m
        state.vy_m = vy_m
        state.ax_m = ax_m
        state.ay_m = ay_m
        state.vf = vf
        state.af = af
        state.vm = vm
        state.am = am
        return (f_img, m_img, state)

def estimateH(I, B, M, F, Hmask=None, state=None, params=None):
    if params is None:
        params = Params()
    if Hmask is None:
        Hmask = np.ones(I.shape[:2]).astype(bool)
        Hmask_small = Hmask
    elif state is None:
        pads = np.ceil((np.array(M.shape) - 1) / 2).astype(int)
        rmin, rmax, cmin, cmax = boundingBox(Hmask, pads)
        I = I[rmin:rmax, cmin:cmax, :]
        B = B[rmin:rmax, cmin:cmax, :]
        Hmask_small = Hmask[rmin:rmax, cmin:cmax]
    else:
        Hmask_small = Hmask
    H = None
    if state is None:
        v_lp = 0
        a_lp = 0
    else:
        H = state.H
        v_lp = state.v_lp
        a_lp = state.a_lp
    if H is None:
        H = np.zeros((np.count_nonzero(Hmask_small),))
    hsize = Hmask_small.shape
    iF = fft2(psfshift(F, hsize), axes=(0, 1))
    iM = fft2(psfshift(M, hsize), axes=(0, 1))
    Fgb = fft2(I - B, axes=(0, 1))
    Fbgb = fft2(B * (I - B), axes=(0, 1))
    iM3 = np.repeat(iM[:, :, np.newaxis], 3, axis=2)
    rhs_const = np.sum(np.real(ifft2(np.conj(iF) * Fgb - np.conj(iM3) * Fbgb, axes=(0, 1))), 2)
    rhs_const = params.gamma * rhs_const[Hmask_small]
    He = np.zeros(hsize)
    rel_tol2 = params.rel_tol_h ** 2
    for iter in range(params.maxiter):
        H_old = H
        if params.beta_h > 0:
            v_lp = H + a_lp
            if params.sum1:
                v_lp = proj2simplex(v_lp)
            elif params.lp == 1:
                temp = v_lp < params.alpha_h / params.beta_h
                v_lp[temp] = 0
                v_lp[~temp] -= params.alpha_h / params.beta_h
            elif params.lp == 0:
                v_lp[v_lp <= np.sqrt(2 * params.alpha_h / params.beta_h)] = 0
            a_lp = a_lp + H - v_lp
        rhs = rhs_const + params.beta_h * (v_lp - a_lp)

        def estimateH_cg_Ax(hfun):
            He[Hmask_small] = hfun
            FH = fft2(He, axes=(0, 1))
            Fh = iF * np.repeat(FH[:, :, np.newaxis], 3, axis=2)
            BMh = B * np.repeat(np.real(ifft2(iM * FH, axes=(0, 1)))[:, :, np.newaxis], 3, axis=2)
            Fh_BMh = Fh - fft2(BMh, axes=(0, 1))
            res = np.sum(np.real(ifft2(np.conj(iF) * Fh_BMh - np.conj(iM3) * fft2(B * np.real(ifft2(Fh_BMh, axes=(0, 1))), axes=(0, 1)), axes=(0, 1))), 2)
            res = params.gamma * res[Hmask_small] + params.beta_h * hfun
            return res
        A = scipy.sparse.linalg.LinearOperator((H.shape[0], H.shape[0]), matvec=estimateH_cg_Ax)
        H, info = scipy.sparse.linalg.cg(A, rhs, H, rtol=params.cg_tol, maxiter=params.cg_maxiter)
        if state is not None:
            continue
        Diff = H - H_old
        rel_diff2 = Diff @ Diff / (H @ H)
        if rel_diff2 < rel_tol2:
            break
    oHe = np.zeros(Hmask.shape)
    oHe[Hmask] = H
    if state is None:
        return oHe
    else:
        state.a_lp = a_lp
        state.v_lp = v_lp
        state.H = H
        return (oHe, state)

class StateH:

    def __init__(self):
        self.H = None
        self.a_lp = 0
        self.v_lp = 0
        self.rgnA = 0
        self.device = None

class StateFM:

    def __init__(self):
        self.vx = 0
        self.vy = 0
        self.ax = 0
        self.ay = 0
        self.vx_m = 0
        self.vy_m = 0
        self.ax_m = 0
        self.ay_m = 0
        self.vf = 0
        self.af = 0
        self.vm = 0
        self.am = 0
        self.Dx = None
        self.Dy = None
        self.DTD = None
        self.Rn = None
        self.vc = 0
        self.ac = 0
        self.vc_m = 0
        self.ac_m = 0
        self.Mask4 = None
        self.device = None
