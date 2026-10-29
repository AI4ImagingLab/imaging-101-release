"""Selected two-source mathematics from pyTSEB, Copyright2016 Hector Nieto and contributors.
GNU LGPL version3 or later. Explicit PT/OSEB, radiative transfer, resistance and stability equations; no domain-library imports.
"""
import numpy as np
from collections import deque
import time
TSEB_L_thres = 0.001
TSEB_U_FRICTION_MIN = 0.01
TSEB_U_S_MIN = 0.01
TSEB_U_C_MIN = 0.01
TSEB_R_A_MIN = 0.1
TSEB_R_A_MAX = None
TSEB_RES_MIN = 0.1
TSEB_RES_MAX = None
TSEB_ITERATIONS = 15
TSEB_KB_1_DEFAULT = 0.0
TSEB_KUSTAS_NORMAN_1999 = 0
TSEB_F_ALL_FLUXES = 0
TSEB_F_ZERO_LE_S = 3
TSEB_F_ZERO_LE = 5
TSEB_F_ALL_FLUXES_OS = 10
TSEB_F_ZERO_LE_OS = 15
TSEB_F_FAILED_TS = 254
TSEB_F_INVALID = 255
TSEB_F_OUT_LIMS = 252
TSEB_FLUX_LIMS = (-200, 1000)
met_sb = 5.670373e-08
met_c_pd = 1003.5
met_c_pv = 1865
met_epsilon = 0.622
met_R_d = 287.04
rad_TAUD_STEP_SIZE_DEG = 5
res_CROP = 12
res_KARMAN = 0.41
res_KN_b = 0.012
res_KN_c = 0.0038
res_KN_C_dash = 90.0
MO_KARMAN = 0.41
MO_GRAVITY = 9.8
MO_UNSTABLE_THRES = None
MO_STABLE_THRES = None
wnd_KARMAN = 0.41

def CI_calc_omega0_Kustas(LAI, f_C, x_LAD=1, isLAIeff=True):
    LAI, f_C, x_LAD = map(np.asarray, (LAI, f_C, x_LAD))
    theta = np.zeros(LAI.shape)
    K_be = np.sqrt(x_LAD ** 2 + np.tan(theta) ** 2) / (x_LAD + 1.774 * (x_LAD + 1.182) ** (-0.733))
    if isLAIeff:
        F = LAI / f_C
    else:
        F = np.array(LAI)
    trans = np.asarray(f_C * np.exp(-K_be * F) + (1.0 - f_C))
    trans[trans <= 0] = 1e-36
    omega0 = -np.log(trans) / (F * K_be)
    return omega0

def CI_calc_omega_Kustas(omega0, theta, w_C=1):
    w_C = 1.0 / w_C
    omega = omega0 / (omega0 + (1.0 - omega0) * np.exp(-2.2 * np.radians(theta) ** (3.8 - 0.46 * w_C)))
    return omega

def MO_calc_L(ustar, T_A_K, rho, c_p, H, LE):
    l_mo = MO_calc_mo_length_hv(ustar, T_A_K, rho, c_p, H, LE)
    return np.asarray(l_mo)

def MO_calc_Psi_H(zoL):
    if MO_UNSTABLE_THRES is not None or MO_STABLE_THRES is not None:
        zoL = np.clip(zoL, MO_UNSTABLE_THRES, MO_STABLE_THRES)
    Psi_H = MO_psi_h_brutsaert(zoL)
    return np.asarray(Psi_H)

def MO_calc_Psi_M(zoL):
    if MO_UNSTABLE_THRES is not None or MO_STABLE_THRES is not None:
        zoL = np.clip(zoL, MO_UNSTABLE_THRES, MO_STABLE_THRES)
    Psi_M = MO_psi_m_brutsaert(zoL)
    return np.asarray(Psi_M)

def MO_calc_mo_length_hv(ustar, T_A_K, rho, c_p, H, LE):
    ustar, T_A_K, rho, c_p, H, LE = map(np.asarray, (ustar, T_A_K, rho, c_p, H, LE))
    Lambda = met_calc_lambda(T_A_K)
    E = LE / Lambda
    del LE, Lambda
    Hv = H + 0.61 * T_A_K * c_p * E
    del H, E
    L = np.asarray(np.ones(ustar.shape) * float('inf'))
    i = Hv != 0
    L_const = np.asarray(MO_KARMAN * MO_GRAVITY / T_A_K)
    L[i] = -ustar[i] ** 3 / (L_const[i] * (Hv[i] / (rho[i] * c_p[i])))
    return np.asarray(L)

def MO_calc_u_star(u, z_u, L, d_0, z_0M):
    u, z_u, L, d_0, z_0M = map(np.asarray, (u, z_u, L, d_0, z_0M))
    L[L == 0.0] = 1e-36
    Psi_M = MO_calc_Psi_M((z_u - d_0) / L)
    Psi_M0 = MO_calc_Psi_M(z_0M / L)
    del L
    u_star = u * MO_KARMAN / (np.log((z_u - d_0) / z_0M) - Psi_M + Psi_M0)
    return np.asarray(u_star)

def MO_psi_h_brutsaert(zol):
    zol = np.asarray(zol)
    psi_h = np.zeros(zol.shape)
    finite = np.isfinite(zol)
    i = np.logical_and(finite, zol >= 0.0)
    a = 6.1
    b = 2.5
    psi_h[i] = -a * np.log(zol[i] + (1.0 + zol[i] ** b) ** (1.0 / b))
    i = np.logical_and(finite, zol < 0.0)
    y = -zol[i]
    del zol
    c = 0.33
    d = 0.057
    n = 0.78
    psi_h[i] = (1.0 - d) / n * np.log((c + y ** n) / c)
    return psi_h

def MO_psi_m_brutsaert(zol):
    zol = np.asarray(zol)
    finite = np.isfinite(zol)
    psi_m = np.zeros(zol.shape)
    i = np.logical_and(finite, zol >= 0.0)
    a = 6.1
    b = 2.5
    psi_m[i] = -a * np.log(zol[i] + (1.0 + zol[i] ** b) ** (1.0 / b))
    i = np.logical_and(finite, zol < 0)
    y = -zol[i]
    del zol
    a = 0.33
    b = 0.41
    x = np.asarray((y / a) ** 0.333333)
    psi_0 = -np.log(a) + 3 ** 0.5 * b * a ** 0.333333 * np.pi / 6.0
    y = np.minimum(y, b ** (-3))
    psi_m[i] = np.log(a + y) - 3.0 * b * y ** 0.333333 + b * a ** 0.333333 / 2.0 * np.log((1.0 + x) ** 2 / (1.0 - x + x ** 2)) + 3.0 ** 0.5 * b * a ** 0.333333 * np.arctan((2.0 * x - 1.0) / 3 ** 0.5) + psi_0
    return psi_m

def TSEB_OSEB(Tr_K, T_A_K, u, ea, p, Sn, L_dn, emis, z_0M, d_0, z_u, z_T, calcG_params=[[1], 0.35], const_L=None, T0_K=[], kB=TSEB_KB_1_DEFAULT):
    Tr_K = np.asarray(Tr_K)
    T_A_K, u, ea, p, Sn, L_dn, emis, z_0M, d_0, z_u, z_T, calcG_array = map(TSEB__check_default_parameter_size, [T_A_K, u, ea, p, Sn, L_dn, emis, z_0M, d_0, z_u, z_T, calcG_params[1]], [Tr_K] * 12)
    [flag, Ln, LE, H, G, R_A] = [np.zeros(Tr_K.shape, np.float32) + np.nan for i in range(6)]
    if const_L is None:
        L = np.zeros(Tr_K.shape) + np.inf
        max_iterations = TSEB_ITERATIONS
    else:
        L = np.ones(Tr_K.shape) * const_L
        max_iterations = 1
    differentialT = False
    L_old = np.ones(Tr_K.shape)
    rho = met_calc_rho(p, ea, T_A_K)
    c_p = met_calc_c_p(p, ea)
    u_friction = MO_calc_u_star(u, z_u, L, d_0, z_0M)
    u_friction = np.maximum(TSEB_U_FRICTION_MIN, u_friction)
    L_old = np.ones(Tr_K.shape)
    L_diff = np.ones(Tr_K.shape) * float('inf')
    z_0H = res_calc_z_0H(z_0M, kB=kB)
    Ln = emis * L_dn - emis * met_calc_stephan_boltzmann(Tr_K)
    Rn = np.asarray(Sn + Ln)
    i = np.ones(Rn.shape, dtype=bool)
    G[i] = TSEB_calc_G([calcG_params[0], calcG_array], Rn, i)
    for n_iterations in range(max_iterations):
        flag = np.zeros(Tr_K.shape) + TSEB_F_ALL_FLUXES_OS
        if np.all(L_diff < TSEB_L_thres):
            break
        R_A_params = {'z_T': z_T, 'u_friction': u_friction, 'L': L, 'd_0': d_0, 'z_0H': z_0H}
        R_A, _, _ = TSEB_calc_resistances(TSEB_KUSTAS_NORMAN_1999, {'R_A': R_A_params})
        H = rho * c_p * (Tr_K - T_A_K) / R_A
        H = np.asarray(H)
        LE = np.asarray(Rn - G - H)
        flag[LE < 0] = TSEB_F_ZERO_LE_OS
        H[LE < 0] = np.minimum(H[LE < 0], Rn[LE < 0] - G[LE < 0])
        G[LE < 0] = np.maximum(G[LE < 0], Rn[LE < 0] - H[LE < 0])
        LE[LE < 0] = 0
        out = np.logical_and(i, TSEB_check_flux_lims(G, LE, H))
        flag[out] = TSEB_F_OUT_LIMS
        if const_L is None:
            L = MO_calc_L(u_friction, T_A_K, rho, c_p, H, LE)
            L_diff = np.fabs(L - L_old) / np.fabs(L_old)
            L_old = np.array(L)
            L_old[np.fabs(L_old) == 0] = 1e-36
            if not differentialT:
                u_friction = MO_calc_u_star(u, z_u, L, d_0, z_0M)
                u_friction = np.maximum(TSEB_U_FRICTION_MIN, u_friction)
    flag, Ln, LE, H, G, R_A, u_friction, L, n_iterations = map(np.asarray, (flag, Ln, LE, H, G, R_A, u_friction, L, n_iterations))
    return (flag, Ln, LE, H, G, R_A, u_friction, L, n_iterations)

def TSEB_TSEB_PT(Tr_K, vza, T_A_K, u, ea, p, Sn_C, Sn_S, L_dn, LAI, h_C, emis_C, emis_S, z_0M, d_0, z_u, z_T, leaf_width=0.1, z0_soil=0.01, alpha_PT=1.26, x_LAD=1, f_c=1.0, f_g=1.0, w_C=1.0, resistance_form=None, calcG_params=None, const_L=None, kB=TSEB_KB_1_DEFAULT, massman_profile=None):
    if calcG_params is None:
        calcG_params = [[1], 0.35]
    if resistance_form is None:
        resistance_form = [0, {}]
    if massman_profile is None:
        massman_profile = [0, []]
    Tr_K = np.asarray(Tr_K, dtype=np.float32)
    vza, T_A_K, u, ea, p, Sn_C, Sn_S, L_dn, LAI, h_C, emis_C, emis_S, z_0M, d_0, z_u, z_T, leaf_width, z0_soil, alpha_PT, x_LAD, f_c, f_g, w_C, calcG_array = map(TSEB__check_default_parameter_size, [vza, T_A_K, u, ea, p, Sn_C, Sn_S, L_dn, LAI, h_C, emis_C, emis_S, z_0M, d_0, z_u, z_T, leaf_width, z0_soil, alpha_PT, x_LAD, f_c, f_g, w_C, calcG_params[1]], [Tr_K] * 24)
    res_params = resistance_form[1]
    resistance_form = resistance_form[0]
    [Ln_S, Ln_C, H, LE, LE_C, H_C, LE_S, H_S, G, R_S, R_x, R_A, delta_Rn, Rn_S, iterations] = [np.zeros(Tr_K.shape, np.float32) + np.nan for i in range(15)]
    if const_L is None:
        L = np.zeros(Tr_K.shape) + np.inf
        max_iterations = TSEB_ITERATIONS
    else:
        L = np.ones(Tr_K.shape) * const_L
        max_iterations = 1
    rho = met_calc_rho(p, ea, T_A_K)
    c_p = met_calc_c_p(p, ea)
    z_0H = res_calc_z_0H(z_0M, kB=kB)
    omega0 = CI_calc_omega0_Kustas(LAI, f_c, x_LAD=x_LAD, isLAIeff=True)
    F = np.asarray(LAI / f_c, dtype=np.float32)
    f_theta = TSEB_calc_F_theta_campbell(vza, F, w_C=w_C, Omega0=omega0, x_LAD=x_LAD)
    del vza, ea
    u_friction = MO_calc_u_star(u, z_u, L, d_0, z_0M)
    u_friction = np.asarray(np.maximum(TSEB_U_FRICTION_MIN, u_friction), dtype=np.float32)
    L_queue = deque([np.array(L, np.float32)], 6)
    L_converged = np.zeros(Tr_K.shape, bool)
    L_diff_max = np.inf
    T_C = np.asarray(np.minimum(Tr_K, T_A_K), dtype=np.float32)
    flag, T_S = TSEB_calc_T_S(Tr_K, T_C, f_theta)
    T_AC = T_A_K.copy()
    alpha_PT_rec = np.full_like(Tr_K, alpha_PT)
    start_time = time.time()
    loop_time = time.time()
    alpha_PT_rec = np.asarray(alpha_PT, dtype=np.float32)
    for n_iterations in range(max_iterations):
        i = flag != TSEB_F_INVALID
        if np.all(L_converged[i]):
            if L_converged[i].size == 0:
                pass
            break
        current_time = time.time()
        loop_duration = current_time - loop_time
        loop_time = current_time
        total_duration = loop_time - start_time
        iterations[np.logical_and(~L_converged, flag != TSEB_F_INVALID)] = n_iterations
        flag[np.logical_and(~L_converged, flag != TSEB_F_INVALID)] = TSEB_F_ALL_FLUXES
        LE_S[np.logical_and(~L_converged, flag != TSEB_F_INVALID)] = -1
        alpha_PT_rec = np.asarray(alpha_PT + 0.1, dtype=np.float32)
        while np.any(LE_S[i] < 0):
            i = np.logical_and.reduce((LE_S < 0, ~L_converged, flag != TSEB_F_INVALID))
            alpha_PT_rec[i] -= 0.1
            alpha_PT_rec[alpha_PT_rec <= 0.0] = 0.0
            flag[np.logical_and(i, alpha_PT_rec == 0.0)] = TSEB_F_ZERO_LE
            flag[np.logical_and.reduce((i, alpha_PT_rec < alpha_PT, alpha_PT_rec > 0.0))] = TSEB_F_ZERO_LE_S
            R_A[i], R_x[i], R_S[i] = TSEB_calc_resistances(resistance_form, {'R_A': {'z_T': z_T[i], 'u_friction': u_friction[i], 'L': L[i], 'd_0': d_0[i], 'z_0H': z_0H[i]}, 'R_x': {'u_friction': u_friction[i], 'h_C': h_C[i], 'd_0': d_0[i], 'z_0M': z_0M[i], 'L': L[i], 'F': F[i], 'LAI': LAI[i], 'leaf_width': leaf_width[i], 'z0_soil': z0_soil[i], 'massman_profile': massman_profile, 'res_params': {k: res_params[k][i] for k in res_params.keys()}}, 'R_S': {'u_friction': u_friction[i], 'h_C': h_C[i], 'd_0': d_0[i], 'z_0M': z_0M[i], 'L': L[i], 'F': F[i], 'omega0': omega0[i], 'LAI': LAI[i], 'leaf_width': leaf_width[i], 'z0_soil': z0_soil[i], 'z_u': z_u[i], 'deltaT': T_S[i] - T_AC[i], 'u': u[i], 'rho': rho[i], 'c_p': c_p[i], 'f_cover': f_c[i], 'w_C': w_C[i], 'massman_profile': massman_profile, 'res_params': {k: res_params[k][i] for k in res_params.keys()}}})
            Ln_C[i], Ln_S[i] = rad_calc_L_n_Campbell(T_C[i], T_S[i], L_dn[i], LAI[i], emis_C[i], emis_S[i], x_LAD=x_LAD[i])
            delta_Rn[i] = Sn_C[i] + Ln_C[i]
            Rn_S[i] = Sn_S[i] + Ln_S[i]
            H_C[i] = TSEB_calc_H_C_PT(delta_Rn[i], f_g[i], T_A_K[i], p[i], c_p[i], alpha_PT_rec[i])
            T_C[i] = TSEB_calc_T_C_series(Tr_K[i], T_A_K[i], R_A[i], R_x[i], R_S[i], f_theta[i], H_C[i], rho[i], c_p[i])
            flag_t = np.zeros(flag.shape) + TSEB_F_ALL_FLUXES
            flag_t[i], T_S[i] = TSEB_calc_T_S(Tr_K[i], T_C[i], f_theta[i])
            flag[flag_t == TSEB_F_INVALID] = TSEB_F_FAILED_TS
            LE_S[flag_t == TSEB_F_INVALID] = 0
            _, _, R_S[i] = TSEB_calc_resistances(resistance_form, {'R_S': {'u_friction': u_friction[i], 'h_C': h_C[i], 'd_0': d_0[i], 'z_0M': z_0M[i], 'L': L[i], 'F': F[i], 'omega0': omega0[i], 'LAI': LAI[i], 'leaf_width': leaf_width[i], 'z0_soil': z0_soil[i], 'z_u': z_u[i], 'deltaT': T_S[i] - T_AC[i], 'u': u[i], 'rho': rho[i], 'c_p': c_p[i], 'f_cover': f_c[i], 'w_C': w_C[i], 'massman_profile': massman_profile, 'res_params': {k: res_params[k][i] for k in res_params.keys()}}})
            i = np.logical_and.reduce((LE_S < 0, ~L_converged, flag != TSEB_F_INVALID))
            T_AC[i] = (T_A_K[i] / R_A[i] + T_S[i] / R_S[i] + T_C[i] / R_x[i]) / (1.0 / R_A[i] + 1.0 / R_S[i] + 1.0 / R_x[i])
            H_S[i] = rho[i] * c_p[i] * (T_S[i] - T_AC[i]) / R_S[i]
            G[i] = TSEB_calc_G([calcG_params[0], calcG_array], Rn_S, i)
            LE_S[i] = Rn_S[i] - G[i] - H_S[i]
            LE_C[i] = delta_Rn[i] - H_C[i]
            noT = np.logical_and(i, LE_C == 0)
            H_S[noT] = np.minimum(H_S[noT], Rn_S[noT] - G[noT])
            G[noT] = np.maximum(G[noT], Rn_S[noT] - H_S[noT])
            LE_S[noT] = 0
            flag[noT] = TSEB_F_ZERO_LE
            out = np.logical_and(i, TSEB_check_flux_lims(G, LE_C, LE_S, H_C, H_S))
            flag[out] = TSEB_F_OUT_LIMS
            H[i] = np.asarray(H_C[i] + H_S[i], dtype=np.float32)
            LE[i] = np.asarray(LE_C[i] + LE_S[i], dtype=np.float32)
            if const_L is None:
                L[i] = MO_calc_L(u_friction[i], T_A_K[i], rho[i], c_p[i], H[i], LE[i])
                u_friction[i] = MO_calc_u_star(u[i], z_u[i], L[i], d_0[i], z_0M[i])
                u_friction[i] = np.asarray(np.maximum(TSEB_U_FRICTION_MIN, u_friction[i]), dtype=np.float32)
        if const_L is None:
            i, L_queue, L_converged, L_diff_max = TSEB_monin_obukhov_convergence(L, L_queue, L_converged, flag)
    flag, T_S, T_C, T_AC, Ln_S, Ln_C, LE_C, H_C, LE_S, H_S, G, R_S, R_x, R_A, u_friction, L, n_iterations, alpha_PT_rec = map(np.asarray, (flag, T_S, T_C, T_AC, Ln_S, Ln_C, LE_C, H_C, LE_S, H_S, G, R_S, R_x, R_A, u_friction, L, iterations, alpha_PT_rec))
    return (flag, T_S, T_C, T_AC, Ln_S, Ln_C, LE_C, H_C, LE_S, H_S, G, R_S, R_x, R_A, u_friction, L, n_iterations, alpha_PT_rec)

def TSEB__L_diff(L, L_old):
    L_diff = np.asarray(np.fabs(L - L_old) / np.fabs(L_old), dtype=np.float32)
    L_diff[np.isnan(L_diff)] = float('inf')
    return L_diff

def TSEB__check_default_parameter_size(parameter, input_array):
    parameter = np.asarray(parameter, dtype=np.float32)
    if parameter.size == 1:
        parameter = np.ones(input_array.shape) * parameter
        return np.asarray(parameter, dtype=np.float32)
    elif parameter.shape != input_array.shape:
        raise ValueError('dimension mismatch between parameter array and input array with shapes %s and %s' % (parameter.shape, input_array.shape))
    else:
        return np.asarray(parameter, dtype=np.float32)

def TSEB_calc_F_theta_campbell(theta, F, w_C=1, Omega0=1, x_LAD=1):
    w_C = 1.0 / w_C
    OmegaTheta = Omega0 / (Omega0 + (1.0 - Omega0) * np.exp(-2.2 * np.radians(theta) ** (3.8 - 0.46 * w_C)))
    K_be = rad_calc_K_be_Campbell(theta, x_LAD)
    ftheta = 1.0 - np.exp(-K_be * OmegaTheta * F)
    return np.asarray(ftheta, dtype=np.float32)

def TSEB_calc_G(calcG_params, Rn_S, i=None):
    if calcG_params[0][0] != 1:
        raise ValueError('Selected soil heat law is G/Rn=0.35')
    if i is None:
        i = np.ones(Rn_S.shape, dtype=bool)
    return np.asarray(Rn_S[i] * calcG_params[1][i])

def TSEB_calc_H_C_PT(delta_R_ni, f_g, T_A_K, P, c_p, alpha):
    s = met_calc_delta_vapor_pressure(T_A_K)
    s = s * 10
    Lambda = met_calc_lambda(T_A_K)
    gama = met_calc_psicr(c_p, P, Lambda)
    s_gama = s / (s + gama)
    H_C = delta_R_ni * (1.0 - alpha * f_g * s_gama)
    return np.asarray(H_C, dtype=np.float32)

def TSEB_calc_T_C_series(Tr_K, T_A_K, R_A, R_x, R_S, f_theta, H_C, rho, c_p):
    T_R_K_4 = Tr_K ** 4
    T_C_lin = (T_A_K / R_A + Tr_K / (R_S * (1.0 - f_theta)) + H_C * R_x / (rho * c_p) * (1.0 / R_A + 1.0 / R_S + 1.0 / R_x)) / (1.0 / R_A + 1.0 / R_S + f_theta / (R_S * (1.0 - f_theta)))
    T_D = T_C_lin * (1 + R_S / R_A) - H_C * R_x / (rho * c_p) * (1.0 + R_S / R_x + R_S / R_A) - T_A_K * R_S / R_A
    delta_T_C = (T_R_K_4 - f_theta * T_C_lin ** 4 - (1.0 - f_theta) * T_D ** 4) / (4.0 * (1.0 - f_theta) * T_D ** 3 * (1.0 + R_S / R_A) + 4.0 * f_theta * T_C_lin ** 3)
    T_C = T_C_lin + delta_T_C
    return np.asarray(T_C, dtype=np.float32)

def TSEB_calc_T_S(T_R, T_C, f_theta):
    T_R, T_C, f_theta = map(np.asarray, (T_R, T_C, f_theta))
    T_temp = T_R ** 4 - f_theta * T_C ** 4
    T_S = np.zeros(T_R.shape, np.float32)
    flag = np.zeros(T_R.shape, np.int32) + TSEB_F_ALL_FLUXES
    T_S[T_temp >= 0] = (T_temp[T_temp >= 0] / (1.0 - f_theta[T_temp >= 0])) ** 0.25
    T_S[np.logical_or(T_temp < 0, np.isnan(T_temp))] = 1e-06
    flag[np.logical_or(T_temp < 0, np.isnan(T_temp))] = TSEB_F_INVALID
    return (np.asarray(flag, dtype=np.int32), np.asarray(T_S, dtype=np.float32))

def TSEB_calc_resistances(res_form, res_types):
    if res_form != 0:
        raise ValueError('Selected method requires KN99 resistance')
    R_A = 0
    R_x = 0
    R_S = 0
    u_C = None
    if 'R_A' in res_types:
        a = res_types['R_A']
        R_A = res_calc_R_A(a['z_T'], a['u_friction'], a['L'], a['d_0'], a['z_0H'])
    if 'R_x' in res_types:
        a = res_types['R_x']
        if a['massman_profile'][0] > 0:
            raise ValueError('Selected wind law is Goudriaan')
        u_C = np.maximum(wnd_calc_u_C_star(a['u_friction'], a['h_C'], a['d_0'], a['z_0M'], a['L']), TSEB_U_C_MIN)
        velocity = np.maximum(wnd_calc_u_Goudriaan(u_C, a['h_C'], a['F'], a['leaf_width'], a['d_0'] + a['z_0M']), TSEB_U_C_MIN)
        R_x = res_calc_R_x_Norman(a['LAI'], a['leaf_width'], velocity, a['res_params'])
    if 'R_S' in res_types:
        a = res_types['R_S']
        if a['massman_profile'][0] > 0:
            raise ValueError('Selected wind law is Goudriaan')
        if u_C is None:
            u_C = np.maximum(wnd_calc_u_C_star(a['u_friction'], a['h_C'], a['d_0'], a['z_0M'], a['L']), TSEB_U_C_MIN)
        velocity = np.maximum(wnd_calc_u_Goudriaan(u_C, a['h_C'], a['LAI'], a['leaf_width'], a['z0_soil']), TSEB_U_S_MIN)
        R_S = res_calc_R_S_Kustas(velocity, a['deltaT'], params=a['res_params'])
    return (np.asarray(np.clip(R_A, TSEB_R_A_MIN, TSEB_R_A_MAX)), np.asarray(np.clip(R_x, TSEB_RES_MIN, TSEB_RES_MAX)), np.asarray(np.clip(R_S, TSEB_RES_MIN, TSEB_RES_MAX)))

def TSEB_check_flux_lims(*fluxes, lims=TSEB_FLUX_LIMS):
    flag = np.zeros_like(fluxes[0], dtype=bool)
    for flux in fluxes:
        out = np.logical_or(flux < lims[0], flux > lims[1])
        flag = np.logical_or(flag, out)
    return flag

def TSEB_monin_obukhov_convergence(l_mo, l_queue, l_converged, flag):
    l_new = np.array(l_mo)
    l_new[l_new == 0] = 1e-36
    l_queue.appendleft(l_new)
    i = np.logical_and(~l_converged, flag != TSEB_F_INVALID)
    if np.sum(i) <= 1 or np.size(i) == 0:
        return (i, l_queue, l_converged, np.inf)
    if len(l_queue) >= 4:
        i = np.logical_and(~l_converged, flag != TSEB_F_INVALID)
        if np.any(i):
            l_converged[i] = np.logical_and(TSEB__L_diff(l_queue[0][i], l_queue[2][i]) < TSEB_L_thres, TSEB__L_diff(l_queue[1][i], l_queue[3][i]) < TSEB_L_thres)
    if len(l_queue) == 6:
        i = np.logical_and(~l_converged, flag != TSEB_F_INVALID)
        if np.any(i):
            l_converged[i] = np.logical_and.reduce((TSEB__L_diff(l_queue[0][i], l_queue[3][i]) < TSEB_L_thres, TSEB__L_diff(l_queue[1][i], l_queue[4][i]) < TSEB_L_thres, TSEB__L_diff(l_queue[2][i], l_queue[5][i]) < TSEB_L_thres))
    if np.sum(i) == 0 or np.size(i) == 0:
        return (i, l_queue, l_converged, np.inf)
    l_diff_max = np.nanmax(TSEB__L_diff(l_queue[0][i], l_queue[1][i]))
    return (i, l_queue, l_converged, l_diff_max)

def met_calc_c_p(p, ea):
    q = met_epsilon * ea / (p + (met_epsilon - 1.0) * ea)
    c_p = (1.0 - q) * met_c_pd + q * met_c_pv
    return np.asarray(c_p)

def met_calc_delta_vapor_pressure(T_K):
    T_C = T_K - 273.15
    s = 4098.0 * (0.6108 * np.exp(17.27 * T_C / (T_C + 237.3))) / (T_C + 237.3) ** 2
    return np.asarray(s)

def met_calc_lambda(T_A_K):
    Lambda = 1000000.0 * (2.501 - 0.002361 * (T_A_K - 273.15))
    return np.asarray(Lambda)

def met_calc_psicr(c_p, p, Lambda):
    psicr = c_p * p / (met_epsilon * Lambda)
    return np.asarray(psicr)

def met_calc_rho(p, ea, T_A_K):
    rho = p * 100.0 / (met_R_d * T_A_K) * (1.0 - (1.0 - met_epsilon) * ea / p)
    return np.asarray(rho)

def met_calc_stephan_boltzmann(T_K):
    M = met_sb * T_K ** 4
    return np.asarray(M)

def met_calc_sun_angles(lat, lon, stdlon, doy, ftime):
    lat, lon, stdlon, doy, ftime = map(np.asarray, (lat, lon, stdlon, doy, ftime))
    declination = 0.409 * np.sin(2.0 * np.pi * doy / 365.0 - 1.39)
    EOT = 0.258 * np.cos(declination) - 7.416 * np.sin(declination) - 3.648 * np.cos(2.0 * declination) - 9.228 * np.sin(2.0 * declination)
    LC = (stdlon - lon) / 15.0
    time_corr = -EOT / 60.0 + LC
    solar_time = ftime - time_corr
    w = np.asarray((solar_time - 12.0) * 15.0)
    sin_thetha = np.cos(np.radians(w)) * np.cos(declination) * np.cos(np.radians(lat)) + np.sin(declination) * np.sin(np.radians(lat))
    sun_elev = np.arcsin(sin_thetha)
    sza = np.pi / 2.0 - sun_elev
    sza = np.asarray(np.degrees(sza))
    cos_phi = np.asarray((np.sin(declination) * np.cos(np.radians(lat)) - np.cos(np.radians(w)) * np.cos(declination) * np.sin(np.radians(lat))) / np.cos(sun_elev))
    saa = np.zeros(sza.shape)
    saa[w <= 0.0] = np.degrees(np.arccos(cos_phi[w <= 0.0]))
    saa[w > 0.0] = 360.0 - np.degrees(np.arccos(cos_phi[w > 0.0]))
    return (np.asarray(sza), np.asarray(saa))

def met_calc_vapor_pressure(T_K):
    T_C = T_K - 273.15
    ea = 6.112 * np.exp(17.67 * T_C / (T_C + 243.5))
    return np.asarray(ea)

def rad__calc_taud(x_lad, lai):
    taud = 0
    for angle in range(0, 90, rad_TAUD_STEP_SIZE_DEG):
        angle = np.radians(angle)
        akd = rad_calc_K_be_Campbell(angle, x_lad, radians=True)
        taub = np.exp(-akd * lai)
        taud += taub * np.cos(angle) * np.sin(angle) * np.radians(rad_TAUD_STEP_SIZE_DEG)
    return 2.0 * taud

def rad_calc_K_be_Campbell(theta, x_lad=1, radians=False):
    if not radians:
        theta = np.radians(theta)
    K_be = np.sqrt(x_lad ** 2 + np.tan(theta) ** 2) / (x_lad + 1.774 * (x_lad + 1.182) ** (-0.733))
    return K_be

def rad_calc_L_n_Campbell(T_C, T_S, L_dn, lai, emisVeg, emisGrd, x_LAD=1):
    L_C = emisVeg * met_calc_stephan_boltzmann(T_C)
    L_C[np.isnan(L_C)] = 0
    L_S = emisGrd * met_calc_stephan_boltzmann(T_S)
    L_S[np.isnan(L_S)] = 0
    _, albl, _, taudl = rad_calc_spectra_Cambpell(lai, np.zeros(emisVeg.shape), 1.0 - emisVeg, np.zeros(emisVeg.shape), 1.0 - emisGrd, x_lad=x_LAD, lai_eff=None)
    L_nS = emisGrd * taudl * L_dn + emisGrd * (1.0 - taudl) * L_C - L_S
    L_nC = (1 - albl) * (1.0 - taudl) * (L_dn + L_S) - 2.0 * (1.0 - taudl) * L_C
    L_nC[np.isnan(L_nC)] = 0
    L_nS[np.isnan(L_nS)] = 0
    return (np.asarray(L_nC), np.asarray(L_nS))

def rad_calc_Sn_Campbell(lai, sza, S_dn_dir, S_dn_dif, fvis, fnir, rho_leaf_vis, tau_leaf_vis, rho_leaf_nir, tau_leaf_nir, rsoilv, rsoiln, x_LAD=1, LAI_eff=None):
    rho_leaf = np.array((rho_leaf_vis, rho_leaf_nir))
    tau_leaf = np.array((tau_leaf_vis, tau_leaf_nir))
    rho_soil = np.array((rsoilv, rsoiln))
    albb, albd, taubt, taudt = rad_calc_spectra_Cambpell(lai, sza, rho_leaf, tau_leaf, rho_soil, x_lad=x_LAD, lai_eff=LAI_eff)
    Sn_C = (1.0 - taubt[0]) * (1.0 - albb[0]) * S_dn_dir * fvis + (1.0 - taubt[1]) * (1.0 - albb[1]) * S_dn_dir * fnir + (1.0 - taudt[0]) * (1.0 - albd[0]) * S_dn_dif * fvis + (1.0 - taudt[1]) * (1.0 - albd[1]) * S_dn_dif * fnir
    Sn_S = taubt[0] * (1.0 - rsoilv) * S_dn_dir * fvis + taubt[1] * (1.0 - rsoiln) * S_dn_dir * fnir + taudt[0] * (1.0 - rsoilv) * S_dn_dif * fvis + taudt[1] * (1.0 - rsoiln) * S_dn_dif * fnir
    return (np.asarray(Sn_C), np.asarray(Sn_S))

def rad_calc_difuse_ratio(S_dn, sza, press=1013.25, SOLAR_CONSTANT=1320):
    S_dn, sza, press = map(np.asarray, (S_dn, sza, press))
    difvis, difnir, fvis, fnir = [np.zeros(S_dn.shape) for i in range(4)]
    fvis = fvis + 0.6
    fnir = fnir + 0.4
    Rdirvis, Rdifvis, Rdirnir, Rdifnir = rad_calc_potential_irradiance_weiss(sza, press=press, SOLAR_CONSTANT=SOLAR_CONSTANT)
    potvis = np.asarray(Rdirvis + Rdifvis)
    potvis[potvis <= 0] = 1e-06
    potnir = np.asarray(Rdirnir + Rdifnir)
    potnir[potnir <= 0] = 1e-06
    fclear = S_dn / (potvis + potnir)
    fclear = np.minimum(1.0, fclear)
    fvis = potvis / (potvis + potnir)
    fnir = potnir / (potvis + potnir)
    fvis = np.clip(fvis, 0.0, 1.0)
    fnir = 1.0 - fvis
    ratiox = np.asarray(fclear)
    ratiox[fclear > 0.9] = 0.9
    dirvis = Rdirvis / potvis * (1.0 - ((0.9 - ratiox) / 0.7) ** 0.6667)
    ratiox = np.asarray(fclear)
    ratiox[fclear > 0.88] = 0.88
    dirnir = Rdirnir / potnir * (1.0 - ((0.88 - ratiox) / 0.68) ** 0.6667)
    dirvis = np.clip(dirvis, 0.0, 1.0)
    dirnir = np.clip(dirnir, 0.0, 1.0)
    difvis = 1.0 - dirvis
    difnir = 1.0 - dirnir
    return (np.asarray(difvis), np.asarray(difnir), np.asarray(fvis), np.asarray(fnir))

def rad_calc_potential_irradiance_weiss(sza, press=1013.25, SOLAR_CONSTANT=1320, fnir_ini=0.5455):
    sza, press = map(np.asarray, (sza, press))
    Rdirvis, Rdifvis, Rdirnir, Rdifnir, w = [np.zeros(sza.shape) for i in range(5)]
    coszen = np.cos(np.radians(sza))
    i = sza < 90
    airmas = 1.0 / coszen
    Sco_vis = SOLAR_CONSTANT * (1.0 - fnir_ini)
    Sco_nir = SOLAR_CONSTANT * fnir_ini
    Rdirvis[i] = (Sco_vis * np.exp(-0.185 * (press[i] / 1013.25) * airmas[i]) - w[i]) * coszen[i]
    Rdirvis = np.maximum(0, Rdirvis)
    Rdifvis[i] = 0.4 * (Sco_vis * coszen[i] - Rdirvis[i])
    Rdifvis = np.maximum(0, Rdifvis)
    w = SOLAR_CONSTANT * 10 ** (-1.195 + 0.4459 * np.log10(airmas[i]) - 0.0345 * np.log10(airmas[i]) ** 2)
    Rdirnir[i] = (Sco_nir * np.exp(-0.06 * (press[i] / 1013.25) * airmas[i]) - w) * coszen[i]
    Rdirnir = np.maximum(0, Rdirnir)
    Rdifnir[i] = 0.6 * (Sco_nir * coszen[i] - Rdirnir[i] - w * coszen[i])
    Rdifnir = np.maximum(0, Rdifnir)
    Rdirvis, Rdifvis, Rdirnir, Rdifnir = map(np.asarray, (Rdirvis, Rdifvis, Rdirnir, Rdifnir))
    return (Rdirvis, Rdifvis, Rdirnir, Rdifnir)

def rad_calc_spectra_Cambpell(lai, sza, rho_leaf, tau_leaf, rho_soil, x_lad=1, lai_eff=None):
    amean = 1.0 - rho_leaf - tau_leaf
    amean_sqrt = np.sqrt(amean)
    del rho_leaf, tau_leaf, amean
    if lai_eff is None:
        lai_eff = np.asarray(lai)
    else:
        lai_eff = np.asarray(lai_eff)
    taud = rad__calc_taud(x_lad, lai)
    akd = -np.log(taud) / lai
    rcpy = (1.0 - amean_sqrt) / (1.0 + amean_sqrt)
    rdcpy = 2.0 * akd * rcpy / (akd + 1.0)
    expfac = amean_sqrt * akd * lai
    del akd
    neg_exp, d_neg_exp = (np.exp(-expfac), np.exp(-2.0 * expfac))
    xnum = (rdcpy * rdcpy - 1.0) * neg_exp
    xden = rdcpy * rho_soil - 1.0 + rdcpy * (rdcpy - rho_soil) * d_neg_exp
    taudt = xnum / xden
    del xnum, xden
    fact = (rdcpy - rho_soil) / (rdcpy * rho_soil - 1.0) * d_neg_exp
    albd = (rdcpy + fact) / (1.0 + rdcpy * fact)
    del rdcpy, fact
    akb = rad_calc_K_be_Campbell(sza, x_lad)
    rbcpy = 2.0 * akb * rcpy / (akb + 1.0)
    del rcpy, sza, x_lad
    expfac = amean_sqrt * akb * lai_eff
    neg_exp, d_neg_exp = (np.exp(-expfac), np.exp(-2.0 * expfac))
    del amean_sqrt, akb, lai_eff
    xnum = (rbcpy * rbcpy - 1.0) * neg_exp
    xden = rbcpy * rho_soil - 1.0 + rbcpy * (rbcpy - rho_soil) * d_neg_exp
    taubt = xnum / xden
    del xnum, xden
    fact = (rbcpy - rho_soil) / (rbcpy * rho_soil - 1.0) * d_neg_exp
    del expfac
    albb = (rbcpy + fact) / (1.0 + rbcpy * fact)
    del rbcpy, fact
    taubt, taudt, albb, albd, rho_soil = map(np.array, [taubt, taudt, albb, albd, rho_soil])
    taubt[np.isnan(taubt)] = 1
    taudt[np.isnan(taudt)] = 1
    albb[np.isnan(albb)] = rho_soil[np.isnan(albb)]
    albd[np.isnan(albd)] = rho_soil[np.isnan(albd)]
    return (albb, albd, taubt, taudt)

def res_calc_R_A(z_T, ustar, L, d_0, z_0H):
    z_T, ustar, L, d_0, z_0H = map(np.asarray, (z_T, ustar, L, d_0, z_0H))
    R_A_log = np.asarray(np.log((z_T - d_0) / z_0H))
    L[L == 0] = 1e-36
    Psi_H = MO_calc_Psi_H((z_T - d_0) / L)
    Psi_H0 = MO_calc_Psi_H(z_0H / L)
    del L, z_0H, z_T
    i = ustar != 0
    R_A = np.asarray(np.ones(ustar.shape) * float('inf'))
    R_A[i] = (R_A_log[i] - Psi_H[i] + Psi_H0[i]) / (ustar[i] * res_KARMAN)
    return np.asarray(R_A)

def res_calc_R_S_Kustas(u_S, deltaT, params=None):
    if params is None:
        params = {}
    if 'KN_b' in params:
        b = params['KN_b']
    else:
        b = res_KN_b
    if 'KN_c' in params:
        c = params['KN_c']
    else:
        c = res_KN_c
    u_S, deltaT = map(np.asarray, (u_S, deltaT))
    deltaT = np.asarray(np.maximum(deltaT, 0.0))
    R_S = 1.0 / (c * deltaT ** (1.0 / 3.0) + b * u_S)
    return np.asarray(R_S)

def res_calc_R_x_Norman(LAI, leaf_width, u_d_zm, params=None):
    if params is None:
        params = {}
    if 'KN_C_dash' in params:
        C_dash = params['KN_C_dash']
    else:
        C_dash = res_KN_C_dash
    R_x = C_dash / LAI * (leaf_width / u_d_zm) ** 0.5
    return np.asarray(R_x)

def res_calc_roughness(LAI, h_C, w_C=1, landcover=res_CROP, f_c=None):
    if np.any(np.asarray(landcover) != res_CROP):
        raise ValueError('Selected nominal landcover is crop')
    return (np.asarray(np.asarray(h_C) * 0.125), np.asarray(np.asarray(h_C) * 0.65))

def res_calc_z_0H(z_0M, kB=0):
    z_0H = z_0M / np.exp(kB)
    return np.asarray(z_0H)

def wnd_calc_A_Goudriaan(h_C, LAI, leaf_width):
    k3_prime = 0.28
    a = k3_prime * LAI ** (2.0 / 3.0) * h_C ** (1.0 / 3.0) * leaf_width ** (-1.0 / 3.0)
    return np.asarray(a)

def wnd_calc_u_C_star(u_friction, h_C, d_0, z_0M, L=float('inf')):
    Psi_M = MO_calc_Psi_M((h_C - d_0) / L)
    Psi_M0 = MO_calc_Psi_M(z_0M / L)
    u_C = u_friction * (np.log((h_C - d_0) / z_0M) - Psi_M + Psi_M0) / wnd_KARMAN
    return np.asarray(u_C)

def wnd_calc_u_Goudriaan(u_C, h_C, LAI, leaf_width, z):
    a = wnd_calc_A_Goudriaan(h_C, LAI, leaf_width)
    del LAI, leaf_width
    u_z = u_C * np.exp(-a * (1.0 - z / h_C))
    return np.asarray(u_z)
