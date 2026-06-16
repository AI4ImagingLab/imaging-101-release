import json
import os
import numpy as np

# ==========================================
# Load Physics Constants from meta_data.json
# ==========================================

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

def _load_meta_data():
    with open(os.path.join(_DATA_DIR, 'meta_data.json'), 'r') as f:
        return json.load(f)

_META = _load_meta_data()

MOL_CONST = _META['molecular_constants']
UNIV_CONST = _META['universal_constants']
CHI_NRS = _META['chi_nrs']


# ==========================================
# Line Shape Functions
# ==========================================

def gaussian_line(w, w0, sigma):
    if sigma == 0:
        return np.zeros_like(w)
    return 2 / sigma * (np.log(2) / np.pi) ** 0.5 * np.exp(-4 * np.log(2) * ((w - w0) / sigma) ** 2)


def lorentz_line(w, w0, sigma):
    if sigma == 0:
        return np.zeros_like(w)
    return 1 / np.pi * (sigma / 2) / ((w - w0) ** 2 + sigma ** 2 / 4)


def asym_Gaussian(w, w0, sigma, k, a_sigma, a_k, offset, power_factor=1.):
    response_low = np.exp(-abs((w[w <= w0] - w0) / (sigma - a_sigma)) ** (k - a_k))
    response_high = np.exp(-abs((w[w > w0] - w0) / (sigma + a_sigma)) ** (k + a_k))
    response = (np.append(response_low, response_high) + offset) ** power_factor
    max_val = response.max()
    if max_val == 0:
        return response
    return np.nan_to_num(response / max_val)


def asym_Voigt(w, w0, sigma, k, a_sigma, a_k, sigma_L_l, sigma_L_h, offset, power_factor=1.):
    response_low = np.exp(-abs((w - w0) / (sigma - a_sigma)) ** (k - a_k))
    response_high = np.exp(-abs((w - w0) / (sigma + a_sigma)) ** (k + a_k))

    l_line_l = lorentz_line(w, (w[0] + w[-1]) / 2, sigma_L_l)
    l_line_h = lorentz_line(w, (w[0] + w[-1]) / 2, sigma_L_h)

    response_low = np.convolve(response_low, l_line_l, 'same')
    response_high = np.convolve(response_high, l_line_h, 'same')

    if response_low.max() > 0:
        response_low = response_low / response_low.max()
    if response_high.max() > 0:
        response_high = response_high / response_high.max()

    response = (np.append(response_low[np.where(w <= w0)], response_high[np.where(w > w0)]) + offset) ** power_factor

    max_val = response.max()
    if max_val == 0:
        return response
    return np.nan_to_num(response / max_val)


def downsample(w, w_fine, spec_fine, mode='local_mean'):
    if mode == 'interp':
        return np.interp(w, w_fine, spec_fine)
    elif mode == 'local_mean':
        hw = int((w[1] - w[0]) / (w_fine[1] - w_fine[0]) / 2)
        if hw < 1:
            hw = 1
        w_fine = np.array(w_fine)
        idx = np.searchsorted(w_fine, w)
        idx[idx >= len(w_fine)] = len(w_fine) - 1
        idx[idx < 0] = 0

        downsampled = []
        for i in idx:
            start = max(0, i - hw)
            end = min(len(spec_fine), i + hw + 1)
            if start >= end:
                downsampled.append(spec_fine[i])
            else:
                downsampled.append(np.mean(spec_fine[start:end]))
        return np.array(downsampled)
    return np.interp(w, w_fine, spec_fine)


# ==========================================
# LineStrength Class
# ==========================================

class LineStrength:
    def __init__(self, species='N2'):
        self.mc_dict = MOL_CONST[species]
        self.Const_D = UNIV_CONST['Const_D']

    def int_corr(self, j, branch=0):
        mc = self.mc_dict
        if branch == 0:  # Q branch
            pt = j * (j + 1) / (2 * j - 1) / (2 * j + 3)
            cd = 1 - 6 * mc['Be'] ** 2 / mc['we'] ** 2 * j * (j + 1)
        elif branch == -2:  # O branch
            pt = 3 * j * (j - 1) / 2 / (2 * j + 1) / (2 * j - 1)
            cd = (1 + 4 * mc['Be'] / mc['we'] * mc['mu'] * (2 * j - 1)) ** 2
        elif branch == 2:  # S branch
            pt = 3 * (j + 1) * (j + 2) / 2 / (2 * j + 1) / (2 * j + 3)
            cd = (1 - 4 * mc['Be'] / mc['we'] * mc['mu'] * (2 * j + 3)) ** 2
        else:
            return 0, 1
        return pt, cd

    def term_values(self, v, j, mode='sum'):
        mc = self.mc_dict
        Bv = mc['Be'] - mc['alpha_e'] * (v + 0.5) + mc['gamma_e'] * (v + 0.5) ** 2
        Dv = mc['De'] + mc['beta_e'] * (v + 0.5)
        Hv = mc['H0'] + mc['He'] * (v + 0.5)
        Fv = Bv * j * (j + 1) - Dv * j ** 2 * (j + 1) ** 2 + Hv * j ** 3 * (j + 1) ** 3

        Gv = (mc['we'] * (v + 0.5) - mc['wx'] * (v + 0.5) ** 2
              + mc['wy'] * (v + 0.5) ** 3 + mc['wz'] * (v + 0.5) ** 4)

        if mode == 'sum':
            return Gv + Fv
        if mode == 'Gv':
            return Gv
        if mode == 'Fv':
            return Fv
        return 0

    def line_pos(self, v, j, branch=0):
        return self.term_values(v + 1, j + branch) - self.term_values(v, j)

    def pop_factor(self, T, v, j, branch=0, del_Tv=0.0):
        def rho_v(v_):
            return np.exp(-1.44 / (T + del_Tv) * self.term_values(v_, 0, mode='Gv'))

        def rho_r(v_, j_):
            gj = 3 * (2 + (-1) * (j_ % 2))
            return (2 * j_ + 1) * gj * np.exp(-1.44 / T * self.term_values(v_, j_, mode='Fv'))

        Qv = rho_v(np.arange(20)).sum()
        Qr = rho_r(v, np.arange(100)).sum()

        f_low = 1 / Qv / Qr * rho_v(v) * rho_r(v, j)
        f_up = 1 / Qv / Qr * rho_v(v + 1) * rho_r(v + 1, j + branch)
        return f_low - f_up

    def doppler_lw(self, T, nu_0=2300.):
        return nu_0 * (T / self.mc_dict['MW']) ** 0.5 * self.Const_D


# ==========================================
# Forward Operator
# ==========================================

def forward_operator(x_params):
    """
    Generates a synthetic CARS spectrum based on physical parameters.

    Args:
        x_params (dict): Dictionary containing:
            - 'nu': Wavenumber axis (array)
            - 'temperature': Gas temperature (K)
            - 'pressure': Pressure (bar)
            - 'x_mol': Mole fraction of resonant species
            - 'species': 'N2'
            - 'pump_lw': Pump laser linewidth
            - 'nu_shift': Spectral shift
            - 'nu_stretch': Spectral stretch
            - 'slit_params': list/tuple for slit function

    Returns:
        np.ndarray: Synthetic intensity spectrum I_as
    """
    nu_expt = x_params['nu']
    T = x_params['temperature']
    P = x_params['pressure']
    x_mol = x_params['x_mol']
    pump_lw = x_params['pump_lw']
    nu_shift = x_params.get('nu_shift', 0)
    nu_stretch = x_params.get('nu_stretch', 1.0)

    ls = LineStrength(x_params.get('species', 'N2'))
    Const_N = UNIV_CONST["Const_N"]
    C_light = UNIV_CONST["c"]

    # Fine grid for calculation
    nu_fine_grid_step = 0.05
    nu_expt_mod = nu_expt * nu_stretch + nu_shift
    start_nu = nu_expt_mod[0] - 10
    end_nu = nu_expt_mod[-1] + 10
    nu_s = np.arange(start_nu, end_nu, nu_fine_grid_step)

    # Relaxation matrix parameters for N2
    fit_param_N2 = [0.0231, 1.67, 1.21, 0.1487]

    js = 30
    gamma_mat = np.zeros([js, js])

    Ej = ls.term_values(0, np.arange(js), 'Fv')
    for _i in range(js):
        for _j in range(_i + 1, js):
            del_E = Ej[_j] - Ej[_i]
            if abs(_i - _j) % 2 == 0:
                alpha, beta, sigma, m = fit_param_N2
                _term_1 = (1 - np.exp(-m)) / (1 - np.exp(-m * T / 295)) * (295 / T) ** 0.5
                _term_2 = ((1 + 1.5 * 1.44 * Ej[_i] / T / sigma) / (1 + 1.5 * 1.44 * Ej[_i] / T)) ** 2
                gamma_ji = alpha * P / 1.01325 * _term_1 * _term_2 * np.exp(-beta * del_E * 1.44 / T)
                gamma_ij = gamma_ji * (2 * _i + 1) / (2 * _j + 1) * np.exp(del_E * 1.44 / T)
                gamma_mat[_j, _i], gamma_mat[_i, _j] = gamma_ji, gamma_ij
    for _i in range(js):
        gamma_mat[_i, _i] = -np.sum(gamma_mat[:, _i])

    # Compute susceptibility
    chi_rs = np.zeros_like(nu_s, dtype='complex128')
    branches = (0, 2, -2)
    vs = 2
    _js = np.arange(js)

    for _branch in branches:
        for _v in range(vs):
            nu_raman = ls.line_pos(_v, _js, branch=_branch)
            K_mat = np.diag(nu_raman) + gamma_mat * 1j
            eigvals, eigvec = np.linalg.eig(K_mat)
            eigvec_inv = np.linalg.inv(eigvec)

            del_pop = ls.pop_factor(T, _v, _js, branch=_branch)

            pt_coeff, cd_coeff = np.zeros(js), np.zeros(js)
            for k_idx, k_val in enumerate(_js):
                pt, cd = ls.int_corr(k_val, _branch)
                pt_coeff[k_idx] = pt
                cd_coeff[k_idx] = cd

            pol_ratio = ls.mc_dict['G/A']
            Const_Raman = ls.mc_dict['Const_Raman']

            if _branch in (2, -2):
                d_sq = Const_Raman ** 2 * (4 / 45) * pt_coeff * pol_ratio ** 2 * (_v + 1) * cd_coeff
            else:
                d_sq = Const_Raman ** 2 * (1 + (4 / 45) * pt_coeff * pol_ratio ** 2) * (_v + 1) * cd_coeff

            d = d_sq ** 0.5

            _term_l = d @ eigvec
            _term_r = eigvec_inv @ np.diag(del_pop) @ d
            _term = _term_l * _term_r

            for _j in _js:
                _term_b = ((-nu_s + np.real(eigvals[_j])) ** 2 + np.imag(eigvals[_j]) ** 2)
                chi_rs += 1 / 2 * _term[_j] * np.conj(-nu_s + eigvals[_j]) / _term_b

    chi_rs = chi_rs / 2 / np.pi / C_light

    # Non-resonant background
    chi_nrs_dict = CHI_NRS["SET 1"]
    chi_val = chi_nrs_dict["SPECIES"]["N2"] * x_mol
    chi_nrs_eff = chi_val * (P / chi_nrs_dict["P0"] * chi_nrs_dict["T0"] / T) * 1e-18

    num_density = P / T * Const_N
    chi_total = (x_mol * num_density * chi_rs + chi_nrs_eff) * 1e15

    # Intensity
    I_as = np.abs(chi_total) ** 2

    # Pump laser linewidth convolution
    if pump_lw > 0:
        pump_ls = gaussian_line(nu_s, (nu_s[0] + nu_s[-1]) / 2, pump_lw)
        chi_convol = np.convolve(chi_total, pump_ls, 'same')
        d_nu = nu_s[1] - nu_s[0]
        I_as = 0.5 * (I_as + d_nu * np.abs(chi_convol) ** 2)

    # Instrument slit function
    slit_params = x_params.get('slit_params', [0.5, 2.0, 0, 0])
    slit_func = asym_Gaussian(nu_s, (nu_s[0] + nu_s[-1]) / 2,
                              sigma=slit_params[0], k=slit_params[1],
                              a_sigma=slit_params[2], a_k=slit_params[3], offset=0)
    I_as = np.convolve(I_as, slit_func, 'same')

    # Downsample to experimental grid
    I_final = downsample(nu_expt_mod, nu_s, I_as, mode='local_mean')

    if I_final.max() > 0:
        I_final /= I_final.max()

    return I_final
