"""Explicit published circle, phase and cable-calibration mathematics.
Adapted from Sebastian Probst et al., resonator_tools commit dc5fe575.
Copyright original authors; LGPL-3.0-or-later, see LICENSE.
Only the notch method's called numerical functions are retained; no framework,
file readers, plotting, GUI, photon calculations or upstream imports.
"""
from __future__ import annotations
import numpy as np
import numpy.typing as npt
import scipy.optimize as spopt
from scipy import stats
from typing import Any
FloatArray = npt.NDArray[np.float64]
ComplexArray = npt.NDArray[np.complex128]
def _S21_notch(f: FloatArray, fr: float=10000000000.0, Ql: float=900, Qc: float=1000.0, phi: float=0.0, a: float=1.0, alpha: float=0.0, delay: float=0.0) -> ComplexArray:
    """
        full model for notch type resonances
        """
    return a * np.exp(complex(0, alpha)) * np.exp(-2j * np.pi * f * delay) * (1.0 - Ql / Qc * np.exp(1j * phi) / (1.0 + 2j * Ql * (f - fr) / fr))

def _center(z_data: ComplexArray, zc: complex) -> ComplexArray:
    return z_data - zc

def _dist(x: FloatArray) -> FloatArray:
    np.absolute(x, x)
    c = (x > np.pi).astype(int)
    return x + c * (-2.0 * x + 2.0 * np.pi)

def _fit_circle(z_data: ComplexArray, refine_results: bool=False) -> tuple[float, float, float]:

    def calc_moments(z_data):
        xi = z_data.real
        xi_sqr = xi * xi
        yi = z_data.imag
        yi_sqr = yi * yi
        zi = xi_sqr + yi_sqr
        Nd = float(len(xi))
        xi_sum = xi.sum()
        yi_sum = yi.sum()
        zi_sum = zi.sum()
        xiyi_sum = (xi * yi).sum()
        xizi_sum = (xi * zi).sum()
        yizi_sum = (yi * zi).sum()
        return np.array([[(zi * zi).sum(), xizi_sum, yizi_sum, zi_sum], [xizi_sum, xi_sqr.sum(), xiyi_sum, xi_sum], [yizi_sum, xiyi_sum, yi_sqr.sum(), yi_sum], [zi_sum, xi_sum, yi_sum, Nd]])

    def standardize(z_data) -> tuple[ComplexArray, complex]:
        mean = np.mean(z_data)
        return (z_data - mean, mean)

    def destandardize(z_data, mean) -> ComplexArray:
        return z_data + mean
    z_data, mean = standardize(z_data)
    M = calc_moments(z_data)
    a0 = ((M[2][0] * M[3][2] - M[2][2] * M[3][0]) * M[1][1] - M[1][2] * M[2][0] * M[3][1] - M[1][0] * M[2][1] * M[3][2] + M[1][0] * M[2][2] * M[3][1] + M[1][2] * M[2][1] * M[3][0]) * M[0][3] + (M[0][2] * M[2][3] * M[3][0] - M[0][2] * M[2][0] * M[3][3] + M[0][0] * M[2][2] * M[3][3] - M[0][0] * M[2][3] * M[3][2]) * M[1][1] + (M[0][1] * M[1][3] * M[3][0] - M[0][1] * M[1][0] * M[3][3] - M[0][0] * M[1][3] * M[3][1]) * M[2][2] + (-M[0][1] * M[1][2] * M[2][3] - M[0][2] * M[1][3] * M[2][1]) * M[3][0] + ((M[2][3] * M[3][1] - M[2][1] * M[3][3]) * M[1][2] + M[2][1] * M[3][2] * M[1][3]) * M[0][0] + (M[1][0] * M[2][3] * M[3][2] + M[2][0] * (M[1][2] * M[3][3] - M[1][3] * M[3][2])) * M[0][1] + ((M[2][1] * M[3][3] - M[2][3] * M[3][1]) * M[1][0] + M[1][3] * M[2][0] * M[3][1]) * M[0][2]
    a1 = ((M[3][0] - 2.0 * M[2][2]) * M[1][1] - M[1][0] * M[3][1] + M[2][2] * M[3][0] + 2.0 * M[1][2] * M[2][1] - M[2][0] * M[3][2]) * M[0][3] + (2.0 * M[2][0] * M[3][2] - M[0][0] * M[3][3] - 2.0 * M[2][2] * M[3][0] + 2.0 * M[0][2] * M[2][3]) * M[1][1] + (-M[0][0] * M[3][3] + 2.0 * M[0][1] * M[1][3] + 2.0 * M[1][0] * M[3][1]) * M[2][2] + (-M[0][1] * M[1][3] + 2.0 * M[1][2] * M[2][1] - M[0][2] * M[2][3]) * M[3][0] + (M[1][3] * M[3][1] + M[2][3] * M[3][2]) * M[0][0] + (M[1][0] * M[3][3] - 2.0 * M[1][2] * M[2][3]) * M[0][1] + (M[2][0] * M[3][3] - 2.0 * M[1][3] * M[2][1]) * M[0][2] - 2.0 * M[1][2] * M[2][0] * M[3][1] - 2.0 * M[1][0] * M[2][1] * M[3][2]
    a2 = (2.0 * M[1][1] - M[3][0] + 2.0 * M[2][2]) * M[0][3] + (2.0 * M[3][0] - 4.0 * M[2][2]) * M[1][1] - 2.0 * M[2][0] * M[3][2] + 2.0 * M[2][2] * M[3][0] + M[0][0] * M[3][3] + 4.0 * M[1][2] * M[2][1] - 2.0 * M[0][1] * M[1][3] - 2.0 * M[1][0] * M[3][1] - 2.0 * M[0][2] * M[2][3]
    a3 = -2.0 * M[3][0] + 4.0 * M[1][1] + 4.0 * M[2][2] - 2.0 * M[0][3]
    a4 = -4.0

    def func(x):
        return a0 + a1 * x + a2 * x * x + a3 * x * x * x + a4 * x * x * x * x

    def d_func(x):
        return a1 + 2 * a2 * x + 3 * a3 * x * x + 4 * a4 * x * x * x
    x0 = spopt.fsolve(func, 0.0, fprime=d_func)

    def solve_eq_sys(val, M):
        M[3][0] = M[3][0] + 2 * val
        M[0][3] = M[0][3] + 2 * val
        M[1][1] = M[1][1] - val
        M[2][2] = M[2][2] - val
        return np.linalg.svd(M)
    _, s, Vt = solve_eq_sys(x0[0], M)
    A_vec = Vt[np.argmin(s), :]
    xc = -A_vec[1] / (2.0 * A_vec[0])
    yc = -A_vec[2] / (2.0 * A_vec[0])
    r0 = 1.0 / (2.0 * np.absolute(A_vec[0])) * np.sqrt(A_vec[1] * A_vec[1] + A_vec[2] * A_vec[2] - 4.0 * A_vec[0] * A_vec[3])
    if refine_results:
        print('agebraic r0: ' + str(r0))
        xc, yc, r0 = _fit_circle_iter(z_data, xc, yc, r0)
        r0 = _fit_circle_iter_radialweight(z_data, xc, yc, r0)
        print('iterative r0: ' + str(r0))
    z_data = destandardize(z_data, mean)
    xc += mean.real
    yc += mean.imag
    return (xc, yc, r0)

def _fit_circle_iter(z_data: ComplexArray, xc: float, yc: float, rc: float) -> tuple[float, float, float]:
    """
        this is the radial weighting procedure
        it improves your fitting value for the radius = Ql/Qc
        use this to improve your fit in presence of heavy noise
        after having used the standard algebraic fit_circle() function
        the weight here is: W=1/sqrt((xc-xi)^2+(yc-yi)^2)
        this works, because the center of the circle is usually much less
        corrupted by noise than the radius
        """
    xdat = z_data.real
    ydat = z_data.imag

    def fitfunc(x, y, xc, yc):
        return np.sqrt((x - xc) ** 2 + (y - yc) ** 2)

    def residuals(p, x, y):
        xc, yc, r = p
        temp = r - fitfunc(x, y, xc, yc)
        return temp
    p0 = [xc, yc, rc]
    p_final = spopt.leastsq(residuals, p0, args=(xdat, ydat), ftol=1e-12, xtol=1e-12)
    xc, yc, rc = p_final[0]
    return (xc, yc, rc)

def _fit_circle_iter_radialweight(z_data: ComplexArray, xc: float, yc: float, rc: float) -> float:
    """
        this is the radial weighting procedure
        it improves your fitting value for the radius = Ql/Qc
        use this to improve your fit in presence of heavy noise
        after having used the standard algebraic fir_circle() function
        the weight here is: W=1/sqrt((xc-xi)^2+(yc-yi)^2)
        this works, because the center of the circle is usually much less
        corrupted by noise than the radius
        """
    xdat = z_data.real
    ydat = z_data.imag

    def fitfunc(x, y):
        return np.sqrt((x - xc) ** 2 + (y - yc) ** 2)

    def weight(x, y):
        try:
            res = 1.0 / np.sqrt((xc - x) ** 2 + (yc - y) ** 2)
        except Exception:
            res = 1.0
        return res

    def residuals(p, x, y):
        r = p[0]
        temp = (r - fitfunc(x, y)) * weight(x, y)
        return temp
    p0 = [rc]
    p_final = spopt.leastsq(residuals, p0, args=(xdat, ydat), ftol=1e-12, xtol=1e-12)
    return p_final[0][0]

def _fit_delay(f_data: FloatArray, z_data: ComplexArray, delay: float=0.0, maxiter: int=0) -> float:
    delay_scale = max(abs(delay), 1.0 / np.mean(f_data))

    def residuals(p, x, y):
        phasedelay = p[0] * delay_scale
        z_data_temp = y * np.exp(1j * (2.0 * np.pi * phasedelay * x))
        xc, yc, r0 = _fit_circle(z_data_temp)
        err = np.sqrt((z_data_temp.real - xc) ** 2 + (z_data_temp.imag - yc) ** 2) - r0
        return err
    p_final = spopt.leastsq(residuals, [delay / delay_scale], args=(f_data, z_data), maxfev=maxiter, ftol=1e-12, xtol=1e-12)
    return p_final[0][0] * delay_scale

def _fit_skewed_lorentzian(f_data: FloatArray, z_data: ComplexArray) -> npt.NDArray[np.float64]:
    amplitude = np.absolute(z_data)
    amplitude_sqr = amplitude ** 2
    amp_sqr = np.array(amplitude_sqr)
    imin = np.argmin(amp_sqr)
    fra = f_data[imin]
    A1a = 0.5 * (amp_sqr[0] + amp_sqr[-1])
    A3a = amp_sqr[imin] - A1a
    mid = A1a + 0.5 * A3a
    i_left = imin
    while i_left > 0 and amp_sqr[i_left] < mid:
        i_left -= 1
    i_right = imin
    while i_right < len(amp_sqr) - 1 and amp_sqr[i_right] < mid:
        i_right += 1
    if i_right > i_left:
        fwhm = f_data[i_right] - f_data[i_left]
        Ql_est = fra / max(fwhm, f_data[1] - f_data[0])
    else:
        Ql_est = 1000.0
    fr_scale = np.mean(f_data)
    f_norm = np.array(f_data) / fr_scale
    fra_n = fra / fr_scale

    def residuals(p, x_n, y):
        A2_n, A4_n, Ql = p
        err = y - (A1a + A2_n * fr_scale * (x_n - fra_n) + (A3a + A4_n * fr_scale * (x_n - fra_n)) / (1.0 + 4.0 * Ql ** 2 * ((x_n - fra_n) / fra_n) ** 2))
        return err
    p0 = [0.0, 0.0, Ql_est]
    p_final = spopt.leastsq(residuals, p0, args=(f_norm, amp_sqr), ftol=1e-12, xtol=1e-12)
    A2a_n, A4a_n, Qla = p_final[0]
    A2a = A2a_n
    A4a = A4a_n

    def fitfunc(x_n, A1, A2_n, A3, A4_n, fr_n, Ql):
        return A1 + A2_n * fr_scale * (x_n - fr_n) + (A3 + A4_n * fr_scale * (x_n - fr_n)) / (1.0 + 4.0 * Ql ** 2 * ((x_n - fr_n) / fr_n) ** 2)
    p0 = [A1a, A2a, A3a, A4a, fra_n, Qla]
    try:
        popt, pcov = spopt.curve_fit(fitfunc, f_norm, amp_sqr, p0=p0, bounds=([-np.inf, -np.inf, -np.inf, -np.inf, f_norm[0], 0], [np.inf, np.inf, np.inf, np.inf, f_norm[-1], np.inf]))
        if pcov is not None:
            df_error = np.sqrt(pcov[4][4]) * fr_scale
            dQl_error = np.sqrt(pcov[5][5])
        else:
            df_error = np.inf
            dQl_error = np.inf
        popt[4] *= fr_scale
    except Exception:
        popt = np.array([A1a, A2a, A3a, A4a, fra, Qla])
        df_error = np.inf
        dQl_error = np.inf
    return popt

def _get_cov_fast_notch(xdata: Any, ydata: Any, fitparams: Any) -> tuple[float, Any]:

    def dS21_dQl(p, f):
        fr, absQc, Ql, phi0 = p
        return -(np.exp(1j * phi0) * fr ** 2) / (absQc * (fr + 2j * Ql * f - 2j * Ql * fr) ** 2)

    def dS21_dQc(p, f):
        fr, absQc, Ql, phi0 = p
        return np.exp(1j * phi0) * Ql * fr / (2j * (f - fr) * absQc ** 2 * Ql + absQc ** 2 * fr)

    def dS21_dphi0(p, f):
        fr, absQc, Ql, phi0 = p
        return -(1j * Ql * fr * np.exp(1j * phi0)) / (2j * (f - fr) * absQc * Ql + absQc * fr)

    def dS21_dfr(p, f):
        fr, absQc, Ql, phi0 = p
        return -(2j * Ql ** 2 * f * np.exp(1j * phi0)) / (absQc * (fr + 2j * Ql * f - 2j * Ql * fr) ** 2)
    u = _residuals_notch_ideal_complex(fitparams, xdata, ydata)
    chi = np.absolute(u)
    u = u / chi
    aa = dS21_dfr(fitparams, xdata)
    bb = dS21_dQc(fitparams, xdata)
    cc = dS21_dQl(fitparams, xdata)
    dd = dS21_dphi0(fitparams, xdata)
    Jt = np.array([aa.real * u.real + aa.imag * u.imag, bb.real * u.real + bb.imag * u.imag, cc.real * u.real + cc.imag * u.imag, dd.real * u.real + dd.imag * u.imag])
    A = np.dot(Jt, np.transpose(Jt))
    chisqr = 1.0 / float(len(xdata) - len(fitparams)) * (chi ** 2).sum()
    try:
        cov = np.linalg.pinv(A) * chisqr
    except Exception:
        cov = None
    return (chisqr, cov)

def _guess_delay(f_data: FloatArray, z_data: ComplexArray) -> float:
    phase2 = np.unwrap(np.angle(z_data))
    gradient = stats.linregress(f_data, phase2)[0]
    gradient = float(gradient)
    return gradient * -1.0 / (np.pi * 2.0)

def _periodic_boundary(x: float, bound: float) -> float:
    return np.fmod(x, bound) - np.trunc(x / bound) * bound

def _phase_fit(f_data: FloatArray, z_data: ComplexArray, theta0: float, Ql: float, fr: float) -> npt.NDArray[np.float64]:
    phase = np.angle(z_data)
    fr_scale = np.mean(f_data)
    Ql_scale = max(abs(Ql), 1.0)
    x_norm = np.array(f_data) / fr_scale

    def _phase_model(x_n, theta0, Ql_n, fr_n):
        return theta0 + 2.0 * np.arctan(2.0 * Ql_n * Ql_scale * (1.0 - x_n / fr_n))

    def residuals_1(p, x_n, y, Ql_n):
        theta0, fr_n = p
        return _dist(y - _phase_model(x_n, theta0, Ql_n, fr_n))

    def residuals_2(p, x_n, y, theta0):
        Ql_n, fr_n = p
        return _dist(y - _phase_model(x_n, theta0, Ql_n, fr_n))

    def residuals_3(p, x_n, y, theta0, Ql_n):
        fr_n = p
        return _dist(y - _phase_model(x_n, theta0, Ql_n, fr_n))

    def residuals_4(p, x_n, y, theta0, fr_n):
        Ql_n = p
        return _dist(y - _phase_model(x_n, theta0, Ql_n, fr_n))

    def residuals_5(p, x_n, y):
        theta0, Ql_n, fr_n = p
        return _dist(y - _phase_model(x_n, theta0, Ql_n, fr_n))
    fr_n = fr / fr_scale
    Ql_n = Ql / Ql_scale
    p0 = [theta0, fr_n]
    p_final = spopt.leastsq(lambda a, b, c: residuals_1(a, b, c, Ql_n), p0, args=(x_norm, phase), ftol=1e-12, xtol=1e-12)
    theta0, fr_n = p_final[0]
    p0 = [Ql_n, fr_n]
    p_final = spopt.leastsq(lambda a, b, c: residuals_2(a, b, c, theta0), p0, args=(x_norm, phase), ftol=1e-12, xtol=1e-12)
    Ql_n, fr_n = p_final[0]
    p0 = fr_n
    p_final = spopt.leastsq(lambda a, b, c: residuals_3(a, b, c, theta0, Ql_n), p0, args=(x_norm, phase), ftol=1e-12, xtol=1e-12)
    fr_n = p_final[0][0]
    p0 = Ql_n
    p_final = spopt.leastsq(lambda a, b, c: residuals_4(a, b, c, theta0, fr_n), p0, args=(x_norm, phase), ftol=1e-12, xtol=1e-12)
    Ql_n = p_final[0][0]
    p0 = np.array([theta0, Ql_n, fr_n], dtype='float64')
    p_final = spopt.leastsq(residuals_5, p0, args=(x_norm, phase), ftol=1e-12, xtol=1e-12)
    theta0_f, Ql_n_f, fr_n_f = p_final[0]
    return np.array([theta0_f, Ql_n_f * Ql_scale, fr_n_f * fr_scale])

def _residuals_notch_ideal(p: Any, x: Any, y: Any) -> Any:
    fr, absQc, Ql, phi0 = p
    err = np.absolute(y - (1.0 - Ql / float(absQc) * np.exp(1j * phi0) / (1 + 2j * Ql * (x - fr) / float(fr))))
    return err

def _residuals_notch_ideal_complex(p: Any, x: Any, y: Any) -> Any:
    fr, absQc, Ql, phi0 = p
    err = y - (1.0 - Ql / float(absQc) * np.exp(1j * phi0) / (1 + 2j * Ql * (x - fr) / float(fr)))
    return err

def circlefit(f_data: FloatArray, z_data: ComplexArray, fr: float | None=None, Ql: float | None=None, refine_results: bool=False, calc_errors: bool=True) -> dict[str, float]:
    """
        performs a circle fit on a frequency vs. complex resonator scattering data set
        Data has to be normalized!!
        INPUT:
        f_data,z_data: input data (frequency, complex S21 data)
        OUTPUT:
        outpus a dictionary {key:value} consisting of the fit values, errors and status information about the fit
        values: {"phi0":phi0, "Ql":Ql, "absolute(Qc)":absQc, "Qi": Qi, "electronic_delay":delay, "complexQc":complQc, "resonance_freq":fr, "prefactor_a":a, "prefactor_alpha":alpha}
        errors: {"phi0_err":phi0_err, "Ql_err":Ql_err, "absolute(Qc)_err":absQc_err, "Qi_err": Qi_err, "electronic_delay_err":delay_err, "resonance_freq_err":fr_err, "prefactor_a_err":a_err, "prefactor_alpha_err":alpha_err}
        for details, see:
                [1] (not diameter corrected) Jiansong Gao, "The Physics of Superconducting Microwave Resonators" (PhD Thesis), Appendix E, California Institute of Technology, (2008)
                [2] (diameter corrected) M. S. Khalil, et. al., J. Appl. Phys. 111, 054510 (2012)
                [3] (fitting techniques) N. CHERNOV AND C. LESORT, "Least Squares Fitting of Circles", Journal of Mathematical Imaging and Vision 23, 239, (2005)
                [4] (further fitting techniques) P. J. Petersan, S. M. Anlage, J. Appl. Phys, 84, 3392 (1998)
        the program fits the circle with the algebraic technique described in [3], the rest of the fitting is done with the scipy.optimize least square fitting toolbox
        also, check out [5] S. Probst et al. "Efficient and reliable analysis of noisy complex scatterung resonator data for superconducting quantum circuits" (in preparation)
        """
    if fr is None:
        fr = float(f_data[np.argmin(np.absolute(z_data))])
    if Ql is None:
        Ql = 1000000.0
    xc, yc, r0 = _fit_circle(z_data, refine_results=refine_results)
    phi0 = -np.arcsin(yc / r0)
    theta0 = _periodic_boundary(phi0 + np.pi, np.pi)
    z_data_corr = _center(z_data, complex(xc, yc))
    theta0, Ql, fr = _phase_fit(f_data, z_data_corr, theta0, Ql, fr)
    assert Ql is not None
    assert fr is not None
    absQc: float = Ql / (2.0 * r0)
    complQc = absQc * np.exp(1j * (-1.0 * phi0))
    Qc: float = float(1.0 / (1.0 / complQc).real)
    Qi_dia_corr: float = 1.0 / (1.0 / Ql - 1.0 / Qc)
    Qi_no_corr: float = 1.0 / (1.0 / Ql - 1.0 / absQc)
    results: dict[str, float] = {'Qi_dia_corr': Qi_dia_corr, 'Qi_no_corr': Qi_no_corr, 'absQc': absQc, 'Qc_dia_corr': Qc, 'Ql': Ql, 'fr': fr, 'theta0': theta0, 'phi0': phi0}
    p: list[float] = [fr, absQc, Ql, phi0]
    if calc_errors is True:
        chi_square, cov = _get_cov_fast_notch(f_data, z_data, p)
        if cov is not None:
            errors = np.sqrt(np.diagonal(cov))
            fr_err, absQc_err, Ql_err, phi0_err = errors
            dQl = 1.0 / ((1.0 / Ql - 1.0 / absQc) ** 2 * Ql ** 2)
            dabsQc = -1.0 / ((1.0 / Ql - 1.0 / absQc) ** 2 * absQc ** 2)
            Qi_no_corr_err = np.sqrt(dQl ** 2 * cov[2][2] + dabsQc ** 2 * cov[1][1] + 2 * dQl * dabsQc * cov[2][1])
            dQl = 1 / ((1 / Ql - np.cos(phi0) / absQc) ** 2 * Ql ** 2)
            dabsQc = -np.cos(phi0) / ((1 / Ql - np.cos(phi0) / absQc) ** 2 * absQc ** 2)
            dphi0 = -np.sin(phi0) / ((1 / Ql - np.cos(phi0) / absQc) ** 2 * absQc)
            err1 = dQl ** 2 * cov[2][2] + dabsQc ** 2 * cov[1][1] + dphi0 ** 2 * cov[3][3]
            err2 = dQl * dabsQc * cov[2][1] + dQl * dphi0 * cov[2][3] + dabsQc * dphi0 * cov[1][3]
            Qi_dia_corr_err = np.sqrt(err1 + 2 * err2)
            errors_dict: dict[str, float] = {'phi0_err': phi0_err, 'Ql_err': Ql_err, 'absQc_err': absQc_err, 'fr_err': fr_err, 'chi_square': chi_square, 'Qi_no_corr_err': Qi_no_corr_err, 'Qi_dia_corr_err': Qi_dia_corr_err}
            results.update(errors_dict)
        else:
            print('WARNING: Error calculation failed!')
    else:
        chi_square = 1.0 / float(len(f_data) - len(p)) * (_residuals_notch_ideal(p, f_data, z_data) ** 2).sum()
        errors_dict = {'chi_square': chi_square}
        results.update(errors_dict)
    return results

def do_calibration(f_data: FloatArray, z_data: ComplexArray, ignoreslope: bool=True, guessdelay: bool=True, fixed_delay: float | None=None, Ql_guess: float | None=None, fr_guess: float | None=None) -> tuple[float, float, float, float, float, float, float]:
    """
        performs an automated calibration and tries to determine the prefactors a, alpha, delay
        fr, Ql, and a possible slope are extra information, which can be used as start parameters for subsequent fits
        see also "do_normalization"
        the calibration procedure works for transmission line resonators as well
        """
    delay, params = get_delay(f_data, z_data, ignoreslope=ignoreslope, guess=guessdelay, delay=fixed_delay)
    z_data = (z_data - params[1] * (f_data - params[4])) * np.exp(2.0 * 1j * np.pi * delay * f_data)
    xc, yc, r0 = _fit_circle(z_data)
    zc = complex(xc, yc)
    if Ql_guess is None:
        Ql_guess = float(np.absolute(params[5]))
    if fr_guess is None:
        fr_guess = float(params[4])
    fitparams = _phase_fit(f_data, _center(z_data, zc), 0.0, Ql_guess, fr_guess)
    theta, Ql, fr = fitparams
    beta = _periodic_boundary(theta + np.pi, np.pi)
    offrespoint = complex(xc + r0 * np.cos(beta), yc + r0 * np.sin(beta))
    alpha = np.angle(offrespoint)
    a = np.absolute(offrespoint)
    return (delay, a, alpha, fr, Ql, params[1], params[4])

def do_normalization(f_data: FloatArray, z_data: ComplexArray, delay: float, amp_norm: float, alpha: float, A2: float, frcal: float) -> ComplexArray:
    """
        removes the prefactors a, alpha, delay and returns the calibrated data, see also "do_calibration"
        works also for transmission line resonators
        """
    return (z_data - A2 * (f_data - frcal)) / amp_norm * np.exp(1j * (-alpha + 2.0 * np.pi * delay * f_data))

def get_delay(f_data: FloatArray, z_data: ComplexArray, delay: float | None=None, ignoreslope: bool=True, guess: bool=True) -> tuple[float, list[float]]:
    """
        retrieves the cable delay assuming the ideal resonance has a circular shape
        modifies the cable delay until the shape Im(S21) vs Re(S21) is circular
        see "do_calibration"
        """
    maxval = np.max(np.absolute(z_data))
    z_data = z_data / maxval
    A1, A2, A3, A4, fr, Ql = _fit_skewed_lorentzian(f_data, z_data)
    if ignoreslope is True:
        A2 = 0.0
    else:
        A2 = 0.0
        print('WARNING: The ignoreslope option is ignored! Corrections to the baseline should be done manually prior to fitting.')
        print('see also: resonator_tools.calibration.fit_baseline_amp() etc. for help on fitting the baseline.')
        print('There is also an example ipython notebook for using this function.')
        print('However, make sure to understand the impact of the baseline (parasitic coupled resonances etc.) on your system.')
    if delay is None:
        if guess is True:
            delay = _guess_delay(f_data, z_data)
        else:
            delay = 0.0
        delay = _fit_delay(f_data, z_data, delay, maxiter=0)
    params: list[float] = [float(A1), float(A2), float(A3), float(A4), float(fr), float(Ql)]
    return (delay, params)

def reconstruct(frequency, scattering):
    """Fit full observations; return physical parameters, response and stages."""
    cal = do_calibration(frequency, scattering)
    delay, amplitude, alpha, fr, ql, slope, frcal = cal
    normalized = do_normalization(frequency, scattering, *[delay, amplitude, alpha, slope, frcal])
    fit = circlefit(frequency, normalized, fr, ql, refine_results=False, calc_errors=True)
    response = slope*(frequency-frcal) + _S21_notch(frequency, fr=fit['fr'], Ql=fit['Ql'], Qc=fit['absQc'], phi=fit['phi0'], a=amplitude, alpha=alpha, delay=delay)
    return fit, response, normalized, np.asarray(cal)
