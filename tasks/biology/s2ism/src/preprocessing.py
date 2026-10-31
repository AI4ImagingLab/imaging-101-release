"""
Preprocessing utilities for s2ISM.

Handles shift vector estimation, magnification finding, and detector misalignment
correction — all steps that transform raw ISM data or calibrate the imaging system
before reconstruction.
"""

import numpy as np
from scipy.optimize import minimize, least_squares
from scipy.signal import convolve
from scipy.special import jv

from brighteyes_ism.analysis.APR_lib import ShiftVectors
from brighteyes_ism.simulation.detector import det_coords, airy_to_hex


# ==========================================
# Shift Vectors Minimizer
# ==========================================

def shift_matrix(geometry: str = 'rect') -> np.ndarray:
    coordinates = -det_coords(3, geometry)
    coordinates = np.swapaxes(coordinates, 0, 1)
    return coordinates


def rotation_matrix(theta: float) -> np.ndarray:
    rot_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                           [np.sin(theta), np.cos(theta)]])
    rot = np.squeeze(rot_matrix)
    return rot


def mirror_matrix(alpha):
    mirror = np.array([[1, 0], [0, alpha]])
    return mirror


def crop_shift(shift_exp: np.ndarray, geometry: str = 'rect') -> np.ndarray:
    n_crop = 3
    nch = shift_exp.shape[0]
    n = int(np.ceil(np.sqrt(nch)))

    if geometry == 'rect':
        shift_exp = shift_exp.reshape(n, n, -1)
        shift_cropped = np.zeros((n_crop, n_crop, 2))

        for i, l in enumerate(np.arange(-1, 2)):
            for j, k in enumerate(np.arange(-1, 2)):
                shift_cropped[i, j, :] = shift_exp[n // 2 + l, n // 2 + k, :]

        shift_cropped = shift_cropped.reshape(n_crop**2, 2)

    elif geometry == 'hex':
        c = nch // 2
        idx_crop = np.sort(np.asarray(
            [c, c - 1, c + 1, c - n, c - n + 1, c + n, c + n - 1], dtype=int))
        shift_cropped = shift_exp[idx_crop]

    else:
        raise Exception("Detector geometry not valid. Select 'rect' or 'hex'.")

    return shift_cropped


def transform_shift_vectors(param, shift):
    a = param[0]
    r = rotation_matrix(param[1])
    m = mirror_matrix(param[2])
    transform_matrix = a * r @ m
    shift_transf = np.einsum('ij,kj -> ki', transform_matrix, shift)
    return shift_transf


def loss_shifts(x0, shift_exp: np.ndarray, shift_theor: np.ndarray, mirror: float) -> float:
    parameters = [*x0, mirror]
    shift_fin = transform_shift_vectors(parameters, shift_theor)
    loss_func = np.linalg.norm(shift_exp - shift_fin) ** 2
    return loss_func


def svm_loss_minimizer(shift_m, shift_t, alpha_0, theta_0, tol, opt, mirror):
    results = minimize(loss_shifts, x0=(alpha_0, theta_0), args=(shift_m, shift_t, mirror),
                       options=opt, tol=tol, method='Nelder-Mead')
    if results.success:
        alpha = results.x[0]
        theta = results.x[1]

        if alpha < 0:
            alpha = abs(alpha)
            theta += np.pi

        return alpha, theta, mirror

    else:
        print('Minimization did not succeed.')
        print(results.message)

        alpha = results.x[0]
        theta = results.x[1]

        if alpha < 0:
            alpha = abs(alpha)
            theta += np.pi

        return alpha, theta, mirror


def find_parameters(shift_exp: np.ndarray, geometry: str = 'rect', name: str = None,
                    alpha_0: float = 2, theta_0: float = 0.5):
    if name == 'airyscan':
        shift_vectors = airy_to_hex(shift_exp)
    else:
        shift_vectors = shift_exp

    shift_crop = crop_shift(shift_vectors, geometry)
    shift_theor = shift_matrix(geometry)
    tol = 1e-6
    opt = {'maxiter': 10000}
    params = svm_loss_minimizer(shift_crop, shift_theor, alpha_0, theta_0, tol, opt, mirror=1)
    params_mirror = svm_loss_minimizer(shift_crop, shift_theor, alpha_0, theta_0, tol, opt, mirror=-1)

    Loss_0 = loss_shifts(params, shift_crop, shift_theor, 1)
    Loss_1 = loss_shifts(params_mirror, shift_crop, shift_theor, -1)

    if Loss_0 < Loss_1:
        alpha = params[0]
        theta = params[1]
        mirror = 1
    else:
        alpha = params_mirror[0]
        theta = params_mirror[1]
        mirror = -1

    return alpha, theta, mirror


def calc_shift_vectors(parameters, geometry: str = 'rect'):
    shift_theor = shift_matrix(geometry)
    shift_array = transform_shift_vectors(parameters, shift_theor)
    return shift_array


# ==========================================
# Magnification Finder
# ==========================================

def scalar_psf(r, wl, na):
    k = 2 * np.pi / wl
    x = k * r * na
    psf = np.ones_like(x) * 0.5
    np.divide(jv(1, x), x, out=psf, where=(x != 0))
    psf = np.abs(psf) ** 2
    return psf


def rect(r, d):
    r = np.where(abs(r) <= d / 2, 1, 0)
    return r / d


def scalar_psf_det(r, wl, na, pxdim, pxpitch, m):
    psf = scalar_psf(r, wl, na)
    pinhole = rect(r - pxpitch / m, pxdim / m)
    psf_det = convolve(psf, pinhole, mode='same')
    return psf_det


def shift_value(m, wl_ex, wl_em, pxpitch, pxdim, na):
    airy_unit = 1.22 * wl_em / na
    pxsize = 0.1
    range_x = int(airy_unit / pxsize)
    ref = np.arange(-range_x, range_x + 1) * pxsize

    psf_det = scalar_psf_det(ref, wl_em, na, pxdim, pxpitch, m)
    psf_ex = scalar_psf(ref, wl_ex, na)
    psf_conf = psf_det * psf_ex

    shift = ref[np.argmax(psf_conf)]
    return shift


def loss_shift(x, shift_exp, wl_ex, wl_em, pxpitch, pxdim, na):
    shift_t = shift_value(x, wl_ex, wl_em, pxpitch, pxdim, na)
    loss_func = np.linalg.norm(shift_t - shift_exp)**2
    return loss_func


def mag_loss_minimizer(shift_t, wl_ex, wl_em, pxpitch, pxdim, na, m_0, tol, opt):
    result = minimize(loss_shift, x0=m_0, args=(shift_t, wl_ex, wl_em, pxpitch, pxdim, na),
                      options=opt, tol=tol, method='Nelder-Mead')
    if not result.success:
        print('Minimization did not succeed.')
        print(result.message)
    return result.x[0]


def find_mag(shift, wl_ex, wl_em, pxpitch, pxdim, na):
    opt = {'maxiter': 10000}
    tol = 1e-6
    m_0 = 500
    m = mag_loss_minimizer(shift, wl_ex, wl_em, pxpitch, pxdim, na, m_0, tol, opt)
    return m


# ==========================================
# Misalignment Detection
# ==========================================

def gaussian_2d(params, x, y):
    a, x0, y0, sigma, b = params
    return a * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2)) + b


def residuals(params, x, y, data):
    model = gaussian_2d(params, x, y)
    return (model - data).ravel()


def gaussian_fit(image):
    image = image / image.max()
    h, w = image.shape
    y, x = np.mgrid[0:h, 0:w]
    a0 = image.max() - image.min()
    x0_0 = (w - 1) / 2
    y0_0 = (h - 1) / 2
    sigma0 = min(h, w) / 4
    b0 = image.min()
    p0 = [a0, x0_0, y0_0, sigma0, b0]
    bounds = (
        [0,           0,  0,  0.5, 0],
        [image.max(), w,  h,  w,   image.min()]
    )
    result = least_squares(
        residuals, p0, args=(x, y, image),
        bounds=bounds, method='trf', loss='cauchy', f_scale=0.1
    )
    gfit = gaussian_2d(result.x, x, y)
    if result.success:
        _, x0, y0, _, _ = result.x
        return x0 - x0_0, y0 - y0_0, result.x, gfit
    else:
        return None


def find_misalignment(dset, pxpitch, mag, na, wl):
    nch = int(np.sqrt(dset.shape[-1]))
    axis_to_sum = tuple(np.arange(dset.ndim - 1))
    fingerprint = dset.sum(axis_to_sum).reshape(nch, nch)
    gauss = gaussian_fit(fingerprint)

    if gauss is None:
        print('\n Warning: Fitting not successful. Using no tip and no tilt.\n')
        tip, tilt = np.zeros(2)
    else:
        scale = 2 * np.pi * na / wl
        pxsize = pxpitch / mag

        coords = -np.asarray([gauss[1], gauss[0]])

        shift, _ = ShiftVectors(dset[5:-5, 5:-5], 50, dset.shape[-1] // 2, filter_sigma=1)
        _, rotation, mirroring = find_parameters(shift)

        if mirroring == -1:
            coords[1] *= -1

        rm = rotation_matrix(rotation)
        rot_coords = np.einsum('ij, j -> i', rm, coords)
        tip, tilt = rot_coords * pxsize * scale

    return tip, tilt


def realign_psf(psf):
    nz, ny, nx, nch = psf.shape
    patch = psf[0].sum(-1)
    yc, xc = ny // 2, nx // 2
    peak_index = np.argmax(patch)
    y_peak, x_peak = np.unravel_index(peak_index, patch.shape)
    y_shift = yc - y_peak
    x_shift = xc - x_peak
    aligned_psf = np.empty_like(psf)
    for z in range(nz):
        aligned_psf[z] = np.roll(psf[z], shift=y_shift, axis=0)
        aligned_psf[z] = np.roll(aligned_psf[z], shift=x_shift, axis=1)
    return aligned_psf
