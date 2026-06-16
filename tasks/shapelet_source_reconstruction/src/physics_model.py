"""
Core physics: shapelet basis functions, gravitational lens models,
ray-tracing, PSF convolution, and image simulation pipeline.

All implemented from first principles using numpy/scipy only.
"""

import numpy as np
import scipy.ndimage
import math


# ============================================================
# Grid and Array Utilities
# ============================================================

def make_grid(numPix, deltapix):
    """Create centered 1D coordinate arrays for a square pixel grid.

    Returns flattened (x, y) where x varies fast (column index) and y
    repeats (row index), matching the convention used by array2image.

    :param numPix: number of pixels per axis
    :param deltapix: pixel size in angular units
    :return: (x_grid, y_grid) each of length numPix^2
    """
    N = int(numPix)
    a = np.arange(N)
    matrix = np.dstack(np.meshgrid(a, a)).reshape(-1, 2)
    x_grid = (matrix[:, 0] - (N - 1) / 2.0) * deltapix
    y_grid = (matrix[:, 1] - (N - 1) / 2.0) * deltapix
    return x_grid, y_grid


def make_grid_2d(numPix, deltaPix):
    """Create 2D coordinate arrays of shape (numPix, numPix).

    :return: (x_coords, y_coords) 2D arrays
    """
    N = int(numPix)
    a = np.arange(N)
    X, Y = np.meshgrid(a, a)
    return (X - (N - 1) / 2.0) * deltaPix, (Y - (N - 1) / 2.0) * deltaPix


def image2array(image):
    """Flatten 2D image to 1D array (row-major)."""
    return image.reshape(image.shape[0] * image.shape[1])


def array2image(array):
    """Reshape 1D array to 2D square image."""
    n = int(np.sqrt(len(array)))
    return array.reshape(n, n)


def re_size(image, factor):
    """Downsample image by averaging factor x factor blocks.

    :param image: 2D array of shape (nx, ny)
    :param factor: integer downsampling factor (must divide nx and ny)
    :return: downsampled 2D array of shape (nx/factor, ny/factor)
    """
    if factor == 1:
        return image.copy()
    f = int(factor)
    nx, ny = image.shape
    return image.reshape(nx // f, f, ny // f, f).mean(3).mean(1)


# ============================================================
# Shapelet Basis Functions (Cartesian Hermite)
# ============================================================

def pre_calc_shapelets(x, y, beta, n_max, center_x=0.0, center_y=0.0):
    """Pre-compute phi_n arrays for all orders up to n_max.

    phi_n(u) = [2^n sqrt(pi) n!]^{-1/2} H_n(u) exp(-u^2/2)
    where H_n is the physicist's Hermite polynomial.

    :param x: 1D array of x-coordinates
    :param y: 1D array of y-coordinates
    :param beta: shapelet scale parameter
    :param n_max: maximum polynomial order
    :param center_x: center x-coordinate
    :param center_y: center y-coordinate
    :return: (H_x, H_y) arrays of shape (n_max+1, len(x))
    """
    x_ = (x - center_x) / beta
    y_ = (y - center_y) / beta
    m = len(x)
    H_x = np.zeros((n_max + 1, m))
    H_y = np.zeros((n_max + 1, m))
    exp_x = np.exp(-x_**2 / 2.0)
    exp_y = np.exp(-y_**2 / 2.0)
    for n in range(n_max + 1):
        prefactor = 1.0 / np.sqrt(2.0**n * np.sqrt(np.pi) * math.factorial(n))
        n_array = np.zeros(n + 1)
        n_array[n] = 1
        x_cut = np.sqrt(n + 1) * 5.0
        mask_x = np.abs(x_) < x_cut
        mask_y = np.abs(y_) < x_cut
        hx = np.zeros(m)
        hy = np.zeros(m)
        if np.any(mask_x):
            hx[mask_x] = np.polynomial.hermite.hermval(x_[mask_x], n_array)
        if np.any(mask_y):
            hy[mask_y] = np.polynomial.hermite.hermval(y_[mask_y], n_array)
        H_x[n] = hx * prefactor * exp_x
        H_y[n] = hy * prefactor * exp_y
    return H_x, H_y


def iterate_n1_n2(n_max):
    """Generate (index, n1, n2) tuples in lenstronomy ordering.

    Order: (0,0), (1,0), (0,1), (2,0), (1,1), (0,2), ...
    Constraint: n1 + n2 <= n_max.
    """
    num_param = (n_max + 1) * (n_max + 2) // 2
    n1, n2 = 0, 0
    for i in range(num_param):
        yield i, n1, n2
        if n1 == 0:
            n1 = n2 + 1
            n2 = 0
        else:
            n1 -= 1
            n2 += 1


def shapelet_function(x, y, amp, n_max, beta, center_x=0.0, center_y=0.0):
    """Evaluate shapelet set: f(x,y) = sum_i amp[i] * phi_{n1}(x/b) * phi_{n2}(y/b).

    :param x: 1D array of x-coordinates
    :param y: 1D array of y-coordinates
    :param amp: coefficient array of length (n_max+1)*(n_max+2)/2
    :param n_max: maximum polynomial order
    :param beta: shapelet scale parameter
    :return: 1D array of surface brightness values
    """
    H_x, H_y = pre_calc_shapelets(x, y, beta, n_max, center_x, center_y)
    f = np.zeros(len(x))
    for i, n1, n2 in iterate_n1_n2(n_max):
        f += amp[i] * H_x[n1] * H_y[n2]
    return np.nan_to_num(f)


def shapelet_decomposition(image_1d, x, y, n_max, beta, deltaPix, center_x=0.0, center_y=0.0):
    """Project image onto shapelet basis to get coefficients.

    Uses orthonormality: c_i = (deltaPix^2 / beta^2) * sum_k I_k * B_i(x_k, y_k).

    :param image_1d: flattened image array
    :param x: 1D coordinate array
    :param y: 1D coordinate array
    :param n_max: maximum polynomial order
    :param beta: shapelet scale parameter
    :param deltaPix: pixel size
    :return: coefficient array
    """
    num_param = (n_max + 1) * (n_max + 2) // 2
    params = np.zeros(num_param)
    amp_norm = deltaPix**2 / beta**2
    H_x, H_y = pre_calc_shapelets(x, y, beta, n_max, center_x, center_y)
    for i, n1, n2 in iterate_n1_n2(n_max):
        base = amp_norm * H_x[n1] * H_y[n2]
        params[i] = np.sum(image_1d * base)
    return params


def shapelet_basis_list(x, y, n_max, beta, center_x=0.0, center_y=0.0):
    """Return list of individual basis function responses (amp=1 each).

    :return: list of 1D arrays, one per basis function
    """
    H_x, H_y = pre_calc_shapelets(x, y, beta, n_max, center_x, center_y)
    basis = []
    for _, n1, n2 in iterate_n1_n2(n_max):
        basis.append(np.nan_to_num(H_x[n1] * H_y[n2]))
    return basis


# ============================================================
# Gravitational Lens Models
# ============================================================

def ellipticity2phi_q(e1, e2):
    """Convert ellipticity components (e1, e2) to position angle and axis ratio.

    :return: (phi, q) where phi is the angle and q is axis ratio (0 < q <= 1)
    """
    phi = np.arctan2(e2, e1) / 2.0
    c = min(np.sqrt(e1**2 + e2**2), 0.9999)
    q = (1.0 - c) / (1.0 + c)
    return phi, q


def spep_deflection(x, y, theta_E, gamma, e1, e2, center_x=0.0, center_y=0.0):
    """Softened Power-law Elliptical Potential (SPEP) deflection angles.

    For e1=e2=0 and gamma=2, reduces to the Singular Isothermal Sphere:
    alpha = theta_E * (x, y) / r.

    Based on Barkana (1998).

    :return: (alpha_x, alpha_y) deflection angle arrays
    """
    phi_G, q = ellipticity2phi_q(e1, e2)
    gamma = max(1.4, min(2.9, float(gamma)))
    q = max(0.01, q)
    phi_E = theta_E * q
    E = phi_E / (((3.0 - gamma) / 2.0) ** (1.0 / (1.0 - gamma)) * np.sqrt(q))
    eta = 3.0 - gamma

    cos_phi, sin_phi = np.cos(phi_G), np.sin(phi_G)
    dx = x - center_x
    dy = y - center_y
    xt1 = cos_phi * dx + sin_phi * dy
    xt2 = -sin_phi * dx + cos_phi * dy
    xt2dq2 = xt2 / (q * q)
    P2 = xt1 * xt1 + xt2 * xt2dq2
    a = np.where(P2 > 0, P2, 1e-6) if hasattr(P2, '__len__') else max(P2, 1e-6)

    fac = (1.0 / eta) * (a / (E * E)) ** (eta / 2.0 - 1.0) * 2.0
    fx_p = fac * xt1
    fy_p = fac * xt2dq2
    return cos_phi * fx_p - sin_phi * fy_p, sin_phi * fx_p + cos_phi * fy_p


def shear_deflection(x, y, gamma1, gamma2):
    """External shear deflection angles.

    :return: (alpha_x, alpha_y) deflection angle arrays
    """
    return gamma1 * x + gamma2 * y, gamma2 * x - gamma1 * y


def ray_shoot(x, y, kwargs_spemd, kwargs_shear):
    """Backward ray-tracing: x_src = x_img - alpha(x_img, y_img).

    :return: (x_source, y_source) source-plane coordinate arrays
    """
    ax1, ay1 = spep_deflection(x, y, **kwargs_spemd)
    ax2, ay2 = shear_deflection(x, y, kwargs_shear['gamma1'], kwargs_shear['gamma2'])
    return x - ax1 - ax2, y - ay1 - ay2


# ============================================================
# PSF Convolution
# ============================================================

def fwhm2sigma(fwhm):
    """Convert full-width-half-maximum to Gaussian sigma."""
    return fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def gaussian_convolve(image, fwhm, pixel_size, truncation=5):
    """Convolve 2D image with Gaussian PSF.

    Uses scipy.ndimage.gaussian_filter which preserves total sum.

    :param image: 2D image array
    :param fwhm: FWHM of Gaussian PSF in angular units
    :param pixel_size: pixel size in same angular units as fwhm
    :param truncation: truncation radius in units of sigma
    :return: convolved 2D image
    """
    if fwhm <= 0:
        return image.copy()
    sigma_pix = fwhm2sigma(fwhm) / pixel_size
    return scipy.ndimage.gaussian_filter(image, sigma_pix, mode='constant', truncate=truncation)


# ============================================================
# Image Simulation Pipeline
# ============================================================

def simulate_image(numPix, deltaPix, supersampling_factor, fwhm, kwargs_source,
                   kwargs_spemd=None, kwargs_shear=None, apply_lens=True, apply_psf=True):
    """Full imaging pipeline: supersampled grid -> ray-trace -> source eval -> convolve -> downsample.

    Returns flux per pixel (surface_brightness x pixel_area).

    :param numPix: detector pixel count per axis
    :param deltaPix: detector pixel size (arcsec)
    :param supersampling_factor: sub-pixel oversampling factor
    :param fwhm: PSF FWHM (arcsec), 0 for no PSF
    :param kwargs_source: dict with keys n_max, beta, amp, center_x, center_y
    :param kwargs_spemd: SPEP lens parameters (None to skip lensing)
    :param kwargs_shear: shear parameters (None to skip)
    :param apply_lens: whether to apply gravitational lensing
    :param apply_psf: whether to apply PSF convolution
    :return: 2D image array of shape (numPix, numPix)
    """
    n_super = numPix * supersampling_factor
    dpix_super = deltaPix / supersampling_factor

    x_grid, y_grid = make_grid(n_super, dpix_super)

    if apply_lens and kwargs_spemd is not None:
        x_src, y_src = ray_shoot(x_grid, y_grid, kwargs_spemd, kwargs_shear)
    else:
        x_src, y_src = x_grid, y_grid

    src = kwargs_source
    flux = shapelet_function(x_src, y_src, src['amp'], src['n_max'], src['beta'],
                             src['center_x'], src['center_y'])
    image = array2image(flux)

    if apply_psf and fwhm > 0:
        image = gaussian_convolve(image, fwhm, dpix_super, truncation=5)

    if supersampling_factor > 1:
        image = re_size(image, supersampling_factor)

    image *= deltaPix**2
    return image


def add_poisson_noise(image, exp_time):
    """Gaussian approximation of Poisson photon-counting noise."""
    sigma = np.sqrt(np.abs(image) / exp_time)
    return np.random.randn(*image.shape) * sigma


def add_background_noise(image, sigma_bkg):
    """Gaussian background noise."""
    return np.random.randn(*image.shape) * sigma_bkg
